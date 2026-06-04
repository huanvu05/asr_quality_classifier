import os
import json
import joblib
import datetime
import warnings
import re
import string

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import pipeline
import jiwer

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CẤU HÌNH PATH & HYPERPARAMETERS
# ==========================================
class Config:
    CSV_PATH = "data/labels.csv"
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "data/transcripts/training.csv"
        
    AUDIO_ROOT = "data/audio/data2"
    MODELS_DIR = "models"
    INTERMEDIATE_CSV = os.path.join(MODELS_DIR, "asr_extracted_features.csv")
    
    # ASR Configuration
    ASR_MODEL = "vinai/PhoWhisper-small" # Rất tốt cho Tiếng Việt
    SAMPLE_RATE = 16000
    
    # XGBoost Configuration
    NUM_FOLDS = 5
    SEED = 42

os.makedirs(Config.MODELS_DIR, exist_ok=True)

# Lựa chọn thiết bị chạy ASR
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[*] ASR Target Device: {DEVICE}")

# ==========================================
# 2. HÀM TIỀN XỬ LÝ TEXT
# ==========================================
def clean_text(text: str) -> str:
    """Viết thường, xóa dấu câu và khoảng trắng thừa."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. GIAI ĐOẠN 1: ASR DECODING & METRICS
# ==========================================
def extract_physical_features(y, sr):
    """Trích xuất Duration, SNR, Silence Ratio"""
    if len(y) == 0:
        return 0.0, 0.0, 1.0
        
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Silence Ratio
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    non_silent_duration = sum([(end - start) / sr for start, end in non_silent_intervals])
    silence_ratio = max(0.0, duration - non_silent_duration) / duration if duration > 0 else 1.0
    
    # SNR Estimate
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    peak = np.max(rms_db)
    noise_floor = np.percentile(rms_db, 10)
    snr = float(np.mean(rms_db - noise_floor)) if peak > noise_floor else 0.0
    
    return float(duration), float(snr), float(silence_ratio)

def run_phase_1_asr_extraction(csv_path: str, audio_root: str):
    if os.path.exists(Config.INTERMEDIATE_CSV):
        print(f"\n[*] Đã tìm thấy file trung gian {Config.INTERMEDIATE_CSV}. Bỏ qua Giai đoạn 1 (ASR).")
        return pd.read_csv(Config.INTERMEDIATE_CSV)
        
    print(f"\n{'='*40}")
    print("GIAI ĐOẠN 1: ASR DECODING & ĐO LƯỜNG LỖI")
    print(f"{'='*40}")
    
    df = pd.read_csv(csv_path)
    
    # Chuẩn hóa nhãn
    if 'label_text' in df.columns:
        df['target'] = df['label_text'].apply(lambda x: 1 if str(x).lower().strip() == 'usable' else 0)
    elif 'label' in df.columns:
        df['target'] = df['label'].apply(lambda x: 1 if str(x) == '1' else 0)
        
    df['transcript'] = df['transcript'].fillna("")

    # Map file paths
    file_path_map = {}
    for root, dirs, files in os.walk(audio_root):
        for f in files:
            if f.endswith('.wav'):
                file_path_map[f] = os.path.join(root, f)
                
    if not file_path_map: # Tự tìm thư mục audio rộng hơn nếu trượt
        for root, dirs, files in os.walk("data/audio"):
             for f in files:
                if f.endswith('.wav'):
                    file_path_map[f] = os.path.join(root, f)

    from concurrent.futures import ThreadPoolExecutor

    # Lọc dữ liệu hợp lệ
    valid_data = []
    for idx, row in df.iterrows():
        fname = str(row['file_name'])
        if fname in file_path_map:
            valid_data.append((row, file_path_map[fname]))

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"[*] Kích hoạt giải mã ASR song song trên {num_gpus} GPUs!")
        pipelines = []
        for i in range(num_gpus):
            pipelines.append(pipeline(
                "automatic-speech-recognition", 
                model=Config.ASR_MODEL, 
                device=f"cuda:{i}",
                chunk_length_s=30,
            ))
    else:
        print(f"[*] Tải mô hình ASR: {Config.ASR_MODEL} trên {DEVICE}...")
        pipelines = [pipeline(
            "automatic-speech-recognition", 
            model=Config.ASR_MODEL, 
            device=DEVICE,
            chunk_length_s=30,
        )]

    # Hàm xử lý một phần dữ liệu cho từng GPU
    def process_chunk(chunk_data, pipe):
        chunk_results = []
        # Chạy pipeline qua một list các đường dẫn để tận dụng batch_size
        audio_paths = [abs_path for _, abs_path in chunk_data]
        
        # Pipeline generator
        dataset_out = pipe(audio_paths, batch_size=8)
        
        for (row, abs_path), out in zip(chunk_data, dataset_out):
            ground_truth = clean_text(row['transcript'])
            predicted_text = ""
            duration, snr, silence_ratio = 0.0, 0.0, 1.0
            wer_score, cer_score = 1.0, 1.0
            
            try:
                predicted_text = clean_text(out["text"])
                
                if len(ground_truth) > 0 and len(predicted_text) > 0:
                    wer_score = jiwer.wer(ground_truth, predicted_text)
                    cer_score = jiwer.cer(ground_truth, predicted_text)
                elif len(ground_truth) == 0 and len(predicted_text) == 0:
                    wer_score, cer_score = 0.0, 0.0
                     
                y, sr = librosa.load(abs_path, sr=Config.SAMPLE_RATE, mono=True)
                duration, snr, silence_ratio = extract_physical_features(y, sr)
            except Exception as e:
                pass
                
            row_res = row.to_dict()
            row_res['predicted_text'] = predicted_text
            row_res['clean_transcript'] = ground_truth
            row_res['wer_score'] = wer_score
            row_res['cer_score'] = cer_score
            row_res['audio_duration'] = duration
            row_res['snr_estimate'] = snr
            row_res['silence_ratio'] = silence_ratio
            
            chunk_results.append(row_res)
            
        return chunk_results

    print("[*] Bắt đầu giải mã ASR và tính toán Đặc trưng...")
    results = []
    
    # Chia dữ liệu cho các GPU
    chunk_size = int(np.ceil(len(valid_data) / len(pipelines)))
    chunks = [valid_data[i:i + chunk_size] for i in range(0, len(valid_data), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
        futures = [executor.submit(process_chunk, chunk, pipelines[i]) for i, chunk in enumerate(chunks)]
        
        # Có thể dùng một progress bar chung nếu cần, nhưng để đơn giản ta hiển thị tiến trình hoàn thành của các chunk lớn
        for future in tqdm(futures, total=len(futures), desc="Processing GPU Chunks"):
            results.extend(future.result())

    results_df = pd.DataFrame(results)
    results_df.to_csv(Config.INTERMEDIATE_CSV, index=False)
    print(f"\n[OK] Đã lưu file trung gian tại: {Config.INTERMEDIATE_CSV}")
    return results_df

# ==========================================
# 4. GIAI ĐOẠN 2: XGBOOST TABULAR MODELING
# ==========================================
def run_phase_2_xgboost(df: pd.DataFrame):
    print(f"\n{'='*40}")
    print("GIAI ĐOẠN 2: XGBOOST MODELING & DATA-CENTRIC")
    print(f"{'='*40}")
    
    # Định nghĩa Features
    FEATURE_COLS = ['wer_score', 'cer_score', 'audio_duration', 'snr_estimate', 'silence_ratio']
    
    # Kiểm tra NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    
    X = df[FEATURE_COLS].values
    y = df['target'].values
    groups = df['transcript'].fillna("unknown").values
    
    # --- TÍNH TOÁN DUAL WEIGHTING ---
    print("[*] Đang tính toán Trọng số kép (Dual Sample Weighting)...")
    group_stats = df.groupby('transcript')['target'].agg(['mean'])
    consensus_weights = []
    
    for transcript in groups:
        if transcript in group_stats.index:
            ratio = group_stats.loc[transcript, 'mean']
            if ratio == 1.0 or ratio == 0.0:
                consensus_weights.append(1.0)
            else:
                consensus_weights.append(0.3)
        else:
            consensus_weights.append(0.3)
            
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    sample_weights = np.array(consensus_weights) * np.array([class_weight_dict[lbl] for lbl in y])
    
    # --- HUẤN LUYỆN GROUP K-FOLD ---
    print("\n[*] Khởi chạy StratifiedGroupKFold (n_splits=5)...")
    sgkf = StratifiedGroupKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    
    oof_probs = np.zeros(len(y))
    oof_targets = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"    -> Đang huấn luyện Fold {fold+1}/{Config.NUM_FOLDS}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        
        # XGBoost Cấu hình cho Dữ liệu bảng (Ít features)
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4, # Cây nông để chống overfit
            learning_rate=0.05,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, sample_weight=w_train)
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_targets[val_idx] = y_val

    # --- TỐI ƯU NGƯỠNG ---
    print("\n[*] Tối ưu hóa Decision Threshold trên Out-of-Fold (OOF)...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0.0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (oof_probs >= thresh).astype(int)
        score = f1_score(oof_targets, y_pred, average='macro')
        if score > best_f1:
            best_f1 = score
            optimal_threshold = thresh

    print(f"    -> Optimal Threshold: {optimal_threshold:.2f} (Macro F1 = {best_f1:.4f})")
    
    y_pred_final = (oof_probs >= optimal_threshold).astype(int)
    print("\n" + "="*40)
    print("BÁO CÁO PHÂN LOẠI (OOF CLASSIFICATION REPORT)")
    print("="*40)
    print(classification_report(oof_targets, y_pred_final))
    
    overall_auc = roc_auc_score(oof_targets, oof_probs)
    print(f"ROC-AUC: {overall_auc:.4f}")
    
    # --- XUẤT ARTIFACTS ---
    print("\n[*] Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu...")
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, tree_method='hist', random_state=42, n_jobs=-1
    )
    final_model.fit(X, y, sample_weight=sample_weights)
    
    joblib.dump({"model": final_model, "optimal_threshold": float(optimal_threshold)}, 
                os.path.join(Config.MODELS_DIR, "asr_xgboost_pipeline.pkl"))
                
    # Biểu đồ mức độ quan trọng đặc trưng (Feature Importance)
    plt.figure(figsize=(10, 6))
    importances = final_model.feature_importances_
    sns.barplot(x=importances, y=FEATURE_COLS)
    plt.title('Feature Importance (ASR Alignment + Acoustic)')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODELS_DIR, "feature_importance.png"))
    plt.close()

    cm = confusion_matrix(oof_targets, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unusable(0)', 'Usable(1)'], yticklabels=['Unusable(0)', 'Usable(1)'])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'OOF Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.savefig(os.path.join(Config.MODELS_DIR, "confusion_matrix.png"))
    plt.close()

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": "ASR_Alignment_XGBoost",
        "features": FEATURE_COLS,
        "metrics": {
            "oof_macro_f1": float(best_f1),
            "oof_roc_auc": float(overall_auc),
            "optimal_threshold": float(optimal_threshold)
        }
    }
    with open(os.path.join(Config.MODELS_DIR, "asr_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Xuất Artifacts thành công vào thư mục: {Config.MODELS_DIR}")

if __name__ == "__main__":
    try:
        # Phase 1
        df_extracted = run_phase_1_asr_extraction(Config.CSV_PATH, Config.AUDIO_ROOT)
        
        # Phase 2
        run_phase_2_xgboost(df_extracted)
        
    except Exception as e:
        print(f"\n[!] LỖI NGHIÊM TRỌNG: {e}")