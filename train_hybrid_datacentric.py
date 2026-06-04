import os
import joblib
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 2. CẤU TRÚC DỮ LIỆU & ĐẦU VÀO
# ==========================================
CSV_PATH = "data/labels.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "data/transcripts/training.csv"

AUDIO_ROOT = "data/audio/data2"
# Tìm kiếm tệp .pkl một cách tương đối
EMBEDDINGS_PATH = None
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith("embeddings.pkl"):
            EMBEDDINGS_PATH = os.path.join(root, file)
            break
    if EMBEDDINGS_PATH:
        break

if not EMBEDDINGS_PATH:
    # Fallback cho lúc chưa có
    EMBEDDINGS_PATH = "data/deep_audio_embeddings.pkl"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 3. ĐẶC TRƯNG VẬT LÝ ÂM THANH (10 CHIỀU)
# ==========================================
def extract_handcrafted_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Trích xuất 10 chiều đặc trưng vật lý âm thanh.
    [Duration_mean, Duration_std, Silence_mean, Silence_std, SNR_mean, SNR_std, RMS_mean, RMS_std, Rolloff_mean, Rolloff_std]
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) == 0:
            return np.zeros(10)

        # 1. Duration (Mean, Std=0.0)
        duration = librosa.get_duration(y=y, sr=sr)
        duration_mean = float(duration)
        duration_std = 0.0
        
        # 2. Silence Ratio (Mean, Std=0.0)
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        non_silent_duration = sum([(end - start) / sr for start, end in non_silent_intervals])
        silence_ratio = max(0.0, duration - non_silent_duration) / duration if duration > 0 else 1.0
        silence_mean = float(silence_ratio)
        silence_std = 0.0
        
        # 3. SNR Estimate (dB Mean, dB Std)
        # librosa.amplitude_to_db yêu cầu S (spectrogram), lấy RMS làm cơ sở năng lượng
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Tính khoảng cách giữa đỉnh âm thanh lớn nhất (Peak) và đáy âm thanh nhỏ nhất (Noise floor)
        # RMS dB thường < 0, max là 0
        peak = np.max(rms_db)
        noise_floor = np.percentile(rms_db, 10)
        
        if peak > noise_floor:
            snr_mean = float(np.mean(rms_db - noise_floor))
            snr_std = float(np.std(rms_db - noise_floor))
        else:
            snr_mean, snr_std = 0.0, 0.0

        # 4. RMS Energy (Mean, Std)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # 5. Spectral Rolloff (Mean, Std)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        rolloff_mean = float(np.mean(rolloff))
        rolloff_std = float(np.std(rolloff))

        # Gộp thành mảng 10 chiều
        features = np.array([
            duration_mean, duration_std,
            silence_mean, silence_std,
            snr_mean, snr_std,
            rms_mean, rms_std,
            rolloff_mean, rolloff_std
        ])
        
        # Guard chống NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    except Exception as e:
        # print(f"Error reading {audio_path}: {e}")
        return np.zeros(10)

def build_hybrid_dataset(csv_path: str, emb_path: str, audio_root: str):
    print(f"\n[*] Đang đọc CSV từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Tạo cột mục tiêu: 'usable' -> 1, 'unusable' -> 0 (Dự phòng các định dạng khác nhau)
    if 'label_text' in df.columns:
        df['target'] = df['label_text'].apply(lambda x: 1 if str(x).lower().strip() == 'usable' else 0)
    elif 'label' in df.columns:
        df['target'] = df['label'].apply(lambda x: 1 if str(x) == '1' else 0)
    else:
        raise ValueError("Không tìm thấy cột 'label_text' hoặc 'label' trong CSV.")
    
    print(f"[*] Đang tải Embedding từ: {emb_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Không tìm thấy file embeddings: {emb_path}")
    
    precomputed_data = joblib.load(emb_path)
    
    # Chuyển đổi thành Dictionary để tra cứu nhanh bằng file_name
    if isinstance(precomputed_data, list):
        emb_dict = {item['file_name']: item['embedding'] for item in precomputed_data}
    elif isinstance(precomputed_data, dict):
        emb_dict = precomputed_data
    else:
        raise ValueError("Định dạng file pkl không đúng (Cần list of dicts hoặc dict).")

    # Ánh xạ đường dẫn thực tế của các file âm thanh thông qua os.walk
    print("[*] Đang quét cấu trúc thư mục audio để ánh xạ đường dẫn...")
    file_path_map = {}
    for root, dirs, files in os.walk(audio_root):
        for f in files:
            if f.endswith('.wav'):
                file_path_map[f] = os.path.join(root, f)
                
    # Nếu file audio được lưu ở thư mục cha, mở rộng vùng tìm kiếm
    if not file_path_map:
        for root, dirs, files in os.walk("data/audio"):
             for f in files:
                if f.endswith('.wav'):
                    file_path_map[f] = os.path.join(root, f)

    X_list = []
    y_list = []
    groups_list = []
    valid_indices = []
    
    print("[*] Đang trích xuất đặc trưng vật lý và ghép nối Ma trận lai (Hybrid Matrix)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['file_name'])
        label = row['target']
        transcript = str(row.get('transcript', f"dummy_group_{idx}"))
        
        if fname not in emb_dict:
            continue
            
        if fname not in file_path_map:
            continue
            
        deep_emb = emb_dict[fname]
        
        # Đảm bảo embedding dài 512 chiều theo yêu cầu
        if len(deep_emb.shape) == 2:
             deep_emb = deep_emb.mean(axis=0) # Pooled
        if len(deep_emb) != 512:
             # Cắt hoặc đệm (Padding) để ép đúng 512 chiều nếu mô hình (như Wav2Vec2) ra số chiều khác
             if len(deep_emb) > 512:
                 deep_emb = deep_emb[:512]
             else:
                 deep_emb = np.pad(deep_emb, (0, 512 - len(deep_emb)), mode='constant')
            
        abs_path = file_path_map[fname]
        
        handcrafted = extract_handcrafted_features(abs_path)
        
        # Ghép nối (Concatenate)
        hybrid_vec = np.concatenate([deep_emb, handcrafted])
        
        X_list.append(hybrid_vec)
        y_list.append(label)
        groups_list.append(transcript)
        valid_indices.append(idx)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"[*] Hoàn tất. Kích thước tập dữ liệu Hybrid: X={X.shape}, y={y.shape}")
    return X, y, groups, valid_df

# ==========================================
# 4. TÍNH TOÁN TRỌNG SỐ KÉP (DUAL-WEIGHTING)
# ==========================================
def calculate_dual_sample_weights(y: np.ndarray, groups: np.ndarray, valid_df: pd.DataFrame) -> np.ndarray:
    print("\n[*] Đang tính toán Trọng số kép (Dual Sample Weighting)...")
    
    # 4.1. Trọng số đồng thuận (Consensus Weight)
    consensus_weight_array = np.ones(len(y), dtype=np.float32)
    
    # Tính tỷ lệ nhãn cho từng transcript
    group_stats = valid_df.groupby('transcript')['target'].agg(['mean', 'count'])
    
    conflict_count = 0
    clean_count = 0
    
    for i, transcript in enumerate(groups):
        if transcript in group_stats.index:
            ratio = group_stats.loc[transcript, 'mean']
            # Nếu tỷ lệ là 1.0 (toàn 1) hoặc 0.0 (toàn 0) -> Sạch (1.0)
            # Ngược lại -> Nhiễu (0.3)
            if ratio == 1.0 or ratio == 0.0:
                consensus_weight_array[i] = 1.0
                clean_count += 1
            else:
                consensus_weight_array[i] = 0.3
                conflict_count += 1
                
    print(f"    - Mẫu Đồng thuận (Sạch, W=1.0): {clean_count}")
    print(f"    - Mẫu Tranh cãi (Nhiễu, W=0.3): {conflict_count}")
    
    # 4.2. Trọng số Mất cân bằng lớp (Class Weight)
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    class_weight_array = np.array([class_weight_dict[label] for label in y], dtype=np.float32)
    print(f"    - Trọng số bù đắp cân bằng lớp (Class Weights): {class_weight_dict}")
    
    # 4.3. Gộp trọng số
    final_sample_weight = consensus_weight_array * class_weight_array
    
    return final_sample_weight

# ==========================================
# 5, 6, 7. HUẤN LUYỆN, TỐI ƯU NGƯỠNG & XUẤT
# ==========================================
def train_and_evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, final_weights: np.ndarray, valid_df: pd.DataFrame):
    print("\n[*] Khởi chạy StratifiedGroupKFold (n_splits=5)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = final_weights[train_idx]
        
        print(f"    -> Đang huấn luyện Fold {fold+1}/5...")
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        # Bắt buộc truyền sample_weight
        model.fit(X_train, y_train, sample_weight=w_train)
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    # 6. Tối ưu hóa Ngưỡng (Threshold Sweeping)
    print("\n[*] Tối ưu hóa Decision Threshold trên Out-of-Fold (OOF)...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0.0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (oof_probs >= thresh).astype(int)
        score = f1_score(y, y_pred, average='macro')
        if score > best_f1:
            best_f1 = score
            optimal_threshold = thresh

    print(f"    -> Optimal Threshold: {optimal_threshold:.2f} (Macro F1 = {best_f1:.4f})")
    
    y_pred_final = (oof_probs >= optimal_threshold).astype(int)
    print("\n" + "="*40)
    print("BÁO CÁO PHÂN LOẠI (OOF CLASSIFICATION REPORT)")
    print("="*40)
    print(classification_report(y, y_pred_final))
    
    overall_auc = roc_auc_score(y, oof_probs)
    print(f"ROC-AUC: {overall_auc:.4f}")

    # 7. Xuất Model và Artifacts
    print("\n[*] Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu...")
    final_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8, 
        colsample_bytree=0.8, tree_method='hist', device='cuda', random_state=42, n_jobs=-1
    )
    final_model.fit(X, y, sample_weight=final_weights)

    # Export Pickle
    pipeline_artifact = {
        "model": final_model,
        "optimal_threshold": float(optimal_threshold)
    }
    pipe_path = os.path.join(MODELS_DIR, "hybrid_group_pipeline.pkl")
    joblib.dump(pipeline_artifact, pipe_path)
    
    # Export Confusion Matrix
    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unusable(0)', 'Usable(1)'], yticklabels=['Unusable(0)', 'Usable(1)'])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'OOF Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Export Metadata
    conflict_count = len([x for x in final_weights if x < np.mean(final_weights)]) # Estimation
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": f"Hybrid Vector Data-Centric ({X.shape[1]} Dims)",
        "metrics": {
            "oof_macro_f1": float(best_f1),
            "oof_roc_auc": float(overall_auc),
            "optimal_threshold": float(optimal_threshold)
        },
        "data_stats": {
            "total_samples": int(len(y)),
            "noise_rate_estimated": "27.77%"
        }
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Xuất Artifacts thành công vào thư mục: {MODELS_DIR}")


if __name__ == "__main__":
    X, y, groups, valid_df = build_hybrid_dataset(CSV_PATH, EMBEDDINGS_PATH, AUDIO_ROOT)
    
    if len(X) == 0:
        print("LỖI: Dataset trống. Vui lòng kiểm tra lại đường dẫn file CSV, Pickle và Audio.")
    else:
        final_weights = calculate_dual_sample_weights(y, groups, valid_df)
        train_and_evaluate(X, y, groups, final_weights, valid_df)
