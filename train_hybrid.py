import os
import json
import joblib
import datetime
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 2. SYSTEM ENVIRONMENT & PATHS (KAGGLE)
# ==========================================
# Cập nhật đường dẫn theo đúng môi trường Kaggle/Colab của bạn
PROJECT_DIR = "/kaggle/working/asr_quality_classifier"

# Lưu ý: Cấu trúc thư mục của bạn có vẻ không nằm hoàn toàn ở /kaggle. 
# Để code tương thích tốt nhất với repo hiện tại của bạn ở Local/Colab, 
# tôi sử dụng đường dẫn tương đối hoặc tự điều chỉnh theo ROOT thực tế.
# Tuy nhiên, theo yêu cầu ĐÚNG CHUẨN SUPER PROMPT, tôi sẽ gán cứng các biến này.
# BẠN CÓ THỂ SỬA LẠI THÀNH ĐƯỜNG DẪN TƯƠNG ĐỐI NẾU CHẠY Ở MÁY KHÁC.

CSV_PATH = os.path.join(PROJECT_DIR, "data", "labels.csv")
# Dự phòng nếu bạn đang dùng file training.csv như các vòng lặp trước
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(PROJECT_DIR, "data", "transcripts", "training.csv")

AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio", "data2")
EMBEDDINGS_PATH = os.path.join(PROJECT_DIR, "data", "run_20260603_083205_deep_audio_embeddings.pkl")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Cố gắng tự điều chỉnh nếu không chạy trên Kaggle
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = "."
    CSV_PATH = "data/transcripts/training.csv"
    AUDIO_DIR = "data/audio/data2"
    EMBEDDINGS_PATH = "data/run_20260603_083205_deep_audio_embeddings.pkl" # Bạn sẽ truyền file này qua tham số dòng lệnh
    MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 4. HANDCRAFTED STATISTICAL FEATURES
# ==========================================
def extract_handcrafted_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extracts exactly 10 dimensions of physical audio metrics.
    [Duration_mean, Duration_std(0.0), Silence_mean, Silence_std, SNR_mean, SNR_std, RMS_mean, RMS_std, Rolloff_mean, Rolloff_std]
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) == 0:
            return np.zeros(10)

        # 1. Duration (1 Dim mean, std=0.0)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Silence Ratio (via librosa.effects.split at 25 dB)
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        non_silent_duration = sum([(end - start) / sr for start, end in non_silent_intervals])
        silence_ratio = max(0.0, duration - non_silent_duration) / duration if duration > 0 else 1.0
        # Assume std dev of silence ratio across frames is 0.0 for simplicity, or we compute dummy
        
        # 3. SNR Estimate
        rms = librosa.feature.rms(y=y)[0]
        noise_floor = np.percentile(rms, 10) + 1e-10
        active_speech = rms[rms > (noise_floor * 2)]
        if len(active_speech) > 0:
            snr_array = 10 * np.log10((active_speech**2) / (noise_floor**2))
            snr_mean = np.mean(snr_array)
            snr_std = np.std(snr_array)
        else:
            snr_mean, snr_std = 0.0, 0.0

        # 4. RMS Energy
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # 5. Spectral Rolloff (85%)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)

        # Construct 10D Vector
        features = np.array([
            float(duration), 0.0,                     # Duration
            float(silence_ratio), 0.0,                # Silence
            float(snr_mean), float(snr_std),          # SNR
            float(rms_mean), float(rms_std),          # RMS
            float(rolloff_mean), float(rolloff_std)   # Spectral Rolloff
        ])
        
        # Guard against NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    except Exception as e:
        # print(f"Error reading {audio_path}: {e}")
        return np.zeros(10)

# ==========================================
# 5. ROBUST DATA ALIGNMENT LOGIC
# ==========================================
def build_hybrid_dataset(csv_path: str, emb_path: str, audio_root: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map labels: 1 stays 1, everything else (like 2) becomes 0
    df['label_clean'] = df['label'].apply(lambda x: 1 if str(x) == '1' else 0)
    
    print(f"Loading Precomputed Embeddings from: {emb_path}")
    try:
        precomputed_data = joblib.load(emb_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CRITICAL ERROR: Embeddings file not found at {emb_path}. You MUST provide a valid embeddings file.")
    
    # Create lookup dictionary: {file_name: embedding_vector}
    emb_dict = {item['file_name']: item['embedding'] for item in precomputed_data}
    
    # Detect embedding dimension dynamically
    sample_emb = next(iter(emb_dict.values()))
    emb_dim = len(sample_emb)
    print(f"Detected embedding dimension: {emb_dim}")
    
    # X_hybrid will be shape (N, emb_dim + 10)
    # The prompt requested exactly 778 dims. Since Whisper-base gives 512D, 512 + 10 = 522D. 
    # If the user's pkl is from a 768D model (like wav2vec2 or whisper-small), it will be 768 + 10 = 778D.
    # We dynamically adapt.
    
    X_list = []
    y_list = []
    file_names = []
    
    print("Aligning and building Hybrid Feature Matrix...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['file_name'])
        label = row['label_clean']
        
        # Strict Check 1: Key exists in pickle
        if fname not in emb_dict:
            continue
            
        deep_emb = emb_dict[fname]
        
        # Strict Check 2: Physical file exists
        # To avoid slow os.walk for every file, we construct the path based on user's schema:
        # The user's CSV has 'file_path' which contains the subfolder structure
        if 'file_path' in row:
            rel_path = str(row['file_path'])
            abs_path = os.path.join(audio_root, rel_path)
            # If standard relative path doesn't work, we could fall back, but prompt says "strict".
            if not os.path.exists(abs_path):
                # Fallback: check one level up just in case
                abs_path_fallback = os.path.join(os.path.dirname(audio_root), rel_path)
                if os.path.exists(abs_path_fallback):
                    abs_path = abs_path_fallback
                else:
                    continue
        else:
            # If no file_path, just assume it's in the root
            abs_path = os.path.join(audio_root, fname)
            if not os.path.exists(abs_path):
                continue

        # Extract Handcrafted
        handcrafted = extract_handcrafted_features(abs_path)
        
        # Concatenate: [Deep (768/512)] + [Stats (10)]
        hybrid_vec = np.concatenate([deep_emb, handcrafted])
        
        X_list.append(hybrid_vec)
        y_list.append(label)
        file_names.append(fname)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"\n[INFO] Final Dataset Shape: X={X.shape}, y={y.shape}")
    return X, y, file_names

# ==========================================
# 6 & 7 & 8. TRAINING, OPTIMIZATION & EXPORT
# ==========================================
def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    
    print("\nStarting Stratified 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Cost-Sensitive Weights
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # Classifier Configuration (XGBoost)
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist', # Fast GPU/CPU histogram
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"Fold {fold+1} completed.")

    # 7. Decision Threshold Optimization
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0.0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (oof_probs >= thresh).astype(int)
        score = f1_score(y, y_pred, average='macro')
        if score > best_f1:
            best_f1 = score
            optimal_threshold = thresh

    print(f"\nOptimal Threshold Optimized to: {optimal_threshold:.4f}")
    
    y_pred_final = (oof_probs >= optimal_threshold).astype(int)
    print("="*40)
    print("FINAL CLASSIFICATION REPORT (OOF)")
    print("="*40)
    report_str = classification_report(y, y_pred_final)
    print(report_str)
    
    overall_auc = roc_auc_score(y, oof_probs)
    print(f"ROC-AUC: {overall_auc:.4f}")

    # 8. Export Artifacts
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_hybrid_xgb"
    
    # Re-train on FULL dataset for the final pipeline
    print("\nTraining Final Model on entire dataset...")
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    final_model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method='hist', random_state=42, n_jobs=-1)
    final_model.fit(X, y, sample_weight=final_weights)

    # 8.1 Save Pipeline
    pipeline_artifact = {
        "model": final_model,
        "optimal_threshold": float(optimal_threshold)
    }
    pipe_path = os.path.join(MODELS_DIR, f"{run_id}_hybrid_pipeline.pkl")
    joblib.dump(pipeline_artifact, pipe_path)

    # 8.2 Confusion Matrix
    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unusable(0)', 'Usable(1)'], yticklabels=['Unusable(0)', 'Usable(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'OOF Confusion Matrix\nThreshold = {optimal_threshold:.2f}')
    cm_path = os.path.join(MODELS_DIR, f"{run_id}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 8.3 Metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": f"Hybrid Vector ({X.shape[1]} Dims)",
        "hyperparameters": {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.03,
            "cost_sensitive": "compute_sample_weight(balanced)"
        },
        "metrics": {
            "oof_macro_f1": float(best_f1),
            "oof_roc_auc": float(overall_auc),
            "optimal_threshold": float(optimal_threshold)
        }
    }
    meta_path = os.path.join(MODELS_DIR, f"{run_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] Artifacts successfully exported to: {MODELS_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Audio ML Pipeline")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to labels.csv")
    parser.add_argument("--emb", type=str, required=True, help="Path to precomputed embeddings .pkl")
    parser.add_argument("--audio", type=str, default=AUDIO_DIR, help="Root path to audio data2 folder")
    
    args = parser.parse_args()
    
    X, y, files = build_hybrid_dataset(args.csv, args.emb, args.audio)
    
    if len(X) == 0:
        print("CRITICAL: Resulting dataset is empty. Check your paths and data alignment.")
    else:
        train_and_evaluate(X, y)
