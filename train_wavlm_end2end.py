import os
import json
import joblib
import datetime
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CẤU HÌNH & HYPERPARAMETERS
# ==========================================
class Config:
    CSV_PATH = "data/labels.csv"
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "data/transcripts/training.csv"
    
    AUDIO_ROOT = "data/audio/data2"
    MODELS_DIR = "models"
    
    MODEL_NAME = "microsoft/wavlm-base-plus"
    SAMPLE_RATE = 16000
    MAX_DURATION_SEC = 5.0 # Cắt/Đệm về 5 giây
    MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SEC)
    
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4
    EPOCHS = 10
    LR = 2e-5
    NUM_FOLDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

os.makedirs(Config.MODELS_DIR, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & ĐÁNH TRỌNG SỐ
# ==========================================
def load_and_weight_data(csv_path: str, audio_root: str):
    print(f"\n[*] Đang đọc CSV từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Chuẩn hóa nhãn
    if 'label_text' in df.columns:
        df['target'] = df['label_text'].apply(lambda x: 1 if str(x).lower().strip() == 'usable' else 0)
    elif 'label' in df.columns:
        df['target'] = df['label'].apply(lambda x: 1 if str(x) == '1' else 0)
    else:
        raise ValueError("Không tìm thấy cột 'label_text' hoặc 'label'.")

    # Điền giá trị rỗng cho transcript
    df['transcript'] = df['transcript'].fillna("unknown_transcript")

    print("[*] Đang quét cấu trúc thư mục audio...")
    file_path_map = {}
    for root, dirs, files in os.walk(audio_root):
        for f in files:
            if f.endswith('.wav'):
                file_path_map[f] = os.path.join(root, f)
                
    if not file_path_map:
        for root, dirs, files in os.walk("data/audio"):
             for f in files:
                if f.endswith('.wav'):
                    file_path_map[f] = os.path.join(root, f)

    # Lọc dữ liệu hợp lệ
    valid_rows = []
    for idx, row in df.iterrows():
        fname = str(row['file_name'])
        if fname in file_path_map:
            row_dict = row.to_dict()
            row_dict['abs_path'] = file_path_map[fname]
            valid_rows.append(row_dict)
            
    valid_df = pd.DataFrame(valid_rows)
    print(f"[*] Tìm thấy {len(valid_df)} file âm thanh hợp lệ trên tổng số {len(df)} dòng CSV.")

    if len(valid_df) == 0:
        raise ValueError("Không tìm thấy file audio nào khớp với CSV.")

    print("[*] Đang tính toán Trọng số kép (Dual Sample Weighting)...")
    
    # 2.1. Trọng số đồng thuận
    group_stats = valid_df.groupby('transcript')['target'].agg(['mean'])
    consensus_weights = []
    clean_count = 0
    conflict_count = 0
    
    for _, row in valid_df.iterrows():
        ratio = group_stats.loc[row['transcript'], 'mean']
        if ratio == 1.0 or ratio == 0.0:
            consensus_weights.append(1.0)
            clean_count += 1
        else:
            consensus_weights.append(0.3)
            conflict_count += 1
            
    valid_df['consensus_weight'] = consensus_weights
    print(f"    - Mẫu Đồng thuận (W=1.0): {clean_count}")
    print(f"    - Mẫu Tranh cãi (W=0.3): {conflict_count}")
    
    # 2.2. Trọng số Mất cân bằng lớp
    y = valid_df['target'].values
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    valid_df['class_weight'] = valid_df['target'].map(class_weight_dict)
    print(f"    - Trọng số bù đắp (Class Weights): {class_weight_dict}")
    
    # 2.3. Gộp trọng số
    valid_df['sample_weight'] = valid_df['consensus_weight'] * valid_df['class_weight']
    
    return valid_df

# ==========================================
# 3. DATASET & DATALOADER
# ==========================================
class WavLMAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['abs_path']
        target = row['target']
        weight = row['sample_weight']
        
        try:
            # Load and resample
            y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
            
            # Truncate or Pad to fixed length
            if len(y) > Config.MAX_SAMPLES:
                y = y[:Config.MAX_SAMPLES]
            else:
                y = np.pad(y, (0, Config.MAX_SAMPLES - len(y)), mode='constant')
                
            # Trả về numpy array mộc, DataLoader/Collate sẽ lo việc đưa vào mô hình
            return {
                "audio": y,
                "target": torch.tensor(target, dtype=torch.float32),
                "weight": torch.tensor(weight, dtype=torch.float32)
            }
        except Exception as e:
            # Fallback tensor 0 nếu file lỗi
            return {
                "audio": np.zeros(Config.MAX_SAMPLES, dtype=np.float32),
                "target": torch.tensor(target, dtype=torch.float32),
                "weight": torch.tensor(0.0, dtype=torch.float32) # Không phạt model nếu file hỏng
            }

def collate_fn_wavlm(batch, feature_extractor):
    audios = [item["audio"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    weights = torch.stack([item["weight"] for item in batch])
    
    # Sử dụng feature_extractor để xử lý list of audios thành Tensor chuẩn
    inputs = feature_extractor(
        audios, 
        sampling_rate=Config.SAMPLE_RATE, 
        return_tensors="pt", 
        padding=True
    )
    
    return inputs.input_values, targets, weights

# ==========================================
# 4. KIẾN TRÚC MÔ HÌNH (WAVLM + CLASSIFICATION HEAD)
# ==========================================
class WavLMClassifier(nn.Module):
    def __init__(self):
        super(WavLMClassifier, self).__init__()
        print(f"[*] Khởi tạo mô hình WavLM: {Config.MODEL_NAME}")
        self.wavlm = WavLMModel.from_pretrained(Config.MODEL_NAME)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1) # Raw logit
        )
        
    def forward(self, input_values):
        # input_values shape: (Batch, Sequence_Length)
        outputs = self.wavlm(input_values)
        
        # WavLM trả về last_hidden_state có shape: (Batch, Frames, 768)
        hidden_states = outputs.last_hidden_state
        
        # Mean Pooling dọc theo trục thời gian (Time dimension = 1) -> (Batch, 768)
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Đưa qua Classification Head -> (Batch, 1)
        logits = self.head(pooled_output)
        
        return logits.squeeze(-1) # -> (Batch)

# ==========================================
# 5 & 6. TRAINING LOOP & EVALUATION
# ==========================================
def train_and_evaluate(valid_df: pd.DataFrame):
    print("\n[*] Khởi chạy StratifiedGroupKFold (n_splits=5)...")
    sgkf = StratifiedGroupKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    
    X = valid_df['abs_path'].values
    y = valid_df['target'].values
    groups = valid_df['transcript'].values
    
    oof_probs = np.zeros(len(y))
    oof_targets = np.zeros(len(y))
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.MODEL_NAME)
    
    # Bọc collate_fn với feature_extractor
    from functools import partial
    collate_wrapper = partial(collate_fn_wavlm, feature_extractor=feature_extractor)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n{'='*40}")
        print(f"BẮT ĐẦU FOLD {fold+1}/{Config.NUM_FOLDS}")
        print(f"{'='*40}")
        
        train_df = valid_df.iloc[train_idx]
        val_df = valid_df.iloc[val_idx]
        
        train_dataset = WavLMAudioDataset(train_df, feature_extractor)
        val_dataset = WavLMAudioDataset(val_df, feature_extractor)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_wrapper, num_workers=0)
        
        model = WavLMClassifier().to(Config.DEVICE)
        if torch.cuda.device_count() > 1 and Config.DEVICE == "cuda":
             print(f"🚀 Kích hoạt DataParallel trên {torch.cuda.device_count()} GPUs!")
             model = nn.DataParallel(model)
        
        # Custom Loss với Data-Centric Penalty
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        best_val_f1 = 0.0
        best_model_path = os.path.join(Config.MODELS_DIR, f"best_wavlm_fold_{fold+1}.pth")
        
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for step, (inputs, targets, weights) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
                inputs, targets, weights = inputs.to(Config.DEVICE), targets.to(Config.DEVICE), weights.to(Config.DEVICE)
                
                logits = model(inputs)
                
                # Tính loss từng mẫu, nhân trọng số, rồi mới tính trung bình
                raw_loss = criterion(logits, targets)
                weighted_loss = (raw_loss * weights).mean()
                
                # Gradient Accumulation
                weighted_loss = weighted_loss / Config.ACCUMULATION_STEPS
                weighted_loss.backward()
                
                if (step + 1) % Config.ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                train_loss += weighted_loss.item() * Config.ACCUMULATION_STEPS
                
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_probs = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
                    inputs = inputs.to(Config.DEVICE)
                    logits = model(inputs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                    val_probs.extend(probs)
                    val_targets.extend(targets.numpy())
            
            val_probs = np.array(val_probs)
            val_targets = np.array(val_targets)
            
            # Tính F1 bằng ngưỡng mặc định 0.5 để theo dõi trong lúc train
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro F1 (Thresh=0.5): {val_f1:.4f}")
            
            scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), best_model_path)
                print(f"🌟 Saved New Best Model (F1: {val_f1:.4f})")
                
        # --- Lưu kết quả OOF của Fold này ---
        print(f"\n[*] Đang đánh giá OOF cho Fold {fold+1} bằng mô hình tốt nhất...")
        # Load best model to predict OOF
        best_model = WavLMClassifier().to(Config.DEVICE)
        best_model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        if torch.cuda.device_count() > 1 and Config.DEVICE == "cuda":
            best_model = nn.DataParallel(best_model)
        best_model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc=f"Predicting OOF Fold {fold+1}"):
                inputs = inputs.to(Config.DEVICE)
                logits = best_model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)
                
        oof_probs[val_idx] = fold_probs
        oof_targets[val_idx] = y_val
        
        # Chỉ chạy 1 Fold trong quá trình Test. Hãy bỏ break nếu muốn chạy full 5 Folds.
        print("\n[!] Dừng sau Fold 1 để tiết kiệm thời gian (Phục vụ Test Pipeline). Bỏ lệnh break trong code để chạy full.")
        break
        
    # --- 6. Tối ưu hóa Ngưỡng (Threshold Sweeping) ---
    print("\n[*] Tối ưu hóa Decision Threshold trên Out-of-Fold (OOF)...")
    # Lọc những mẫu đã được chạy qua Validation (Nếu break ở Fold 1 thì mảng có một số phần tử bằng 0 chưa được chạy)
    # Tìm các index đã được predict (oof_probs khác 0, hoặc có thể check qua vòng lặp)
    # Để an toàn cho 1 fold break, ta chỉ tính threshold trên tập validation của fold 1
    
    valid_mask = oof_probs > 0 # Trick để lọc ra tập val của Fold 1
    final_y_true = oof_targets[valid_mask]
    final_y_probs = oof_probs[valid_mask]
    
    if len(final_y_true) == 0:
        print("Lỗi: Không có dự đoán OOF nào được thực hiện.")
        return
        
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1_overall = 0.0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (final_y_probs >= thresh).astype(int)
        score = f1_score(final_y_true, y_pred, average='macro')
        if score > best_f1_overall:
            best_f1_overall = score
            optimal_threshold = thresh

    print(f"    -> Optimal Threshold: {optimal_threshold:.2f} (Macro F1 = {best_f1_overall:.4f})")
    
    y_pred_final = (final_y_probs >= optimal_threshold).astype(int)
    print("\n" + "="*40)
    print("BÁO CÁO PHÂN LOẠI (OOF CLASSIFICATION REPORT)")
    print("="*40)
    print(classification_report(final_y_true, y_pred_final))
    
    overall_auc = roc_auc_score(final_y_true, final_y_probs)
    print(f"ROC-AUC: {overall_auc:.4f}")
    
    # Export Metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": f"End-to-End {Config.MODEL_NAME}",
        "metrics": {
            "oof_macro_f1": float(best_f1_overall),
            "oof_roc_auc": float(overall_auc),
            "optimal_threshold": float(optimal_threshold)
        }
    }
    meta_path = os.path.join(Config.MODELS_DIR, "wavlm_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Xuất Artifacts thành công vào thư mục: {Config.MODELS_DIR}")

if __name__ == "__main__":
    try:
        valid_df = load_and_weight_data(Config.CSV_PATH, Config.AUDIO_ROOT)
        train_and_evaluate(valid_df)
    except Exception as e:
         print(f"\n[!] LỖI NGHIÊM TRỌNG: {e}")