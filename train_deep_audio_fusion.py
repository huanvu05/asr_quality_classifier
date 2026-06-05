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
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    TARGET_LAYER = 6 # BÍ MẬT CỐT LÕI: Lấy Layer trung gian để giữ đặc trưng âm học
    
    SAMPLE_RATE = 16000
    MAX_DURATION_SEC = 5.0
    MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SEC)
    
    BATCH_SIZE = 8 # Có thể chạy batch lớn hơn vì chúng ta đóng băng WavLM làm feature extractor
    EPOCHS = 15
    LR = 5e-4
    WEIGHT_DECAY = 1e-2
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
# 2. XỬ LÝ DỮ LIỆU & LABEL SMOOTHING
# ==========================================
def extract_handcrafted_features(y, sr):
    """Trích xuất 5 đặc trưng vật lý cho Branch 2"""
    if len(y) == 0:
        return np.zeros(5, dtype=np.float32)
        
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
    
    # RMS Energy Mean
    rms_mean = float(np.mean(rms))
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))
    
    return np.array([duration, silence_ratio, snr, rms_mean, rolloff_mean], dtype=np.float32)

def prepare_data(csv_path: str, audio_root: str):
    print(f"\n[*] Đang đọc CSV từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'label_text' in df.columns:
        df['target'] = df['label_text'].apply(lambda x: 1.0 if str(x).lower().strip() == 'usable' else 0.0)
    else:
        df['target'] = df['label'].apply(lambda x: 1.0 if str(x) == '1' else 0.0)

    df['transcript'] = df['transcript'].fillna("unknown_transcript")

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

    valid_rows = []
    print("[*] Tiền xử lý Audio (Resample, Padding) & Trích xuất đặc trưng vật lý...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['file_name'])
        if fname in file_path_map:
            row_dict = row.to_dict()
            row_dict['abs_path'] = file_path_map[fname]
            valid_rows.append(row_dict)
            
    valid_df = pd.DataFrame(valid_rows)
    
    print("[*] Áp dụng Label Smoothing cho Dữ liệu Nhiễu...")
    group_stats = valid_df.groupby('transcript')['target'].agg(['mean'])
    smoothed_targets = []
    
    for _, row in valid_df.iterrows():
        ratio = group_stats.loc[row['transcript'], 'mean']
        original_target = row['target']
        
        # Nếu transcript này có sự mâu thuẫn (nhãn 0 và 1 trộn lẫn)
        if ratio > 0.0 and ratio < 1.0:
            if original_target == 1.0:
                smoothed_targets.append(0.9) # Không tự tin 100% Usable
            else:
                smoothed_targets.append(0.1) # Không tự tin 100% Unusable
        else:
            smoothed_targets.append(original_target) # Giữ nguyên Ground Truth tuyệt đối
            
    valid_df['smoothed_target'] = smoothed_targets
    
    return valid_df

# ==========================================
# 3. DATASET
# ==========================================
class DeepFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor, scaler=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        
        # Trích xuất toàn bộ Physical features
        print(f"[*] Đang Load Physical Features ({'Train' if is_train else 'Valid'})")
        self.physical_features = []
        self.audios = []
        
        # Chỉ hiển thị progress bar, không in ra từng dòng
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), mininterval=2.0):
            try:
                y, sr = librosa.load(row['abs_path'], sr=Config.SAMPLE_RATE, mono=True)
                phys_feat = extract_handcrafted_features(y, sr)
                
                if len(y) > Config.MAX_SAMPLES:
                    y = y[:Config.MAX_SAMPLES]
                else:
                    y = np.pad(y, (0, Config.MAX_SAMPLES - len(y)), mode='constant')
            except:
                y = np.zeros(Config.MAX_SAMPLES, dtype=np.float32)
                phys_feat = np.zeros(5, dtype=np.float32)
                
            self.audios.append(y)
            self.physical_features.append(phys_feat)
            
        self.physical_features = np.array(self.physical_features)
        
        # Chuẩn hóa Physical Features bằng StandardScaler
        if is_train:
            self.scaler = StandardScaler()
            self.physical_features = self.scaler.fit_transform(self.physical_features)
        else:
            self.scaler = scaler
            self.physical_features = self.scaler.transform(self.physical_features)
            
        # Target
        self.targets = self.df['smoothed_target'].values
        # Giữ lại nhãn gốc để tính metric (F1/AUC không dùng nhãn smoothed)
        self.original_targets = self.df['target'].values
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        return {
            "audio": self.audios[idx],
            "phys_feat": torch.tensor(self.physical_features[idx], dtype=torch.float32),
            "smoothed_target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "original_target": torch.tensor(self.original_targets[idx], dtype=torch.float32)
        }

def collate_fn_fusion(batch, feature_extractor):
    audios = [item["audio"] for item in batch]
    phys_feats = torch.stack([item["phys_feat"] for item in batch])
    smoothed_targets = torch.stack([item["smoothed_target"] for item in batch])
    original_targets = torch.stack([item["original_target"] for item in batch])
    
    inputs = feature_extractor(
        audios, 
        sampling_rate=Config.SAMPLE_RATE, 
        return_tensors="pt", 
        padding=True
    )
    
    return inputs.input_values, phys_feats, smoothed_targets, original_targets

# ==========================================
# 4. KIẾN TRÚC MẠNG: DEEP FUSION NETWORK
# ==========================================
class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states):
        # hidden_states: (Batch, Seq_Len, Hidden_Size)
        attn_weights = self.attention(hidden_states) # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # (Batch, Seq_Len, Hidden_Size) * (Batch, Seq_Len, 1) -> sum qua Seq_Len
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1) # (Batch, Hidden_Size)
        return pooled_output

class DeepAudioFusionNetwork(nn.Module):
    def __init__(self):
        super(DeepAudioFusionNetwork, self).__init__()
        print(f"[*] Khởi tạo WavLM và trích xuất từ Layer {Config.TARGET_LAYER} (Intermediate)")
        
        # Load WavLM và yêu cầu trả về toàn bộ hidden states các layer
        self.wavlm = WavLMModel.from_pretrained(Config.MODEL_NAME, output_hidden_states=True)
        
        # ĐÓNG BĂNG TOÀN BỘ WAVLM (Chúng ta chỉ dùng nó như Feature Extractor tĩnh)
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        # Branch 1: Acoustic Latent Branch
        self.attention_pooling = SelfAttentionPooling(hidden_size=768)
        
        # Branch 2 & Fusion Layer
        # Nối: Branch 1 (768) + Branch 2 (Physical 5 chiều) = 773 chiều
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768 + 5, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout mạnh chống overfit
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logit output
        )
        
    def forward(self, input_values, phys_feats):
        with torch.no_grad(): # Đảm bảo hoàn toàn không flow gradient ngược về WavLM
            outputs = self.wavlm(input_values)
            # Lấy Hidden State từ Layer số 6 (index 6 của tuple hidden_states)
            # Tuple có 13 phần tử: 0 là embedding layer, 1->12 là các transformer layer
            latent_representation = outputs.hidden_states[Config.TARGET_LAYER] 
        
        # Branch 1: Self-Attention Pooling để bắt lỗi tiếng vấp, nhiễu mic
        acoustic_features = self.attention_pooling(latent_representation) # (Batch, 768)
        
        # Fusion: Nối Branch 1 và Branch 2
        fused_vector = torch.cat((acoustic_features, phys_feats), dim=1) # (Batch, 773)
        
        # Đưa qua MLP
        logits = self.fusion_mlp(fused_vector)
        
        return logits.squeeze(-1)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_fusion_model(valid_df: pd.DataFrame):
    print("\n[*] Khởi chạy StratifiedGroupKFold (n_splits=5)...")
    sgkf = StratifiedGroupKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    
    # Hack để chia KFold (Chỉ mượn y và groups, dữ liệu thật truyền qua indices)
    X_dummy = np.zeros(len(valid_df))
    y = valid_df['target'].values
    groups = valid_df['transcript'].values
    
    oof_probs = np.zeros(len(y))
    oof_targets = np.zeros(len(y))
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.MODEL_NAME)
    from functools import partial
    collate_wrapper = partial(collate_fn_fusion, feature_extractor=feature_extractor)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
        print(f"\n{'='*40}")
        print(f"BẮT ĐẦU FOLD {fold+1}/{Config.NUM_FOLDS}")
        print(f"{'='*40}")
        
        train_df = valid_df.iloc[train_idx]
        val_df = valid_df.iloc[val_idx]
        
        train_dataset = DeepFusionDataset(train_df, feature_extractor, is_train=True)
        # Tái sử dụng scaler của train cho val
        val_dataset = DeepFusionDataset(val_df, feature_extractor, scaler=train_dataset.scaler, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_wrapper, num_workers=0)
        
        model = DeepAudioFusionNetwork().to(Config.DEVICE)
        
        # Chỉ train các Layer mới thêm vào (Attention Pooling, MLP Fusion)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        
        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        # Hàm loss BCE tính trên smoothed target
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_f1 = 0.0
        best_model_path = os.path.join(Config.MODELS_DIR, f"best_fusion_fold_{fold+1}.pth")
        
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0
            
            for inputs, phys_feats, smoothed_targs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                inputs, phys_feats, smoothed_targs = inputs.to(Config.DEVICE), phys_feats.to(Config.DEVICE), smoothed_targs.to(Config.DEVICE)
                
                optimizer.zero_grad()
                logits = model(inputs, phys_feats)
                
                loss = criterion(logits, smoothed_targs)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation (Đánh giá trên Original Target, KHÔNG dùng Smoothed Target)
            model.eval()
            val_probs = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, phys_feats, _, orig_targs in val_loader:
                    inputs, phys_feats = inputs.to(Config.DEVICE), phys_feats.to(Config.DEVICE)
                    logits = model(inputs, phys_feats)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                    val_probs.extend(probs)
                    val_targets.extend(orig_targs.numpy())
            
            val_probs = np.array(val_probs)
            val_targets = np.array(val_targets)
            
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro F1 (Thresh=0.5): {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 Saved New Best Model (F1: {val_f1:.4f})")
                
        # Load best model for OOF
        print(f"[*] Đánh giá OOF cho Fold {fold+1}...")
        best_model = DeepAudioFusionNetwork().to(Config.DEVICE)
        best_model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        if torch.cuda.device_count() > 1 and Config.DEVICE == "cuda":
            best_model = nn.DataParallel(best_model)
        best_model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for inputs, phys_feats, _, _ in val_loader:
                inputs, phys_feats = inputs.to(Config.DEVICE), phys_feats.to(Config.DEVICE)
                logits = best_model(inputs, phys_feats)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)
                
        y_val = val_df['target'].values
        oof_probs[val_idx] = fold_probs
        oof_targets[val_idx] = y_val
        
        # Lưu Scaler của fold (nếu cần inference sau này)
        joblib.dump(train_dataset.scaler, os.path.join(Config.MODELS_DIR, f"scaler_fold_{fold+1}.pkl"))
        
        print("\n[!] Dừng sau Fold 1 để tiết kiệm thời gian (Phục vụ Test Pipeline). Bỏ lệnh break trong code để chạy full.")
        break

    # --- TỐI ƯU HÓA NGƯỠNG OOF ---
    valid_mask = oof_probs > 0 
    final_y_true = oof_targets[valid_mask]
    final_y_probs = oof_probs[valid_mask]
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1_overall = 0.0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (final_y_probs >= thresh).astype(int)
        score = f1_score(final_y_true, y_pred, average='macro')
        if score > best_f1_overall:
            best_f1_overall = score
            optimal_threshold = thresh

    print(f"\n[*] Tối ưu hóa Threshold OOF -> {optimal_threshold:.2f} (Macro F1 = {best_f1_overall:.4f})")
    
    y_pred_final = (final_y_probs >= optimal_threshold).astype(int)
    print("\n" + "="*40)
    print("BÁO CÁO PHÂN LOẠI (OOF CLASSIFICATION REPORT)")
    print("="*40)
    print(classification_report(final_y_true, y_pred_final))
    
    overall_auc = roc_auc_score(final_y_true, final_y_probs)
    print(f"ROC-AUC: {overall_auc:.4f}")
    
    cm = confusion_matrix(final_y_true, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unusable(0)', 'Usable(1)'], yticklabels=['Unusable(0)', 'Usable(1)'])
    plt.title(f'OOF Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.savefig(os.path.join(Config.MODELS_DIR, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    valid_df = prepare_data(Config.CSV_PATH, Config.AUDIO_ROOT)
    train_fusion_model(valid_df)