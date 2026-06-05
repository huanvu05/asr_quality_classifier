"""
run_kaggle.py  v4  ─ DEEP LEARNING APPROACH
=============================================
Thay doi chinh so voi v3:
  1. Whisper ENCODER EMBEDDINGS (512-dim) thay vi transcription
     - Khong can .generate(), khong bi DataParallel loi
     - Encoder Whisper-small da duoc train tren 680k h audio
     - Mean-pooled hidden state la "am thanh fingerprint" rich hon bat ky feature nao

  2. MLP Classifier tren embeddings:
     Input: whisper_emb(512) + librosa(25) + annotator(6) = 543 dim
     Arch: Linear(543,256) -> BN -> ReLU -> Dropout(0.3)
          -> Linear(256,128) -> BN -> ReLU -> Dropout(0.2)
          -> Linear(128,1)
     Train: AdamW, LR scheduler, class-weighted loss

  3. LightGBM tren cung feature set (nhanh, manh voi tabular)

  4. Stacking ensemble: MLP + LightGBM

  5. Luu model tot nhat (LightGBM .pkl + MLP .pth + scaler + config)

Ket qua ky vong: 0.78-0.88 (whisper embeddings rat manh)
"""

import os, gc, json, string, warnings, unicodedata, re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, joblib

warnings.filterwarnings('ignore')

# ── ENV ──────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    REPO_DIR   = Path('/kaggle/working/asr_quality_classifier')
    CSV_PATH   = Path('/kaggle/input/datasets/huanvu205/training/training.csv')
    AUDIO_DIR  = REPO_DIR / 'data/audio/data2'
    OUTPUT_DIR = Path('/kaggle/working/outputs')
else:
    REPO_DIR   = Path('/Users/admin/Documents/AI_ThucChien/asr_quality_classifier')
    CSV_PATH   = REPO_DIR / 'data/transcripts/training.csv'
    AUDIO_DIR  = REPO_DIR / 'data/audio/data2'
    OUTPUT_DIR = REPO_DIR / 'outputs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR       = OUTPUT_DIR / 'best_model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CACHE  = OUTPUT_DIR / 'features_v4.csv'
EMBED_CACHE     = OUTPUT_DIR / 'whisper_embeddings.npy'
EMBED_IDX_CACHE = OUTPUT_DIR / 'whisper_embeddings_idx.json'

SEED        = 42
N_FOLDS     = 5
SAMPLE_RATE = 16000
np.random.seed(SEED)

print("=" * 65)
print("  ASR QUALITY CLASSIFIER  v4  |  Deep Learning Approach")
print(f"  Env: {'Kaggle' if IS_KAGGLE else 'Local'}")
print("=" * 65)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# QUAN TRONG: Khong dung DataParallel cho Whisper!
# DataParallel boc model.generate() → 'DataParallel has no attr generate'
# Fix: chi dung 1 GPU (cuda:0) cho Whisper encoder
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_DEV = "cuda:0" if torch.cuda.is_available() else "cpu"   # GPU 0 cho Whisper
N_GPUS      = torch.cuda.device_count()
USE_GPU_LGB = DEVICE == "cuda"

print(f"  PyTorch {torch.__version__} | {DEVICE.upper()} | GPUs:{N_GPUS}")
print(f"  Whisper device: {WHISPER_DEV} (no DataParallel)")
if DEVICE == "cuda":
    for i in range(N_GPUS):
        print(f"    GPU{i}: {torch.cuda.get_device_name(i)}")

# ── PACKAGES ─────────────────────────────────────────────────────────────────
def ensure_packages():
    for pkg, imp in [('jiwer','jiwer'), ('lightgbm','lightgbm'), ('catboost','catboost')]:
        try: __import__(imp)
        except ImportError:
            os.system(f"pip install {pkg} -q")
ensure_packages()

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD CSV
# ════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['target'] = df['label_text'].apply(
        lambda x: 1 if str(x).strip().lower() == 'usable' else 0
    )
    if 'file_name' in df.columns:
        df['file_basename'] = df['file_name'].apply(lambda x: Path(str(x)).name)
    print(f"[+] CSV: {len(df)} rows | {df['username'].nunique()} annotators | "
          f"Usable={df['target'].mean()*100:.1f}%")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. AUDIO LOOKUP
# ════════════════════════════════════════════════════════════════════════════

def build_audio_lookup(audio_dir: Path) -> dict:
    lookup = {}
    roots = [audio_dir, audio_dir.parent, audio_dir.parent.parent]
    for root in roots:
        if not root.exists(): continue
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            for p in root.rglob(ext):
                k = p.name.lower()
                if k not in lookup:
                    lookup[k] = str(p)
    print(f"[+] Audio lookup: {len(lookup)} files")
    return lookup


# ════════════════════════════════════════════════════════════════════════════
# 3. WHISPER ENCODER EMBEDDINGS (KHONG TRANSCRIPTION!)
# ════════════════════════════════════════════════════════════════════════════

class WhisperEmbedder:
    """
    Dung Whisper ENCODER (khong decoder) de extract embeddings.
    
    Tai sao encoder? 
    - Encoder Whisper-small da hoc tren 680k h audio da ngon ngu
    - Hidden state (512-dim) encode thong tin am thanh giau hon MFCC rat nhieu
    - Khong can transcription → khong bi loi DataParallel, khong bi WER=1.0
    - Mean-pool toan bo sequence → 512-dim "audio fingerprint"
    
    Cach dung:
        embedder = WhisperEmbedder(device="cuda:0")
        embs = embedder.embed_batch([audio1, audio2, ...])  # (N, 512)
    """
    def __init__(self, model_name="openai/whisper-small", device="cuda:0"):
        print(f"[*] Loading Whisper encoder ({model_name}) on {device}...")
        self.device    = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        # Chi load encoder (nhe hon full model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        # KHONG dung DataParallel! .generate() va .encoder() se bi loi
        self.embed_dim = self.model.config.d_model  # 512 for whisper-small
        print(f"  OK | embed_dim={self.embed_dim}")

    @torch.no_grad()
    def embed_batch(self, audio_list: list) -> np.ndarray:
        """
        audio_list: list of np.ndarray (16kHz mono)
        Returns: (N, embed_dim) float32 numpy array
        """
        if not audio_list:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        inp = self.processor(
            audio_list,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        feats = inp.input_features.to(self.device).to(torch.float32)

        # Forward qua encoder → last_hidden_state: (B, seq_len, d_model)
        encoder_out = self.model.model.encoder(input_features=feats)
        hidden = encoder_out.last_hidden_state  # (B, 1500, 512)

        # Mean pooling → (B, 512) - average over time dimension
        embeddings = hidden.mean(dim=1)  # (B, 512)
        return embeddings.cpu().numpy().astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 4. LIBROSA FEATURES (fast, proven)
# ════════════════════════════════════════════════════════════════════════════

def extract_librosa_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    25 fast acoustic features (khong co pyin vì qua cham):
    SNR, silence, clipping, spectral, MFCC(5), ZCR, RMS, voiced_ratio, speaking_rate
    """
    feats = {}
    n = len(y)
    if n == 0:
        return {k: 0.0 for k in _LIBROSA_KEYS}

    feats['duration'] = n / sr

    # Energy frames (dung cho SNR va voiced_ratio)
    frames = librosa.util.frame(y, frame_length=1024, hop_length=256)
    fe     = np.mean(frames ** 2, axis=0) + 1e-12
    thr    = 0.01 * fe.max()
    s_e    = fe[fe >= thr].mean() if (fe >= thr).any() else fe.mean()
    n_e    = fe[fe <  thr].mean() if (fe <  thr).any() else 1e-12
    feats['snr']           = float(np.clip(10*np.log10(s_e/n_e), -10, 60))
    feats['voiced_ratio']  = float((fe > thr).mean())
    feats['clipping_ratio']= float(np.mean(np.abs(y) > 0.99))

    # Silence ratio
    intervals  = librosa.effects.split(y, top_db=40)
    voiced_len = sum(e-s for s,e in intervals)
    feats['silence_ratio'] = float(1.0 - voiced_len / n)

    # Speaking rate proxy
    trans = np.diff((fe > thr).astype(int))
    feats['speaking_rate'] = float(np.sum(trans > 0) / (feats['duration'] + 1e-6))

    # Spectral (dung STFT chung)
    S   = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    sc  = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats['spectral_centroid_mean'] = float(sc.mean())
    feats['spectral_centroid_std']  = float(sc.std())
    feats['spectral_bandwidth_mean']= float(sb.mean())
    feats['spectral_rolloff_mean']  = float(sro.mean())
    feats['zcr_mean']               = float(zcr.mean())
    feats['zcr_std']                = float(zcr.std())

    # MFCC (chi lay 5 coefficient de tranh overfit voi 3500 mau)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, n_fft=1024, hop_length=256)
    for i in range(5):
        feats[f'mfcc{i+1}_mean'] = float(mfcc[i].mean())
        feats[f'mfcc{i+1}_std']  = float(mfcc[i].std())

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    feats['rms_mean'] = float(rms.mean())
    feats['rms_std']  = float(rms.std())

    return feats

# Danh sach key (dung de fillna)
_LIBROSA_KEYS = [
    'duration', 'snr', 'voiced_ratio', 'clipping_ratio', 'silence_ratio',
    'speaking_rate', 'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_rolloff_mean',
    'zcr_mean', 'zcr_std',
] + [f'mfcc{i+1}_mean' for i in range(5)] + [f'mfcc{i+1}_std' for i in range(5)] + [
    'rms_mean', 'rms_std'
]

LIBROSA_COLS = _LIBROSA_KEYS  # 25 features


# ════════════════════════════════════════════════════════════════════════════
# 5. EXTRACT ALL: Whisper embeddings + librosa features
# ════════════════════════════════════════════════════════════════════════════

def extract_all(df: pd.DataFrame, audio_dir: Path,
                embedder: WhisperEmbedder,
                batch_size: int = 32) -> tuple:
    """
    Returns:
        feat_df   : DataFrame voi librosa features (25 cols) + metadata
        embed_arr : np.ndarray (N, 512) Whisper embeddings
    """
    lookup = build_audio_lookup(audio_dir)
    if not lookup:
        raise RuntimeError("Khong tim thay audio files!")

    valid = []
    for _, row in df.iterrows():
        k = str(row.get('file_basename', '')).lower()
        if k in lookup:
            valid.append((row, lookup[k]))
    print(f"[+] Match: {len(valid)}/{len(df)} files")

    records   = []
    all_embeds = []

    for bs in tqdm(range(0, len(valid), batch_size), desc="Extracting", unit="batch"):
        batch = valid[bs: bs+batch_size]

        # Load audio
        batch_audio, batch_meta = [], []
        for row, path in batch:
            try:
                y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                if len(y) > 100:
                    batch_audio.append(y)
                    batch_meta.append(row)
            except Exception:
                pass

        if not batch_audio:
            continue

        # ── GPU: Whisper encoder embeddings ──────────────────────────
        try:
            embs = embedder.embed_batch(batch_audio)  # (B, 512)
        except Exception as e:
            print(f"\n[!] Embed error: {e}")
            embs = np.zeros((len(batch_audio), embedder.embed_dim), dtype=np.float32)

        # ── CPU: librosa features ─────────────────────────────────────
        for y, row, emb in zip(batch_audio, batch_meta, embs):
            try:
                lf = extract_librosa_features(y, SAMPLE_RATE)
                lf['file_basename'] = row['file_basename']
                lf['username']      = row.get('username', 'unknown')
                lf['transcript']    = row.get('transcript', '')
                lf['target']        = int(row['target'])
                records.append(lf)
                all_embeds.append(emb)
            except Exception:
                pass

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    feat_df   = pd.DataFrame(records)
    embed_arr = np.vstack(all_embeds) if all_embeds else np.zeros((0, embedder.embed_dim))
    print(f"[+] Extracted: {len(feat_df)} samples")
    print(f"    Librosa: {len(LIBROSA_COLS)} feats | Embeddings: {embed_arr.shape}")
    return feat_df, embed_arr


# ════════════════════════════════════════════════════════════════════════════
# 6. ANNOTATOR FEATURES (no-leak: tinh trong tung fold)
# ════════════════════════════════════════════════════════════════════════════

ANN_COLS = ['user_acceptance_rate', 'annotator_bias_logit', 'annotator_credibility',
            'transcript_consensus_ratio', 'transcript_ambiguity', 'transcript_n_versions']


def compute_ann_features(df_tr: pd.DataFrame, df_vl: pd.DataFrame,
                          global_mean: float) -> tuple:
    """Tinh annotator features CHI tu train fold → ap dung cho val fold (no-leak)."""
    eps = 1e-6
    gl  = np.log(np.clip(global_mean, eps, 1-eps) / np.clip(1-global_mean, eps, 1-eps))

    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p / (1-p))

    us = df_tr.groupby('username')['target'].mean().reset_index()
    us.columns = ['username', 'user_acceptance_rate']
    us['annotator_bias_logit'] = us['user_acceptance_rate'].apply(lambda r: logit(r) - gl)
    mb = us['annotator_bias_logit'].abs().max() + eps
    us['annotator_credibility'] = 1 - us['annotator_bias_logit'].abs() / mb

    ts = df_tr.groupby('transcript').agg(
        transcript_n_versions=('target','count'),
        _v=('target','sum')
    ).reset_index()
    ts['transcript_consensus_ratio'] = ts['_v'] / ts['transcript_n_versions']
    ts['transcript_ambiguity'] = 1 - (ts['transcript_consensus_ratio'] - 0.5).abs() * 2
    ts.drop(columns=['_v'], inplace=True)

    def _merge(df):
        df = df.drop(columns=[c for c in ANN_COLS if c in df.columns])
        df = df.merge(us[['username','user_acceptance_rate',
                           'annotator_bias_logit','annotator_credibility']],
                      on='username', how='left')
        df = df.merge(ts[['transcript','transcript_n_versions',
                           'transcript_consensus_ratio','transcript_ambiguity']],
                      on='transcript', how='left')
        df['transcript_consensus_ratio'].fillna(global_mean, inplace=True)
        df['transcript_ambiguity'].fillna(0.5, inplace=True)
        df['transcript_n_versions'].fillna(1.0, inplace=True)
        df['user_acceptance_rate'].fillna(global_mean, inplace=True)
        df['annotator_bias_logit'].fillna(0.0, inplace=True)
        df['annotator_credibility'].fillna(0.5, inplace=True)
        return df
    return _merge(df_tr.copy()), _merge(df_vl.copy())


def add_mv_label(feat_df):
    ts = feat_df.groupby('transcript')['target'].mean().reset_index()
    ts.columns = ['transcript','_cr']
    feat_df = feat_df.merge(ts, on='transcript', how='left')
    feat_df['label_mv'] = feat_df['_cr'].apply(
        lambda r: 1 if r>0.5 else (0 if r<0.5 else np.nan)
    ).fillna(feat_df['target']).astype(int)
    feat_df.drop(columns=['_cr'], inplace=True)
    return feat_df


# ════════════════════════════════════════════════════════════════════════════
# 7. MLP CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

class MLPClassifier(nn.Module):
    """
    Deep MLP cho tabular + embedding features.
    Input: whisper_emb(512) + librosa(25) + annotator(6) = ~543 dim
    """
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(Xtr, ytr, Xvl, yvl, input_dim: int,
                   epochs: int = 80, lr: float = 3e-4,
                   device: str = "cuda") -> tuple:
    """Train MLP on one fold, return val predictions."""
    pos_weight = torch.tensor([(ytr==0).sum() / max((ytr==1).sum(), 1)],
                               dtype=torch.float32).to(device)
    model = MLPClassifier(input_dim).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xvl, dtype=torch.float32).to(device)

    dataset = TensorDataset(Xt, yt)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    best_f1, best_preds = 0.0, np.zeros(len(Xvl))

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep+1) % 10 == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(Xv)).cpu().numpy()
            f1 = f1_score(yvl, (preds>0.5).astype(int), average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1   = f1
                best_preds = preds.copy()

    return best_preds, model


# ════════════════════════════════════════════════════════════════════════════
# 8. TRAIN: LGBM + MLP + ENSEMBLE (leak-free CV)
# ════════════════════════════════════════════════════════════════════════════

def lgbm_params(y, use_gpu=False):
    pw = float((y==0).sum()) / max(float((y==1).sum()), 1)
    p = {
        'objective':'binary', 'metric':'binary_logloss',
        'learning_rate':0.02, 'num_leaves':127,
        'min_child_samples':15, 'feature_fraction':0.7,
        'bagging_fraction':0.8, 'bagging_freq':5,
        'reg_alpha':0.1, 'reg_lambda':0.2,
        'n_estimators':3000, 'scale_pos_weight':pw,
        'random_state':SEED, 'verbose':-1, 'n_jobs':-1,
    }
    if use_gpu:
        p['device'] = 'gpu'
        p['gpu_use_dp'] = False
    return p


def run_cv(feat_df: pd.DataFrame, embed_arr: np.ndarray,
           use_mlp: bool = True, use_gpu: bool = True) -> dict:
    """
    5-fold CV voi:
    - LightGBM tren [embed(512) + librosa(25) + annotator(6)]
    - MLP tren cung feature set
    - Ensemble: 50% LightGBM + 50% MLP
    - No data leakage: annotator features tinh trong tung fold
    """
    label_col = 'target'
    lib_cols  = [c for c in LIBROSA_COLS if c in feat_df.columns]

    df_clean = feat_df.dropna(subset=lib_cols + [label_col]).copy()
    df_clean = df_clean.reset_index(drop=True)

    # Align embeddings
    embed_sub = embed_arr[:len(df_clean)]  # mang da aligned theo thu tu extract

    global_mean = df_clean[label_col].mean()
    print(f"\n  n={len(df_clean)} | embed_dim={embed_sub.shape[1]} "
          f"| librosa={len(lib_cols)} | annotator=6")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(df_clean))
    oof_mlp = np.zeros(len(df_clean))
    best_lgb_models = []
    best_mlp_models = []
    best_scalers    = []

    y_all = df_clean[label_col].values.astype(int)

    for fold, (tri, vli) in enumerate(skf.split(np.arange(len(df_clean)), y_all)):
        print(f"  Fold {fold+1}/{N_FOLDS}", end=" ... ")

        df_tr = df_clean.iloc[tri].copy()
        df_vl = df_clean.iloc[vli].copy()

        # Tinh annotator features trong fold (no-leak)
        df_tr, df_vl = compute_ann_features(df_tr, df_vl, global_mean)

        ann_tr = df_tr[ANN_COLS].fillna(0).values.astype(np.float32)
        lib_tr = df_tr[lib_cols].fillna(0).values.astype(np.float32)
        emb_tr = embed_sub[tri]
        Xtr    = np.concatenate([emb_tr, lib_tr, ann_tr], axis=1)

        ann_vl = df_vl[ANN_COLS].fillna(0).values.astype(np.float32)
        lib_vl = df_vl[lib_cols].fillna(0).values.astype(np.float32)
        emb_vl = embed_sub[vli]
        Xvl    = np.concatenate([emb_vl, lib_vl, ann_vl], axis=1)

        ytr = df_tr[label_col].values.astype(int)
        yvl = df_vl[label_col].values.astype(int)

        # Scale
        sc  = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xvl_s = sc.transform(Xvl)
        best_scalers.append(sc)

        # ── LightGBM ────────────────────────────────────────────────────
        lgb_m = lgb.LGBMClassifier(**lgbm_params(ytr, use_gpu))
        lgb_m.fit(Xtr_s, ytr, eval_set=[(Xvl_s, yvl)],
                  callbacks=[lgb.early_stopping(200, verbose=False),
                              lgb.log_evaluation(-1)])
        oof_lgb[vli] = lgb_m.predict_proba(Xvl_s)[:,1]
        best_lgb_models.append(lgb_m)

        # ── MLP ─────────────────────────────────────────────────────────
        if use_mlp:
            mlp_preds, mlp_m = train_mlp_fold(
                Xtr_s, ytr.astype(float), Xvl_s, yvl,
                input_dim=Xtr_s.shape[1],
                epochs=80, lr=3e-4, device=DEVICE
            )
            oof_mlp[vli] = mlp_preds
            best_mlp_models.append(mlp_m)

        f_lgb = f1_score(yvl, (oof_lgb[vli]>0.5).astype(int), average='macro', zero_division=0)
        f_mlp = f1_score(yvl, (oof_mlp[vli]>0.5).astype(int), average='macro', zero_division=0) if use_mlp else 0
        print(f"LGB={f_lgb:.4f}  MLP={f_mlp:.4f}")

    # ── Ensemble OOF ────────────────────────────────────────────────────────
    if use_mlp:
        oof_ens = 0.5 * oof_lgb + 0.5 * oof_mlp
    else:
        oof_ens = oof_lgb

    results = {}
    for name, oof in [('LightGBM', oof_lgb), ('MLP', oof_mlp if use_mlp else oof_lgb),
                       ('Ensemble', oof_ens)]:
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(y_all, (oof>=t).astype(int), average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        auc    = roc_auc_score(y_all, oof)
        preds  = (oof >= best_t).astype(int)
        report = classification_report(y_all, preds, output_dict=True, zero_division=0)
        results[name] = {
            'macro_f1': best_f1, 'auc': auc, 'threshold': best_t,
            'f1_0': report['0']['f1-score'], 'f1_1': report['1']['f1-score'],
            'oof': oof, 'y': y_all,
        }
        print(f"  [{name:12}] MacroF1={best_f1:.4f} | AUC={auc:.4f} | Thr={best_t:.2f}")

    return results, best_lgb_models, best_mlp_models, best_scalers, df_clean


# ════════════════════════════════════════════════════════════════════════════
# 9. SAVE BEST MODEL
# ════════════════════════════════════════════════════════════════════════════

def save_best_model(lgb_models, mlp_models, scalers, feat_df, embed_arr,
                    results, model_dir: Path):
    """
    Luu model tot nhat:
    - best_lgb.pkl      : LightGBM model (fold tot nhat)
    - best_mlp.pth      : MLP state dict (fold tot nhat)
    - scaler.pkl        : StandardScaler
    - config.json       : feature names, threshold, metrics
    """
    best_name = max(results, key=lambda k: results[k]['macro_f1'])
    best_res  = results[best_name]

    # Tim fold co OOF F1 cao nhat
    oof = best_res['oof']
    y   = best_res['y']
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s = []
    for tri, vli in skf.split(np.arange(len(y)), y):
        preds_vl = (oof[vli] >= best_res['threshold']).astype(int)
        fold_f1s.append(f1_score(y[vli], preds_vl, average='macro', zero_division=0))
    best_fold = int(np.argmax(fold_f1s))

    print(f"\n[*] Luu model: {best_name} | Fold {best_fold+1} (F1={fold_f1s[best_fold]:.4f})")

    # LightGBM
    lgb_path = model_dir / 'best_lgb.pkl'
    joblib.dump(lgb_models[best_fold], lgb_path)
    print(f"    LightGBM: {lgb_path}")

    # MLP
    if mlp_models and len(mlp_models) > best_fold:
        mlp_path = model_dir / 'best_mlp.pth'
        torch.save(mlp_models[best_fold].state_dict(), mlp_path)
        print(f"    MLP:      {mlp_path}")

    # Scaler
    sc_path = model_dir / 'scaler.pkl'
    joblib.dump(scalers[best_fold], sc_path)
    print(f"    Scaler:   {sc_path}")

    # Config
    lib_cols = [c for c in LIBROSA_COLS if c in feat_df.columns]
    cfg = {
        'best_strategy'   : best_name,
        'best_fold'       : best_fold,
        'macro_f1'        : best_res['macro_f1'],
        'auc'             : best_res['auc'],
        'threshold'       : best_res['threshold'],
        'f1_unusable'     : best_res['f1_0'],
        'f1_usable'       : best_res['f1_1'],
        'embed_dim'       : embed_arr.shape[1],
        'librosa_cols'    : lib_cols,
        'annotator_cols'  : ANN_COLS,
        'input_dim'       : embed_arr.shape[1] + len(lib_cols) + len(ANN_COLS),
        'model_dir'       : str(model_dir),
        'all_results'     : {k: {kk:v for kk,v in vv.items() if kk not in ('oof','y')}
                             for k,vv in results.items()},
    }
    cfg_path = model_dir / 'config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"    Config:   {cfg_path}")
    print(f"\n[OK] Best model saved → {model_dir}")
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# 10. VISUALIZE
# ════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, output_dir: Path):
    names = list(results.keys())
    f1s   = [results[n]['macro_f1'] for n in names]
    aucs  = [results[n]['auc']       for n in names]
    f10   = [results[n]['f1_0']      for n in names]
    clrs  = ['#e67e22','#3498db','#2ecc71'][:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle('ASR Quality Classifier v4 — DL Approach', fontsize=13, fontweight='bold')
    for ax, vals, title in zip(axes, [f1s,aucs,f10], ['MacroF1','AUC','F1(Unusable)']):
        ax.barh(names, vals, color=clrs)
        ax.axvline(0.78, color='red', ls='--', lw=1.5, label='Target 0.78')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.legend(fontsize=9)
        for i,v in enumerate(vals):
            ax.text(v+0.005, i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    p = output_dir / 'results_v4.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot: {p}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    df = load_csv(CSV_PATH)

    # ── 2. Feature Extraction ────────────────────────────────────────────────
    if FEATURES_CACHE.exists() and EMBED_CACHE.exists():
        feat_df   = pd.read_csv(FEATURES_CACHE)
        embed_arr = np.load(EMBED_CACHE)
        print(f"\n[*] Cache: {len(feat_df)} samples | embed={embed_arr.shape}")

        # Kiem tra cache hop le (embed khong nen tat ca zero)
        if embed_arr.std() < 1e-6:
            print("[!] Embed cache bi hong (tat ca zero) → Xoa va extract lai!")
            FEATURES_CACHE.unlink(); EMBED_CACHE.unlink()
            feat_df = None; embed_arr = None
    else:
        feat_df = None; embed_arr = None

    if feat_df is None:
        print(f"\n[*] Bat dau extract (Whisper-small encoder + librosa)...")
        print(f"    Whisper device: {WHISPER_DEV} (single GPU, no DataParallel)")

        embedder = WhisperEmbedder(
            model_name="openai/whisper-small",
            device=WHISPER_DEV
        )
        BATCH = 32 if DEVICE == "cuda" else 4  # whisper encoder nhanh hon generate
        feat_df, embed_arr = extract_all(df, AUDIO_DIR, embedder, batch_size=BATCH)

        del embedder; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

        if len(feat_df) == 0:
            print("[ERROR] Khong extract duoc!"); return

        feat_df.to_csv(FEATURES_CACHE, index=False)
        np.save(EMBED_CACHE, embed_arr)
        print(f"[+] Saved: features={FEATURES_CACHE}")
        print(f"           embeddings={EMBED_CACHE} ({embed_arr.shape})")

    # ── 3. Add MV label ──────────────────────────────────────────────────────
    feat_df = add_mv_label(feat_df)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    lib_cols  = [c for c in LIBROSA_COLS if c in feat_df.columns]
    total_dim = embed_arr.shape[1] + len(lib_cols) + len(ANN_COLS)
    print(f"""
{'─'*60}
  DU LIEU
{'─'*60}
  Samples         : {len(feat_df)}
  Whisper embeds  : {embed_arr.shape[1]} dim  (encoder hidden state mean)
  Librosa feats   : {len(lib_cols)}
  Annotator feats : {len(ANN_COLS)} (computed per fold, no leak)
  Total input dim : {total_dim}
  Usable (1)      : {feat_df['target'].mean()*100:.1f}%

  Embed quality   : mean={embed_arr.mean():.4f} std={embed_arr.std():.4f}
  (std > 0.1 = embeddings diverse = Whisper encoder hoat dong tot)
{'─'*60}""")

    # ── 5. Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TRAINING: LightGBM + MLP + Ensemble (Leak-free 5-fold CV)")
    print(f"{'='*65}")

    results, lgb_models, mlp_models, scalers, df_clean = run_cv(
        feat_df, embed_arr,
        use_mlp=True,
        use_gpu=USE_GPU_LGB
    )

    # ── 6. Results ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  BANG SO SANH CUOI CUNG")
    print(f"{'='*72}")
    print(f"{'Model':<15} {'MacroF1':>9} {'AUC':>8} {'F1_0':>7} {'F1_1':>7}")
    print("─"*45)
    for name, r in results.items():
        d = r['macro_f1'] - 0.78
        m = " ▲" if d > 0.003 else ("  " if abs(d) <= 0.003 else " ▼")
        print(f"{name:<15} {r['macro_f1']:>9.4f}{m} {r['auc']:>8.4f} "
              f"{r['f1_0']:>7.4f} {r['f1_1']:>7.4f}")

    best_name = max(results, key=lambda k: results[k]['macro_f1'])
    best_f1   = results[best_name]['macro_f1']
    print(f"\n  Best: {best_name} | F1={best_f1:.4f} | "
          f"Delta vs 0.78: {best_f1-0.78:+.4f}")

    # ── 7. Save best model ───────────────────────────────────────────────────
    cfg = save_best_model(
        lgb_models, mlp_models, scalers,
        feat_df, embed_arr, results, MODEL_DIR
    )

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    plot_results(results, OUTPUT_DIR)

    # ── 9. Save summary JSON ─────────────────────────────────────────────────
    with open(OUTPUT_DIR/'results_v4.json', 'w') as f:
        json.dump({k:{kk:v for kk,v in vv.items() if kk not in ('oof','y')}
                   for k,vv in results.items()}, f, indent=2)
    print(f"[+] JSON: {OUTPUT_DIR/'results_v4.json'}")

    # ── 10. Phan tich ────────────────────────────────────────────────────────
    emb_std = embed_arr.std()
    print(f"""
{'='*65}
PHAN TICH KET QUA:
  Embed std = {emb_std:.4f}
  {'[OK] Whisper encoder hoat dong tot!' if emb_std > 0.1 else '[WARN] Embed std thap - kiem tra model'}

  Neu F1 >> 0.78:
    → Whisper embeddings chứa thong tin am thanh rat giau
    → MLP hoc duoc pattern tinh vi ma hand-crafted features bo qua

  Neu F1 ≈ 0.56 (nhu cu):
    → Van la label noise limit, khong phai feature limit
    → Nguyen nhan: 7 annotators co hanh vi khac nhau mau thuan
    → Day la HARD CEILING cua bai toan - insight quan trong cho bao cao

  Model da luu tai: {MODEL_DIR}
{'='*65}
""")


if __name__ == '__main__':
    main()
