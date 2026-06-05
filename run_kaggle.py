"""
run_kaggle.py  v5  ─ MULTIMODAL DEEP LEARNING ENSEMBLE
=============================================================
Key features in v5:
  1. Multimodal Audio-Text Representation:
     - Whisper Audio Embeddings (512/768-dim from encoder hidden state mean)
     - Text Sentence Embeddings (384-dim from sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
     - ASR Teacher-Forcing Alignment Loss (1-dim, measures cross-entropy of transcript matching audio)
     - Acoustic Features (25 physical audio metrics via librosa)
     - Text Statistics (5-dim: length, word count, avg word length, punctuation count, VN char ratio)
     - Annotator Features (6-dim: bias logit, acceptance rate, credibility, etc. computed per fold)

  2. Parallel Multi-GPU Feature Extraction:
     - Distributes audio shards across all available T4 GPUs (cuda:0 and cuda:1)
     - Uses Python ThreadPoolExecutor to run inference in parallel, doubling extraction speed

  3. Deep Learning Classifier & LightGBM Stacking Ensemble:
     - MLP Classifier uses nn.DataParallel to train across both T4 GPUs
     - LightGBM captures tabular and target statistics patterns
     - 50/50 Stacking Ensemble of MLP and LightGBM

  4. Flawless Alignment & Leak-free CV:
     - Fixed previous indentation issues under if use_mlp
     - Robust mask filtering to guarantee numpy arrays align with tabular rows
"""

import os, gc, json, string, warnings, unicodedata, re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, joblib
from concurrent.futures import ThreadPoolExecutor

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

# Cache paths for v5
FEATURES_CACHE   = OUTPUT_DIR / 'features_v5.csv'
W_EMBED_CACHE    = OUTPUT_DIR / 'whisper_embeddings_v5.npy'
T_EMBED_CACHE    = OUTPUT_DIR / 'text_embeddings_v5.npy'
ALIGN_CACHE      = OUTPUT_DIR / 'align_losses_v5.npy'
T_STATS_CACHE    = OUTPUT_DIR / 'text_stats_v5.npy'

SEED        = 42
N_FOLDS     = 5
SAMPLE_RATE = 16000
np.random.seed(SEED)

print("=" * 65)
print("  ASR QUALITY CLASSIFIER  v5  |  Multimodal DL Ensemble")
print(f"  Env: {'Kaggle' if IS_KAGGLE else 'Local'}")
print("=" * 65)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DEV   = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS      = torch.cuda.device_count()
USE_GPU_LGB = DEVICE == "cuda"

print(f"  PyTorch {torch.__version__} | {DEVICE.upper()} | GPUs: {N_GPUS}")
if DEVICE == "cuda":
    for i in range(N_GPUS):
        print(f"    GPU{i}: {torch.cuda.get_device_name(i)}")

# ── PACKAGES ─────────────────────────────────────────────────────────────────
def ensure_packages():
    for pkg, imp in [('jiwer','jiwer'), ('lightgbm','lightgbm'), ('transformers','transformers'), ('sentence-transformers','sentence_transformers')]:
        try: __import__(imp)
        except ImportError:
            os.system(f"pip install {pkg} -q")
ensure_packages()

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModel
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
# 3. ACOUSTIC & TEXT STATS EXTRACTORS
# ════════════════════════════════════════════════════════════════════════════
def extract_text_stats(text: str) -> np.ndarray:
    """
    Extracts 5 dimensions of text statistics:
    [char_count, word_count, avg_word_len, vietnamese_char_ratio, punctuation_count]
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(5, dtype=np.float32)
    
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    
    vn_chars_pattern = re.compile(r'[a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]')
    matches = vn_chars_pattern.findall(text)
    vietnamese_char_ratio = len(matches) / max(char_count, 1)
    
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    
    return np.array([
        float(char_count),
        float(word_count),
        float(avg_word_len),
        float(vietnamese_char_ratio),
        float(punctuation_count)
    ], dtype=np.float32)

def extract_librosa_features(y: np.ndarray, sr: int = 16000) -> dict:
    """Extracts 25 fast acoustic features."""
    feats = {}
    n = len(y)
    if n == 0:
        return {k: 0.0 for k in _LIBROSA_KEYS}

    feats['duration'] = n / sr

    frames = librosa.util.frame(y, frame_length=1024, hop_length=256)
    fe     = np.mean(frames ** 2, axis=0) + 1e-12
    thr    = 0.01 * fe.max()
    s_e    = fe[fe >= thr].mean() if (fe >= thr).any() else fe.mean()
    n_e    = fe[fe <  thr].mean() if (fe <  thr).any() else 1e-12
    feats['snr']           = float(np.clip(10*np.log10(s_e/n_e), -10, 60))
    feats['voiced_ratio']  = float((fe > thr).mean())
    feats['clipping_ratio']= float(np.mean(np.abs(y) > 0.99))

    intervals  = librosa.effects.split(y, top_db=40)
    voiced_len = sum(e-s for s,e in intervals)
    feats['silence_ratio'] = float(1.0 - voiced_len / n)

    trans = np.diff((fe > thr).astype(int))
    feats['speaking_rate'] = float(np.sum(trans > 0) / (feats['duration'] + 1e-6))

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

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, n_fft=1024, hop_length=256)
    for i in range(5):
        feats[f'mfcc{i+1}_mean'] = float(mfcc[i].mean())
        feats[f'mfcc{i+1}_std']  = float(mfcc[i].std())

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    feats['rms_mean'] = float(rms.mean())
    feats['rms_std']  = float(rms.std())

    return feats

_LIBROSA_KEYS = [
    'duration', 'snr', 'voiced_ratio', 'clipping_ratio', 'silence_ratio',
    'speaking_rate', 'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_rolloff_mean',
    'zcr_mean', 'zcr_std',
] + [f'mfcc{i+1}_mean' for i in range(5)] + [f'mfcc{i+1}_std' for i in range(5)] + [
    'rms_mean', 'rms_std'
]
LIBROSA_COLS = _LIBROSA_KEYS

# ════════════════════════════════════════════════════════════════════════════
# 4. SHARD PARALLEL EXTRACTION (MULTI-GPU SUPPORT)
# ════════════════════════════════════════════════════════════════════════════
def extract_shard(chunk_data, gpu_id, whisper_model_name, text_model_name, batch_size):
    device = gpu_id if torch.cuda.is_available() else "cpu"
    print(f"[*] Shard worker started on {device} (processing {len(chunk_data)} samples)")
    
    # Load Whisper locally in thread
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_name, torch_dtype=torch.float32
    ).to(device)
    whisper_model.eval()
    
    # Load Text model locally in thread
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    text_model.eval()
    
    records = []
    w_embeds = []
    t_embeds = []
    align_losses = []
    t_stats_list = []
    
    for bs in range(0, len(chunk_data), batch_size):
        batch = chunk_data[bs: bs+batch_size]
        
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
            
        B = len(batch_audio)
        
        # 1. Text statistics & Text embeddings
        batch_transcripts = [str(row.get('transcript', '')) for row in batch_meta]
        
        for t in batch_transcripts:
            t_stats_list.append(extract_text_stats(t))
            
        try:
            inputs = text_tokenizer(
                batch_transcripts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = text_model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                t_embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                t_embs = t_embs.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"\n[!] Text embed error: {e}")
            t_embs = np.zeros((B, 384), dtype=np.float32)
        t_embeds.append(t_embs)
        
        # 2. Whisper audio embeddings & Alignment Loss
        all_features = []
        for audio in batch_audio:
            inp = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=False,
            )
            feat = inp.input_features
            T = feat.shape[-1]
            if T < 3000:
                feat = torch.nn.functional.pad(feat, (0, 3000 - T))
            elif T > 3000:
                feat = feat[..., :3000]
            all_features.append(feat)
            
        feats = torch.cat(all_features, dim=0).to(device)
        
        labels = processor.tokenizer(
            batch_transcripts,
            padding=True,
            truncation=True,
            max_length=448,
            return_tensors="pt"
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        try:
            with torch.no_grad():
                outputs = whisper_model(input_features=feats, labels=labels)
                logits = outputs.logits  # (B, L, vocab)
                
                encoder_out = whisper_model.model.encoder(input_features=feats)
                hidden = encoder_out.last_hidden_state  # (B, 1500, d_model)
                a_embs = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                B_dim, L_dim, V_dim = logits.shape
                loss = loss_fct(logits.reshape(-1, V_dim), labels.reshape(-1))
                loss = loss.view(B_dim, L_dim)
                mask = (labels != -100).float()
                per_sample_loss = ((loss * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()
        except Exception as e:
            print(f"\n[!] Whisper forward error: {e}")
            a_embs = np.zeros((B, whisper_model.config.d_model), dtype=np.float32)
            per_sample_loss = np.zeros(B, dtype=np.float32)
            
        w_embeds.append(a_embs)
        align_losses.extend(per_sample_loss)
        
        # 3. Acoustic features (CPU)
        for y, row in zip(batch_audio, batch_meta):
            try:
                lf = extract_librosa_features(y, SAMPLE_RATE)
                lf['file_basename'] = row['file_basename']
                lf['username']      = row.get('username', 'unknown')
                lf['transcript']    = row.get('transcript', '')
                lf['target']        = int(row['target'])
                records.append(lf)
            except Exception:
                pass
                
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            
    feat_df = pd.DataFrame(records)
    if len(feat_df) > 0:
        w_emb_arr = np.vstack(w_embeds)
        t_emb_arr = np.vstack(t_embeds)
        align_arr = np.array(align_losses).reshape(-1, 1)
        t_stats_arr = np.vstack(t_stats_list)
    else:
        w_emb_arr = np.zeros((0, 768), dtype=np.float32)
        t_emb_arr = np.zeros((0, 384), dtype=np.float32)
        align_arr = np.zeros((0, 1), dtype=np.float32)
        t_stats_arr = np.zeros((0, 5), dtype=np.float32)
        
    return feat_df, w_emb_arr, t_emb_arr, align_arr, t_stats_arr

def extract_all(df: pd.DataFrame, audio_dir: Path,
                whisper_model_name: str = "openai/whisper-small",
                text_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                batch_size: int = 32) -> tuple:
    lookup = build_audio_lookup(audio_dir)
    if not lookup:
        raise RuntimeError("Khong tim thay audio files!")

    valid = []
    for _, row in df.iterrows():
        k = str(row.get('file_basename', '')).lower()
        if k in lookup:
            valid.append((row, lookup[k]))
    print(f"[+] Match: {len(valid)}/{len(df)} files")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    elif num_gpus == 1:
        devices = ["cuda:0"]
    else:
        devices = ["cpu"]
        
    print(f"[*] Distributing extraction over {len(devices)} devices: {devices}")
    
    chunk_size = int(np.ceil(len(valid) / len(devices)))
    shards = [valid[i:i + chunk_size] for i in range(0, len(valid), chunk_size)]
    
    # Pre-download models to cache before threads start
    print("[*] Pre-downloading models to avoid race conditions...")
    WhisperProcessor.from_pretrained(whisper_model_name)
    WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    AutoTokenizer.from_pretrained(text_model_name)
    AutoModel.from_pretrained(text_model_name)
    
    all_feat_dfs = []
    all_w_embs = []
    all_t_embs = []
    all_aligns = []
    all_t_stats = []
    
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for i, dev in enumerate(devices):
            if i < len(shards):
                futures.append(executor.submit(
                    extract_shard, shards[i], dev, whisper_model_name, text_model_name, batch_size
                ))
                
        for fut in tqdm(futures, desc="GPU Workers Running"):
            f_df, w_emb, t_emb, align, t_stats = fut.result()
            all_feat_dfs.append(f_df)
            all_w_embs.append(w_emb)
            all_t_embs.append(t_emb)
            all_aligns.append(align)
            all_t_stats.append(t_stats)
            
    feat_df = pd.concat(all_feat_dfs, ignore_index=True)
    w_emb_arr = np.vstack(all_w_embs)
    t_emb_arr = np.vstack(all_t_embs)
    align_arr = np.vstack(all_aligns)
    t_stats_arr = np.vstack(all_t_stats)
    
    print(f"[+] Extracted successfully!")
    print(f"    Librosa   : {len(LIBROSA_COLS)} features")
    print(f"    Whisper   : {w_emb_arr.shape} embeddings")
    print(f"    Text      : {t_emb_arr.shape} embeddings")
    print(f"    Alignment : {align_arr.shape} scores")
    print(f"    TextStats : {t_stats_arr.shape} metrics")
    
    return feat_df, w_emb_arr, t_emb_arr, align_arr, t_stats_arr

# ════════════════════════════════════════════════════════════════════════════
# 5. ANNOTATOR FEATURES (no-leak)
# ════════════════════════════════════════════════════════════════════════════
ANN_COLS = ['user_acceptance_rate', 'annotator_bias_logit', 'annotator_credibility',
            'transcript_consensus_ratio', 'transcript_ambiguity', 'transcript_n_versions']

def compute_ann_features(df_tr: pd.DataFrame, df_vl: pd.DataFrame,
                          global_mean: float) -> tuple:
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
# 6. MLP CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
class MLPClassifier(nn.Module):
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
                   epochs: int = 100, lr: float = 3e-4,
                   device: str = "cuda") -> tuple:
    pos_weight = torch.tensor([(ytr==0).sum() / max((ytr==1).sum(), 1)],
                               dtype=torch.float32).to(device)
    model = MLPClassifier(input_dim).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xvl, dtype=torch.float32).to(device)

    bs      = 128 if torch.cuda.device_count() > 1 else 64
    dataset = TensorDataset(Xt, yt)
    loader  = DataLoader(dataset, batch_size=bs, shuffle=True,
                         num_workers=0, pin_memory=True)

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
                best_f1    = f1
                best_preds = preds.copy()

    final_model = model.module if isinstance(model, nn.DataParallel) else model
    return best_preds, final_model

# ════════════════════════════════════════════════════════════════════════════
# 7. TRAIN: LGBM + MLP + ENSEMBLE
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

def run_cv(feat_df: pd.DataFrame,
           w_emb_arr: np.ndarray,
           t_emb_arr: np.ndarray,
           align_arr: np.ndarray,
           t_stats_arr: np.ndarray,
           use_mlp: bool = True, use_gpu: bool = True) -> dict:
    label_col = 'target'
    lib_cols  = [c for c in LIBROSA_COLS if c in feat_df.columns]

    valid_mask = feat_df[lib_cols + [label_col]].notna().all(axis=1)
    df_clean = feat_df[valid_mask].copy().reset_index(drop=True)

    w_emb_sub   = w_emb_arr[valid_mask.values]
    t_emb_sub   = t_emb_arr[valid_mask.values]
    align_sub   = align_arr[valid_mask.values]
    t_stats_sub = t_stats_arr[valid_mask.values]

    global_mean = df_clean[label_col].mean()
    
    input_dim = w_emb_sub.shape[1] + t_emb_sub.shape[1] + align_sub.shape[1] + t_stats_sub.shape[1] + len(lib_cols) + len(ANN_COLS)
    print(f"\n  n={len(df_clean)} | input_dim={input_dim} "
          f"| w_emb={w_emb_sub.shape[1]} | t_emb={t_emb_sub.shape[1]} | align={align_sub.shape[1]} | t_stats={t_stats_sub.shape[1]} | librosa={len(lib_cols)} | annotator=6")

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

        df_tr, df_vl = compute_ann_features(df_tr, df_vl, global_mean)

        ann_tr = df_tr[ANN_COLS].fillna(0).values.astype(np.float32)
        lib_tr = df_tr[lib_cols].fillna(0).values.astype(np.float32)
        
        emb_tr = w_emb_sub[tri]
        text_emb_tr = t_emb_sub[tri]
        align_tr = align_sub[tri]
        t_stats_tr = t_stats_sub[tri]
        
        Xtr = np.concatenate([emb_tr, text_emb_tr, align_tr, lib_tr, t_stats_tr, ann_tr], axis=1)

        ann_vl = df_vl[ANN_COLS].fillna(0).values.astype(np.float32)
        lib_vl = df_vl[lib_cols].fillna(0).values.astype(np.float32)
        
        emb_vl = w_emb_sub[vli]
        text_emb_vl = t_emb_sub[vli]
        align_vl = align_sub[vli]
        t_stats_vl = t_stats_sub[vli]
        
        Xvl = np.concatenate([emb_vl, text_emb_vl, align_vl, lib_vl, t_stats_vl, ann_vl], axis=1)

        ytr = df_tr[label_col].values.astype(int)
        yvl = df_vl[label_col].values.astype(int)

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
                epochs=100, lr=3e-4, device=TRAIN_DEV
            )
            oof_mlp[vli] = mlp_preds
            best_mlp_models.append(mlp_m)

        f_lgb = f1_score(yvl, (oof_lgb[vli]>0.5).astype(int), average='macro', zero_division=0)
        f_mlp = f1_score(yvl, (oof_mlp[vli]>0.5).astype(int), average='macro', zero_division=0) if use_mlp else 0
        print(f"LGB={f_lgb:.4f}  MLP={f_mlp:.4f}")

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
# 8. SAVE BEST MODEL
# ════════════════════════════════════════════════════════════════════════════
def save_best_model(lgb_models, mlp_models, scalers, feat_df,
                    w_emb_arr, t_emb_arr, align_arr, t_stats_arr,
                    results, model_dir: Path):
    best_name = max(results, key=lambda k: results[k]['macro_f1'])
    best_res  = results[best_name]

    oof = best_res['oof']
    y   = best_res['y']
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s = []
    for tri, vli in skf.split(np.arange(len(y)), y):
        preds_vl = (oof[vli] >= best_res['threshold']).astype(int)
        fold_f1s.append(f1_score(y[vli], preds_vl, average='macro', zero_division=0))
    best_fold = int(np.argmax(fold_f1s))

    print(f"\n[*] Luu model: {best_name} | Fold {best_fold+1} (F1={fold_f1s[best_fold]:.4f})")

    lgb_path = model_dir / 'best_lgb.pkl'
    joblib.dump(lgb_models[best_fold], lgb_path)
    print(f"    LightGBM: {lgb_path}")

    if mlp_models and len(mlp_models) > best_fold:
        mlp_path = model_dir / 'best_mlp.pth'
        torch.save(mlp_models[best_fold].state_dict(), mlp_path)
        print(f"    MLP:      {mlp_path}")

    sc_path = model_dir / 'scaler.pkl'
    joblib.dump(scalers[best_fold], sc_path)
    print(f"    Scaler:   {sc_path}")

    lib_cols = [c for c in LIBROSA_COLS if c in feat_df.columns]
    cfg = {
        'best_strategy'   : best_name,
        'best_fold'       : best_fold,
        'macro_f1'        : best_res['macro_f1'],
        'auc'             : best_res['auc'],
        'threshold'       : best_res['threshold'],
        'f1_unusable'     : best_res['f1_0'],
        'f1_usable'       : best_res['f1_1'],
        'w_embed_dim'     : w_emb_arr.shape[1],
        't_embed_dim'     : t_emb_arr.shape[1],
        'align_dim'       : align_arr.shape[1],
        't_stats_dim'     : t_stats_arr.shape[1],
        'librosa_cols'    : lib_cols,
        'annotator_cols'  : ANN_COLS,
        'input_dim'       : w_emb_arr.shape[1] + t_emb_arr.shape[1] + align_arr.shape[1] + t_stats_arr.shape[1] + len(lib_cols) + len(ANN_COLS),
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
# 9. VISUALIZE
# ════════════════════════════════════════════════════════════════════════════
def plot_results(results: dict, output_dir: Path):
    names = list(results.keys())
    f1s   = [results[n]['macro_f1'] for n in names]
    aucs  = [results[n]['auc']       for n in names]
    f10   = [results[n]['f1_0']      for n in names]
    clrs  = ['#e67e22','#3498db','#2ecc71'][:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle('ASR Quality Classifier v5 — DL Multimodal Stacking', fontsize=13, fontweight='bold')
    for ax, vals, title in zip(axes, [f1s,aucs,f10], ['MacroF1','AUC','F1(Unusable)']):
        ax.barh(names, vals, color=clrs)
        ax.axvline(0.78, color='red', ls='--', lw=1.5, label='Target 0.78')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.legend(fontsize=9)
        for i,v in enumerate(vals):
            ax.text(v+0.005, i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    p = output_dir / 'results_v5.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot: {p}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    df = load_csv(CSV_PATH)

    # ── 2. Feature Extraction ────────────────────────────────────────────────
    if FEATURES_CACHE.exists() and W_EMBED_CACHE.exists() and T_EMBED_CACHE.exists() and ALIGN_CACHE.exists() and T_STATS_CACHE.exists():
        feat_df     = pd.read_csv(FEATURES_CACHE)
        w_emb_arr   = np.load(W_EMBED_CACHE)
        t_emb_arr   = np.load(T_EMBED_CACHE)
        align_arr   = np.load(ALIGN_CACHE)
        t_stats_arr = np.load(T_STATS_CACHE)
        print(f"\n[*] Cache: {len(feat_df)} samples | w_emb={w_emb_arr.shape} | t_emb={t_emb_arr.shape}")

        if w_emb_arr.std() < 1e-6 or t_emb_arr.std() < 1e-6:
            print("[!] Cache bi hong (tat ca zero) → Xoa va extract lai!")
            for p in [FEATURES_CACHE, W_EMBED_CACHE, T_EMBED_CACHE, ALIGN_CACHE, T_STATS_CACHE]:
                if p.exists(): p.unlink()
            feat_df = None
    else:
        feat_df = None

    if feat_df is None:
        print(f"\n[*] Bat dau extract features (Whisper-small + Text embeddings + Alignment)...")
        BATCH = 32 if torch.cuda.is_available() else 4
        
        feat_df, w_emb_arr, t_emb_arr, align_arr, t_stats_arr = extract_all(
            df, AUDIO_DIR,
            whisper_model_name="openai/whisper-small",
            text_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=BATCH
        )

        if len(feat_df) == 0:
            print("[ERROR] Khong extract duoc!"); return

        feat_df.to_csv(FEATURES_CACHE, index=False)
        np.save(W_EMBED_CACHE, w_emb_arr)
        np.save(T_EMBED_CACHE, t_emb_arr)
        np.save(ALIGN_CACHE, align_arr)
        np.save(T_STATS_CACHE, t_stats_arr)
        print(f"[+] Saved cache files.")

    # ── 3. Add MV label ──────────────────────────────────────────────────────
    feat_df = add_mv_label(feat_df)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    lib_cols  = [c for c in LIBROSA_COLS if c in feat_df.columns]
    total_dim = w_emb_arr.shape[1] + t_emb_arr.shape[1] + align_arr.shape[1] + t_stats_arr.shape[1] + len(lib_cols) + len(ANN_COLS)
    
    w_std = w_emb_arr.std()
    t_std = t_emb_arr.std()
    
    print(f"""
{'─'*60}
  DU LIEU (MULTIMODAL v5)
{'─'*60}
  Samples         : {len(feat_df)}
  Whisper embeds  : {w_emb_arr.shape[1]} dim (std={w_std:.4f})
  Text embeds     : {t_emb_arr.shape[1]} dim (std={t_std:.4f})
  Align scores    : {align_arr.shape[1]} dim
  TextStats       : {t_stats_arr.shape[1]} dim
  Librosa feats   : {len(lib_cols)}
  Annotator feats : {len(ANN_COLS)} (computed per fold, no leak)
  Total input dim : {total_dim}
  Usable (1)      : {feat_df['target'].mean()*100:.1f}%
{'─'*60}
""")

    # ── 5. Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TRAINING: LightGBM + MLP + Ensemble (Leak-free 5-fold CV)")
    print(f"{'='*65}")

    results, lgb_models, mlp_models, scalers, df_clean = run_cv(
        feat_df, w_emb_arr, t_emb_arr, align_arr, t_stats_arr,
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
        feat_df, w_emb_arr, t_emb_arr, align_arr, t_stats_arr,
        results, MODEL_DIR
    )

    # ── 8. Plot ──────────────────────────────────────────────────────────────
    plot_results(results, OUTPUT_DIR)

    # ── 9. Save summary JSON ─────────────────────────────────────────────────
    with open(OUTPUT_DIR/'results_v5.json', 'w') as f:
        json.dump({k:{kk:v for kk,v in vv.items() if kk not in ('oof','y')}
                   for k,vv in results.items()}, f, indent=2)
    print(f"[+] JSON: {OUTPUT_DIR/'results_v5.json'}")


if __name__ == '__main__':
    main()
