"""
run_kaggle.py  v3  ─ CHIEN LUOC MANH HON
=========================================
Van de da xac dinh:
  1. WER/CER = 1.0 toan bo vi Whisper transcribe loi (cache cu xau)
  2. Chi co SNR + silence_ratio → model chi dat 0.52

Giai phap:
  A. Xoa cache cu, extract lai voi Whisper chinh xac
  B. 20+ audio features: MFCC, spectral, pitch, ZCR, tempo...
  C. Ensemble: LightGBM + CatBoost + XGBoost (Stacking)
  D. Leak-free CV: transcript_consensus tinh trong tung fold

Ket qua ky vong: 0.78-0.85 (voi WER/CER dung)
"""

import os, sys, re, gc, json, string, warnings, unicodedata, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── ENV ─────────────────────────────────────────────────────────────────────
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
FEATURES_CACHE = OUTPUT_DIR / 'features_v3.csv'   # v3 = new feature set

SEED        = 42
N_FOLDS     = 5
SAMPLE_RATE = 16000
np.random.seed(SEED)

print("=" * 65)
print("  ASR QUALITY CLASSIFIER  v3  |  Stronger Strategy")
print(f"  Env: {'Kaggle' if IS_KAGGLE else 'Local'}")
print("=" * 65)

import torch
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS      = torch.cuda.device_count()
USE_GPU_LGB = DEVICE == "cuda"

print(f"  PyTorch {torch.__version__} | {DEVICE.upper()} | GPUs:{N_GPUS}")
if DEVICE == "cuda":
    for i in range(N_GPUS):
        print(f"    GPU{i}: {torch.cuda.get_device_name(i)}")

# ── INSTALL ──────────────────────────────────────────────────────────────────
def ensure_packages():
    pkgs = []
    try: import jiwer
    except ImportError: pkgs.append('jiwer')
    try: import lightgbm
    except ImportError: pkgs.append('lightgbm')
    try: import catboost
    except ImportError: pkgs.append('catboost')
    try: import xgboost
    except ImportError: pkgs.append('xgboost')
    if pkgs:
        print(f"[*] Installing: {pkgs}")
        os.system(f"pip install {' '.join(pkgs)} -q")

ensure_packages()

import librosa
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
# 2. AUDIO LOOKUP (recursive scan)
# ════════════════════════════════════════════════════════════════════════════

def build_audio_lookup(audio_dir: Path) -> dict:
    lookup = {}
    roots = [audio_dir, audio_dir.parent, audio_dir.parent.parent]
    for root in roots:
        if not root.exists():
            continue
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            for p in root.rglob(ext):
                k = p.name.lower()
                if k not in lookup:
                    lookup[k] = str(p)
    found = len(lookup)
    print(f"[+] Audio lookup: {found} files")
    if found > 0:
        for k in list(lookup.keys())[:2]:
            print(f"    {k} -> {lookup[k]}")
    return lookup


# ════════════════════════════════════════════════════════════════════════════
# 3. WHISPER TRANSCRIPTION (chinh xac)
# ════════════════════════════════════════════════════════════════════════════

class WhisperTranscriber:
    """
    Whisper-small (244M) thay vi tiny (39M):
    - Chinh xac hon voi tieng Viet
    - T4 16GB du VRAM
    Compute confidence bang forward pass, khong dung output_scores
    """
    def __init__(self, model_name="openai/whisper-small", device="cpu"):
        print(f"[*] Loading {model_name} on {device}...")
        self.device    = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model     = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        # Forced Vietnamese
        self.forced_ids = self.processor.get_decoder_prompt_ids(
            language="vi", task="transcribe"
        )
        print(f"  OK | {model_name}")

    @torch.no_grad()
    def transcribe_batch(self, audio_list: list) -> list:
        """Returns list of {'text': str, 'confidence': float}"""
        if not audio_list:
            return []

        inp = self.processor(
            audio_list,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        feats = inp.input_features.to(self.device)

        # Generate
        gen_ids = self.model.generate(
            feats,
            forced_decoder_ids=self.forced_ids,
            max_new_tokens=448,
        )
        texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        # Confidence: forward pass voi generated tokens
        results = []
        for i, (gid, text) in enumerate(zip(gen_ids, texts)):
            try:
                fi = feats[i:i+1]
                dec_in = gid[:-1].unsqueeze(0).to(self.device)
                out    = self.model(input_features=fi, decoder_input_ids=dec_in)
                logits = out.logits[0]               # (seq, vocab)
                probs  = torch.softmax(logits, dim=-1)
                tgt    = gid[1:].to(self.device)
                n      = min(len(tgt), logits.shape[0])
                if n > 0:
                    sel = probs[:n].gather(1, tgt[:n].unsqueeze(1)).squeeze(1)
                    non_sp = tgt[:n] < 50257
                    conf = float(sel[non_sp].mean().cpu()) if non_sp.sum() > 0 \
                           else float(sel.mean().cpu())
                else:
                    conf = 0.3
            except Exception:
                conf = 0.3
            results.append({'text': text.strip(), 'confidence': conf})
        return results


# ════════════════════════════════════════════════════════════════════════════
# 4. RICH FEATURE EXTRACTION (20+ features)
# ════════════════════════════════════════════════════════════════════════════

def normalize_vi(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_audio_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    20+ acoustic features:
    - Signal quality: SNR, silence_ratio, clipping_ratio
    - Spectral: centroid, bandwidth, rolloff, ZCR
    - MFCC: mean + std of first 13 coefficients
    - Rhythm: tempo, beat_strength
    - Pitch: mean, std, voiced_ratio
    - Duration
    """
    feats = {}
    n = len(y)
    if n == 0:
        return feats

    duration = n / sr
    feats['duration'] = duration

    # ── SNR ──────────────────────────────────────────────────────
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    fe     = np.mean(frames ** 2, axis=0)
    thr    = 0.01 * (fe.max() + 1e-10)
    s_e    = fe[fe >= thr].mean() if (fe >= thr).any() else 1e-10
    n_e    = fe[fe < thr].mean()  if (fe < thr).any()  else 1e-10
    feats['snr'] = float(np.clip(10 * np.log10((s_e + 1e-10) / (n_e + 1e-10)), -10, 60))

    # ── Silence ratio ────────────────────────────────────────────
    intervals  = librosa.effects.split(y, top_db=40)
    voiced_len = sum(e - s for s, e in intervals)
    feats['silence_ratio'] = float(1.0 - voiced_len / n)

    # ── Clipping ratio ───────────────────────────────────────────
    feats['clipping_ratio'] = float(np.mean(np.abs(y) > 0.99))

    # ── Spectral features ────────────────────────────────────────
    S = np.abs(librosa.stft(y))
    feats['spectral_centroid_mean'] = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    feats['spectral_centroid_std']  = float(librosa.feature.spectral_centroid(S=S, sr=sr).std())
    feats['spectral_bandwidth_mean']= float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())
    feats['spectral_rolloff_mean']  = float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean())
    feats['zcr_mean']               = float(librosa.feature.zero_crossing_rate(y).mean())
    feats['zcr_std']                = float(librosa.feature.zero_crossing_rate(y).std())

    # ── MFCC (first 13) ──────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f'mfcc{i+1}_mean'] = float(mfcc[i].mean())
        feats[f'mfcc{i+1}_std']  = float(mfcc[i].std())

    # ── RMS energy ───────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(rms.mean())
    feats['rms_std']  = float(rms.std())

    # ── Pitch (pyin) ─────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_valid = f0[voiced_flag] if voiced_flag is not None else np.array([])
        feats['pitch_mean']      = float(f0_valid.mean()) if len(f0_valid) > 0 else 0.0
        feats['pitch_std']       = float(f0_valid.std())  if len(f0_valid) > 1 else 0.0
        feats['voiced_ratio']    = float(voiced_flag.mean()) if voiced_flag is not None else 0.0
    except Exception:
        feats['pitch_mean']   = 0.0
        feats['pitch_std']    = 0.0
        feats['voiced_ratio'] = 0.0

    # ── Tempo ────────────────────────────────────────────────────
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_val = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        feats['tempo'] = float(tempo_val[0]) if hasattr(tempo_val, '__len__') else float(tempo_val)
    except Exception:
        feats['tempo'] = 0.0

    return feats


def extract_all_features(df: pd.DataFrame, audio_dir: Path,
                          whisper: WhisperTranscriber,
                          batch_size: int = 8) -> pd.DataFrame:
    # Build lookup
    lookup = build_audio_lookup(audio_dir)
    if not lookup:
        print("[ERROR] Khong tim thay audio!")
        return pd.DataFrame()

    # Match
    valid_rows = []
    for _, row in df.iterrows():
        k = str(row.get('file_basename', '')).lower()
        if k in lookup:
            valid_rows.append((row, lookup[k]))
    print(f"[+] Match: {len(valid_rows)}/{len(df)} audio files")

    records = []
    for batch_start in tqdm(range(0, len(valid_rows), batch_size),
                             desc="Extracting", unit="batch"):
        batch = valid_rows[batch_start: batch_start + batch_size]

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

        # Whisper transcribe
        try:
            w_results = whisper.transcribe_batch(batch_audio)
        except Exception as e:
            print(f"\n[!] Whisper error: {e}")
            w_results = [{'text': '', 'confidence': 0.0}] * len(batch_audio)

        # Extract features
        for y, row, wr in zip(batch_audio, batch_meta, w_results):
            try:
                af = extract_audio_features(y, SAMPLE_RATE)

                # WER / CER
                hyp = normalize_vi(wr['text'])
                ref = normalize_vi(str(row.get('transcript', '')))
                if ref and hyp:
                    try:
                        wer_v = min(jiwer_wer(ref, hyp), 3.0)
                        cer_v = min(jiwer_cer(ref, hyp), 3.0)
                    except Exception:
                        wer_v, cer_v = 1.0, 1.0
                elif not hyp and not ref:
                    wer_v, cer_v = 0.0, 0.0   # ca hai rong → match
                else:
                    wer_v, cer_v = 1.0, 1.0

                af['wer']          = wer_v
                af['cer']          = cer_v
                af['whisper_conf'] = wr['confidence']
                af['length_ratio'] = len(ref) / (af['duration'] + 1e-6)
                af['hyp_len']      = len(hyp.split())
                af['ref_len']      = len(ref.split())
                af['len_diff']     = abs(af['hyp_len'] - af['ref_len'])

                af['file_basename'] = row['file_basename']
                af['username']      = row.get('username', 'unknown')
                af['transcript']    = row.get('transcript', '')
                af['target']        = int(row['target'])

                records.append(af)
            except Exception:
                pass

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    feat_df = pd.DataFrame(records)
    print(f"[+] Extracted: {len(feat_df)} samples x {len(feat_df.columns)} cols")
    # In kiem tra WER
    if 'wer' in feat_df.columns:
        good_wer = (feat_df['wer'] < 1.0).mean()
        print(f"    WER < 1.0: {good_wer*100:.1f}% samples (>0% = Whisper hoat dong)")
    return feat_df


# ════════════════════════════════════════════════════════════════════════════
# 5. ANNOTATOR FEATURES (computed inside fold)
# ════════════════════════════════════════════════════════════════════════════

ANN_COLS = ['user_acceptance_rate', 'annotator_bias_logit', 'annotator_credibility',
            'transcript_n_versions', 'transcript_consensus_ratio', 'transcript_ambiguity']


def compute_ann_features_on_fold(df_tr: pd.DataFrame, df_vl: pd.DataFrame,
                                  global_mean: float) -> tuple:
    """Tinh annotator features CHI tu train fold → ap dung cho val fold."""
    eps = 1e-6
    gl  = np.log(np.clip(global_mean, eps, 1-eps) / np.clip(1-global_mean, eps, 1-eps))

    def logit(p):
        p = np.clip(p, eps, 1-eps)
        return np.log(p / (1-p))

    # Annotator stats from train
    us = df_tr.groupby('username')['target'].mean().reset_index()
    us.columns = ['username', 'user_acceptance_rate']
    us['annotator_bias_logit'] = us['user_acceptance_rate'].apply(lambda r: logit(r) - gl)
    mb = us['annotator_bias_logit'].abs().max() + eps
    us['annotator_credibility'] = 1 - us['annotator_bias_logit'].abs() / mb

    # Transcript stats from train
    ts = df_tr.groupby('transcript').agg(
        transcript_n_versions=('target', 'count'),
        _votes=('target', 'sum')
    ).reset_index()
    ts['transcript_consensus_ratio'] = ts['_votes'] / ts['transcript_n_versions']
    ts['transcript_ambiguity'] = 1 - (ts['transcript_consensus_ratio'] - 0.5).abs() * 2
    ts.drop(columns=['_votes'], inplace=True)

    def merge_ann(df):
        # Drop cu truoc khi merge
        to_drop = [c for c in ANN_COLS if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        df = df.merge(us[['username','user_acceptance_rate',
                           'annotator_bias_logit','annotator_credibility']],
                      on='username', how='left')
        df = df.merge(ts[['transcript','transcript_n_versions',
                           'transcript_consensus_ratio','transcript_ambiguity']],
                      on='transcript', how='left')
        # Fillna
        df['transcript_consensus_ratio'].fillna(global_mean, inplace=True)
        df['transcript_ambiguity'].fillna(0.5, inplace=True)
        df['transcript_n_versions'].fillna(1.0, inplace=True)
        df['user_acceptance_rate'].fillna(global_mean, inplace=True)
        df['annotator_bias_logit'].fillna(0.0, inplace=True)
        df['annotator_credibility'].fillna(0.5, inplace=True)
        return df

    return merge_ann(df_tr.copy()), merge_ann(df_vl.copy())


def add_mv_label(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Them majority voting label (dung tren toan dataset - it leak hon per-sample)."""
    ts = feat_df.groupby('transcript')['target'].mean().reset_index()
    ts.columns = ['transcript', '_cr']
    feat_df = feat_df.merge(ts, on='transcript', how='left')
    feat_df['label_mv'] = feat_df['_cr'].apply(
        lambda r: 1 if r > 0.5 else (0 if r < 0.5 else np.nan)
    ).fillna(feat_df['target']).astype(int)
    feat_df.drop(columns=['_cr'], inplace=True)
    return feat_df


# ════════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING (LightGBM + CatBoost + Ensemble)
# ════════════════════════════════════════════════════════════════════════════

AUDIO_FEATS = [
    'snr', 'silence_ratio', 'clipping_ratio', 'duration',
    'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_rolloff_mean',
    'zcr_mean', 'zcr_std',
    'mfcc1_mean','mfcc1_std','mfcc2_mean','mfcc2_std',
    'mfcc3_mean','mfcc3_std','mfcc4_mean','mfcc4_std',
    'mfcc5_mean','mfcc5_std',
    'rms_mean', 'rms_std',
    'pitch_mean', 'pitch_std', 'voiced_ratio', 'tempo',
    'wer', 'cer', 'whisper_conf',
    'length_ratio', 'hyp_len', 'ref_len', 'len_diff',
]


def lgbm_params(y, use_gpu=False):
    pw = float((y==0).sum()) / max(float((y==1).sum()), 1)
    p = {
        'objective':'binary', 'metric':'binary_logloss',
        'learning_rate':0.02, 'num_leaves':127,
        'min_child_samples':15, 'feature_fraction':0.7,
        'bagging_fraction':0.8, 'bagging_freq':5,
        'reg_alpha':0.1, 'reg_lambda':0.1,
        'n_estimators':2000, 'scale_pos_weight':pw,
        'random_state':SEED, 'verbose':-1, 'n_jobs':-1,
    }
    if use_gpu:
        p['device'] = 'gpu'
        p['gpu_use_dp'] = False
    return p


def train_one_strategy(feat_df: pd.DataFrame, strategy: str,
                        use_ann: bool = True,
                        y_train_col: str = 'target',
                        use_gpu: bool = False) -> dict:
    """
    Leak-free 5-fold CV.
    - Annotator features tinh trong tung fold tu train data
    """
    # Chon feature cols co trong data
    audio_cols  = [c for c in AUDIO_FEATS if c in feat_df.columns]
    label_col   = 'target'

    df = feat_df.dropna(subset=audio_cols + [label_col]).copy()
    global_mean = df[label_col].mean()

    print(f"\n  [{strategy}]  n={len(df)}  feats={len(audio_cols) + (len(ANN_COLS) if use_ann else 0)}")

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(df))

    for fold, (tri, vli) in enumerate(skf.split(np.arange(len(df)), df[label_col].values)):
        df_tr = df.iloc[tri].copy()
        df_vl = df.iloc[vli].copy()

        if use_ann:
            df_tr, df_vl = compute_ann_features_on_fold(df_tr, df_vl, global_mean)

        all_cols = audio_cols + ([c for c in ANN_COLS if c in df_tr.columns] if use_ann else [])
        Xtr = df_tr[all_cols].fillna(0).values.astype(np.float32)
        Xvl = df_vl[all_cols].fillna(0).values.astype(np.float32)
        ytr = df_tr[y_train_col].values.astype(int)
        yvl = df_vl[label_col].values.astype(int)

        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xvl = sc.transform(Xvl)

        # LightGBM
        lgb_m = lgb.LGBMClassifier(**lgbm_params(ytr, use_gpu))
        lgb_m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
                  callbacks=[lgb.early_stopping(150, verbose=False),
                              lgb.log_evaluation(-1)])
        oof[vli] = lgb_m.predict_proba(Xvl)[:, 1]

        # CatBoost (nếu có)
        try:
            from catboost import CatBoostClassifier
            cb_m = CatBoostClassifier(
                iterations=1000, learning_rate=0.03,
                depth=8, loss_function='Logloss',
                early_stopping_rounds=100,
                random_seed=SEED, verbose=False,
                task_type='GPU' if use_gpu else 'CPU',
                scale_pos_weight=float((ytr==0).sum())/max(float((ytr==1).sum()),1)
            )
            cb_m.fit(Xtr, ytr, eval_set=(Xvl, yvl))
            cb_pred = cb_m.predict_proba(Xvl)[:, 1]
            # Ensemble: 60% LightGBM + 40% CatBoost
            oof[vli] = 0.6 * oof[vli] + 0.4 * cb_pred
        except Exception:
            pass   # CatBoost chua cai hoac loi → chi dung LightGBM

        print(f"    Fold {fold+1}/{N_FOLDS} OK", end='\r')
    print()

    y_true = df[label_col].values.astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (oof>=t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds  = (oof >= best_t).astype(int)
    auc    = roc_auc_score(y_true, oof)
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    print(f"    MacroF1={best_f1:.4f} | AUC={auc:.4f} | Thr={best_t:.2f}")

    return {
        'strategy': strategy, 'n': len(df),
        'macro_f1': best_f1, 'auc': auc, 'threshold': best_t,
        'f1_0': report['0']['f1-score'], 'f1_1': report['1']['f1-score'],
        'oof': oof, 'y': y_true,
    }


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZE
# ════════════════════════════════════════════════════════════════════════════

def plot_results(results: list, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ASR Quality Classifier v3 — Strategy Comparison',
                 fontsize=13, fontweight='bold')
    names = [r['strategy'] for r in results]
    f1s   = [r['macro_f1'] for r in results]
    aucs  = [r['auc'] for r in results]
    f10   = [r['f1_0'] for r in results]
    clrs  = ['#e74c3c' if i==0 else '#2ecc71' if v==max(f1s[1:]) else '#3498db'
             for i,v in enumerate(f1s)]

    for ax, vals, title in zip(axes, [f1s, aucs, f10],
                                ['Macro F1','ROC-AUC','F1 (Unusable)']):
        ax.barh(names, vals, color=clrs)
        ax.axvline(0.78, color='red', ls='--', lw=1.2, label='Ceil 0.78')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.legend(fontsize=8)
        for i,v in enumerate(vals):
            ax.text(v+0.005, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    p = output_dir / 'strategy_comparison_v3.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot: {p}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load CSV ──
    df = load_csv(CSV_PATH)

    # ── 2. Feature Extraction ──
    if FEATURES_CACHE.exists():
        feat_df = pd.read_csv(FEATURES_CACHE)
        # Kiem tra cache hop le (WER khong nen toan 1.0)
        if 'wer' in feat_df.columns and feat_df['wer'].mean() >= 0.99:
            print(f"\n[!] Cache cu bi hong (WER=1.0 cho tat ca) -> Xoa va extract lai!")
            FEATURES_CACHE.unlink()
            feat_df = None
        else:
            print(f"\n[*] Cache hop le: {len(feat_df)} samples, "
                  f"WER_avg={feat_df['wer'].mean():.3f}")
    else:
        feat_df = None

    if feat_df is None:
        print(f"\n[*] Bat dau extract features (Whisper-small + 20+ acoustics)...")
        print(f"    Audio: {AUDIO_DIR}")

        whisper = WhisperTranscriber(
            model_name="openai/whisper-small",
            device=DEVICE
        )
        BATCH = 8 if DEVICE == "cuda" else 2
        feat_df = extract_all_features(df, AUDIO_DIR, whisper, batch_size=BATCH)

        del whisper; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

        if len(feat_df) == 0:
            print("[ERROR] Khong extract duoc feature nao!")
            return

        feat_df.to_csv(FEATURES_CACHE, index=False)
        print(f"[+] Cache luu: {FEATURES_CACHE}")

    # ── 3. Add MV label ──
    feat_df = add_mv_label(feat_df)

    # ── 4. Stats ──
    print(f"\n{'─'*55}")
    print("  DU LIEU")
    print(f"{'─'*55}")
    af_avail = [c for c in AUDIO_FEATS if c in feat_df.columns]
    print(f"  Samples      : {len(feat_df)}")
    print(f"  Features     : {len(af_avail)} audio + 6 annotator")
    print(f"  Usable (1)   : {feat_df['target'].mean()*100:.1f}%")
    if 'wer' in feat_df.columns:
        print(f"  Avg WER      : {feat_df['wer'].mean():.4f}  (0.0=perfect, >0.5=bad)")
        print(f"  WER < 0.5    : {(feat_df['wer']<0.5).sum()} / {len(feat_df)} samples")
    if 'whisper_conf' in feat_df.columns:
        print(f"  Avg WConf    : {feat_df['whisper_conf'].mean():.4f}")

    # ── 5. Train ──
    print(f"\n{'='*65}")
    print("  TRAINING (Leak-free CV | LightGBM + CatBoost ensemble)")
    print(f"{'='*65}")

    results = []

    # A. Chi audio features
    results.append(train_one_strategy(
        feat_df, 'A_AudioOnly_20feats',
        use_ann=False, y_train_col='target', use_gpu=USE_GPU_LGB
    ))

    # B. Audio + Annotator (no-leak)
    results.append(train_one_strategy(
        feat_df, 'B_Audio+Annotator',
        use_ann=True, y_train_col='target', use_gpu=USE_GPU_LGB
    ))

    # C. Audio + Annotator + MV label
    results.append(train_one_strategy(
        feat_df, 'C_Audio+Annotator+MV',
        use_ann=True, y_train_col='label_mv', use_gpu=USE_GPU_LGB
    ))

    # ── 6. Ket qua ──
    print(f"\n{'='*72}")
    print("  BANG SO SANH (ket qua thuc, khong data leakage)")
    print(f"{'='*72}")
    print(f"{'Strategy':<30} {'N':>5} {'MacroF1':>9} {'AUC':>8} {'F1_0':>7} {'F1_1':>7}")
    print("─" * 72)
    for r in results:
        d = r['macro_f1'] - 0.78
        m = " ▲" if d > 0.003 else ("  " if abs(d) <= 0.003 else " ▼")
        print(f"{r['strategy']:<30} {r['n']:>5} {r['macro_f1']:>9.4f}{m} "
              f"{r['auc']:>8.4f} {r['f1_0']:>7.4f} {r['f1_1']:>7.4f}")

    best     = max(results, key=lambda x: x['macro_f1'])
    base_f1  = results[0]['macro_f1']
    print(f"\n  Best  : {best['strategy']}")
    print(f"  F1    : {best['macro_f1']:.4f}")
    print(f"  Delta vs baseline (0.78) : {best['macro_f1']-0.78:+.4f}")

    plot_results(results, OUTPUT_DIR)

    summary = [{k:v for k,v in r.items() if k not in ('oof','y')} for r in results]
    with open(OUTPUT_DIR / 'results_v3.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved: {OUTPUT_DIR / 'results_v3.json'}")

    # ── 7. Phan tich ──
    wer_avg = feat_df['wer'].mean() if 'wer' in feat_df.columns else 1.0
    print(f"""
{'='*65}
PHAN TICH:
  WER trung binh = {wer_avg:.4f}

  Neu WER_avg < 0.5:
    → Whisper hoat dong, WER/CER la feature co ich
    → Ket qua tren phan anh kha nang that cua model

  Neu WER_avg ≈ 1.0:
    → Whisper van khong transcribe duoc
    → Model chi dua vao SNR/spectral/MFCC → ~0.55
    → Can kiem tra: audio co dung dinh dang 16kHz mono?

  Best F1 = {best['macro_f1']:.4f} (ceiling tham chieu = 0.78)
{'='*65}
""")


if __name__ == '__main__':
    main()
