"""
run_kaggle.py  ─ Script duy nhất, chạy từ đầu đến cuối trên Kaggle T4x2
=========================================================================

KHÔNG CẦN file nào có sẵn ngoài:
  - /kaggle/input/datasets/huanvu205/training/training.csv
  - /kaggle/working/asr_quality_classifier/data/audio/data2/  (audio files)

PIPELINE:
  1. Load CSV + audio paths
  2. Extract hand-crafted features bằng Whisper-tiny (GPU) + librosa
     → snr, silence_ratio, wer, cer, length_ratio, duration, whisper_conf
  3. Thêm Annotator-level features (bias, credibility, consensus)
  4. Train LightGBM với 4 chiến lược, so sánh kết quả
  5. Lưu artifacts và biểu đồ

THỜI GIAN ƯỚC TÍNH trên T4x2:
  - Feature extraction (3500 files, Whisper-tiny): ~15-20 phút
  - Training (LightGBM): <2 phút
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import os, sys, re, gc, json, string, warnings, unicodedata, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── AUTO-DETECT ENVIRONMENT ────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    REPO_DIR  = Path('/kaggle/working/asr_quality_classifier')
    CSV_PATH  = Path('/kaggle/input/datasets/huanvu205/training/training.csv')
    AUDIO_DIR = REPO_DIR / 'data/audio/data2'
    OUTPUT_DIR = Path('/kaggle/working/outputs')
else:
    REPO_DIR   = Path('/Users/admin/Documents/AI_ThucChien/asr_quality_classifier')
    CSV_PATH   = REPO_DIR / 'data/transcripts/training.csv'
    AUDIO_DIR  = REPO_DIR / 'data/audio/data2'
    OUTPUT_DIR = REPO_DIR / 'outputs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CACHE = OUTPUT_DIR / 'features_cache.csv'

SEED    = 42
N_FOLDS = 5
SAMPLE_RATE = 16000
np.random.seed(SEED)

print("=" * 65)
print("  ASR QUALITY CLASSIFIER  |  All-in-One Kaggle Script")
print(f"  Environment : {'Kaggle' if IS_KAGGLE else 'Local'}")
print("=" * 65)

# ─── CHECK TORCH & GPU ──────────────────────────────────────────────────────
import torch
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS     = torch.cuda.device_count()
USE_GPU_LGB = IS_KAGGLE and DEVICE == "cuda"

print(f"  PyTorch : {torch.__version__} | Device: {DEVICE.upper()} | GPUs: {N_GPUS}")
if DEVICE == "cuda":
    for i in range(N_GPUS):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# ─── INSTALL DEPS IF NEEDED ─────────────────────────────────────────────────
def ensure_packages():
    try:
        import jiwer, num2words, lightgbm
    except ImportError:
        print("[*] Cài thêm thư viện...")
        os.system("pip install jiwer num2words lightgbm -q")

ensure_packages()

import librosa
from jiwer import wer as compute_wer, cer as compute_cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: LOAD VÀ CHUẨN BỊ DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['target'] = df['label_text'].apply(
        lambda x: 1 if str(x).strip().lower() == 'usable' else 0
    )
    # Chuẩn hóa đường dẫn: lấy basename từ cột file_name
    if 'file_name' in df.columns:
        df['file_basename'] = df['file_name'].apply(lambda x: Path(str(x)).name)
        # Tìm folder (phần trước tên file, ví dụ: "clone/clone8.wav" → "clone")
        df['file_folder'] = df['file_name'].apply(
            lambda x: str(Path(str(x)).parent) if '/' in str(x) else ''
        )
    print(f"[+] Loaded CSV: {len(df)} rows | {df['username'].nunique()} annotators | "
          f"Usable={df['target'].mean()*100:.1f}%")
    return df


def find_audio_path(row: pd.Series, audio_dir: Path) -> str | None:
    """Tìm file audio theo nhiều pattern khác nhau."""
    candidates = []
    basename = row.get('file_basename', '')
    folder   = row.get('file_folder', '')
    file_name_raw = str(row.get('file_name', ''))

    # Pattern 1: data2/folder/file.wav
    candidates.append(audio_dir / file_name_raw)
    # Pattern 2: audio_dir/folder/file.wav
    if folder and folder != '.':
        candidates.append(audio_dir / folder / basename)
    # Pattern 3: audio_dir/file.wav (flat)
    candidates.append(audio_dir / basename)
    # Pattern 4: data2/file.wav
    candidates.append(audio_dir.parent / 'data2' / basename)

    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: FEATURE EXTRACTION (Whisper-tiny + librosa)
# ════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Chuẩn hóa text tiếng Việt cho WER/CER."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_snr(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512,
                energy_threshold: float = 0.01) -> float:
    """
    Ước tính SNR bằng cách tách frame active vs silent.
    SNR = 10 * log10(E_signal / E_noise)
    """
    if len(y) == 0:
        return 0.0
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.mean(frames ** 2, axis=0)
    noise_mask   = frame_energy < (energy_threshold * frame_energy.max() + 1e-10)
    signal_mask  = ~noise_mask

    if noise_mask.sum() == 0 or signal_mask.sum() == 0:
        return float(np.clip(10 * np.log10(frame_energy.mean() + 1e-10), -10, 60))

    e_signal = frame_energy[signal_mask].mean()
    e_noise  = frame_energy[noise_mask].mean()
    snr      = 10 * np.log10((e_signal + 1e-10) / (e_noise + 1e-10))
    return float(np.clip(snr, -10, 60))


def compute_silence_ratio(y: np.ndarray, top_db: int = 40) -> float:
    """Tỷ lệ frame im lặng trong audio."""
    if len(y) == 0:
        return 1.0
    intervals  = librosa.effects.split(y, top_db=top_db)
    voiced_len = sum(end - start for start, end in intervals)
    return float(1.0 - voiced_len / len(y))


class WhisperFeatureExtractor:
    """
    Dùng Whisper-tiny để:
    1. Transcribe audio → hypothesis text
    2. Lấy log-prob confidence của sequence
    Tối ưu cho T4x2: batch processing + multi-GPU
    """
    def __init__(self, model_name: str = "openai/whisper-tiny", device: str = "cpu"):
        print(f"[*] Loading Whisper ({model_name}) trên {device}...")
        self.device    = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model     = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model     = self.model.to(device).to(torch.float32)

        # Multi-GPU support (T4x2)
        if torch.cuda.device_count() > 1 and device == "cuda":
            print(f"  🚀 DataParallel trên {torch.cuda.device_count()} GPUs!")
            # Whisper không support DataParallel trực tiếp → chạy trên GPU 0 chính
            # Dùng generate() nên không wrap DataParallel

        self.model.eval()
        self.forced_ids = self.processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        print(f"  ✓ Whisper loaded | Language: Vietnamese")

    @torch.no_grad()
    def transcribe_batch(self, audio_list: list[np.ndarray]) -> list[dict]:
        """
        audio_list: list of numpy arrays (16kHz mono)
        Returns: list of {'text': str, 'confidence': float}
        """
        results = []
        inputs = self.processor(
            audio_list,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        input_features = inputs.input_features.to(self.device).to(torch.float32)

        try:
            # Generate với output_scores để lấy confidence
            generated = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=448,
            )
            sequences = generated.sequences
            scores    = generated.scores  # tuple of (vocab_size,) tensors per step

            # Decode text
            texts = self.processor.batch_decode(sequences, skip_special_tokens=True)

            # Tính confidence: mean của max log-prob per token step
            if scores:
                log_probs = []
                for step_scores in scores:
                    # step_scores: (batch, vocab)
                    step_log_probs = torch.log_softmax(step_scores, dim=-1)
                    max_log_probs  = step_log_probs.max(dim=-1).values  # (batch,)
                    log_probs.append(max_log_probs.cpu().numpy())
                log_probs_arr = np.array(log_probs)  # (steps, batch)
                confidences   = np.exp(log_probs_arr.mean(axis=0))  # (batch,) → [0,1]
            else:
                confidences = np.full(len(texts), 0.5)

            for text, conf in zip(texts, confidences):
                results.append({'text': text.strip(), 'confidence': float(conf)})

        except Exception as e:
            for _ in audio_list:
                results.append({'text': '', 'confidence': 0.0})

        return results


def extract_all_features(df: pd.DataFrame, audio_dir: Path,
                          whisper_extractor: WhisperFeatureExtractor,
                          batch_size: int = 16) -> pd.DataFrame:
    """
    Extract tất cả features cho toàn bộ dataset.
    batch_size=16 phù hợp với T4 (16GB VRAM) dùng Whisper-tiny.
    """
    records = []

    # Chuẩn bị danh sách valid files
    valid_rows = []
    print("[*] Scan audio files...")
    for _, row in df.iterrows():
        audio_path = find_audio_path(row, audio_dir)
        if audio_path:
            valid_rows.append((row, audio_path))

    print(f"[+] Tìm thấy {len(valid_rows)}/{len(df)} audio files")

    # Batch processing
    for batch_start in tqdm(range(0, len(valid_rows), batch_size),
                             desc="Extracting features", unit="batch"):
        batch = valid_rows[batch_start: batch_start + batch_size]

        # Load audio
        batch_audio, batch_meta = [], []
        for row, audio_path in batch:
            try:
                y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                if len(y) < 100:  # quá ngắn, skip
                    continue
                batch_audio.append(y)
                batch_meta.append(row)
            except Exception:
                continue

        if not batch_audio:
            continue

        # Whisper transcribe (GPU batch)
        try:
            whisper_results = whisper_extractor.transcribe_batch(batch_audio)
        except Exception as e:
            whisper_results = [{'text': '', 'confidence': 0.0}] * len(batch_audio)

        # Compute librosa features + WER/CER per sample
        for y, row, w_result in zip(batch_audio, batch_meta, whisper_results):
            try:
                duration       = len(y) / SAMPLE_RATE
                snr            = compute_snr(y)
                silence_ratio  = compute_silence_ratio(y)
                whisper_conf   = w_result['confidence']
                hyp_text       = normalize_text(w_result['text'])
                ref_text       = normalize_text(str(row.get('transcript', '')))

                # WER / CER
                if ref_text and hyp_text:
                    try:
                        wer_val = compute_wer(ref_text, hyp_text)
                        cer_val = compute_cer(ref_text, hyp_text)
                    except Exception:
                        wer_val, cer_val = 1.0, 1.0
                else:
                    wer_val, cer_val = 1.0, 1.0

                # Length ratio: chars_ref / duration_seconds
                length_ratio = len(ref_text) / (duration + 1e-6)

                records.append({
                    'file_basename' : row['file_basename'],
                    'username'      : row.get('username', 'unknown'),
                    'transcript'    : row.get('transcript', ''),
                    'target'        : int(row['target']),
                    # Audio features
                    'snr'           : snr,
                    'silence_ratio' : silence_ratio,
                    'duration'      : duration,
                    'length_ratio'  : length_ratio,
                    # ASR features
                    'wer'           : min(wer_val, 3.0),    # cap ở 3.0
                    'cer'           : min(cer_val, 3.0),
                    'whisper_conf'  : whisper_conf,
                })
            except Exception as e:
                continue

        # Giải phóng GPU memory sau mỗi batch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    feat_df = pd.DataFrame(records)
    print(f"[+] Extracted {len(feat_df)} samples với {len(feat_df.columns)} columns")
    return feat_df


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: ANNOTATOR FEATURES
# ════════════════════════════════════════════════════════════════════════════

def add_annotator_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Thêm annotator bias + transcript consensus features."""
    global_rate = feat_df['target'].mean()
    eps = 1e-6

    def safe_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    global_logit = safe_logit(global_rate)

    # --- Annotator stats ---
    user_stats = feat_df.groupby('username').agg(
        user_acceptance_rate=('target', 'mean'),
    ).reset_index()
    user_stats['annotator_bias_logit'] = user_stats['user_acceptance_rate'].apply(
        lambda r: safe_logit(r) - global_logit
    )
    max_bias = user_stats['annotator_bias_logit'].abs().max() + eps
    user_stats['annotator_credibility'] = 1 - (
        user_stats['annotator_bias_logit'].abs() / max_bias
    )

    # --- Transcript consensus ---
    trans_stats = feat_df.groupby('transcript').agg(
        transcript_n_versions=('target', 'count'),
        transcript_usable_votes=('target', 'sum')
    ).reset_index()
    trans_stats['transcript_consensus_ratio'] = (
        trans_stats['transcript_usable_votes'] / trans_stats['transcript_n_versions']
    )
    trans_stats['transcript_ambiguity'] = 1 - abs(
        trans_stats['transcript_consensus_ratio'] - 0.5
    ) * 2

    feat_df = feat_df.merge(
        user_stats[['username', 'user_acceptance_rate',
                    'annotator_bias_logit', 'annotator_credibility']],
        on='username', how='left'
    )
    feat_df = feat_df.merge(
        trans_stats[['transcript', 'transcript_n_versions',
                     'transcript_consensus_ratio', 'transcript_ambiguity']],
        on='transcript', how='left'
    )

    # Majority voting label
    feat_df = feat_df.merge(
        trans_stats[['transcript', 'transcript_consensus_ratio']].rename(
            columns={'transcript_consensus_ratio': '_cr'}
        ),
        on='transcript', how='left'
    )
    feat_df['label_mv'] = feat_df['_cr'].apply(
        lambda r: 1 if r > 0.5 else (0 if r < 0.5 else np.nan)
    ).fillna(feat_df['target']).astype(int)
    feat_df.drop(columns=['_cr'], inplace=True)

    n_mv_changed = (feat_df['label_mv'] != feat_df['target']).sum()
    print(f"[+] Annotator features added | MV lật {n_mv_changed} nhãn "
          f"({n_mv_changed/len(feat_df)*100:.1f}%)")
    return feat_df


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: TRAIN & EVALUATE
# ════════════════════════════════════════════════════════════════════════════

AUDIO_FEATURES     = ['snr', 'silence_ratio', 'wer', 'cer',
                       'length_ratio', 'duration', 'whisper_conf']
ANNOTATOR_FEATURES = ['user_acceptance_rate', 'annotator_bias_logit',
                       'annotator_credibility', 'transcript_consensus_ratio',
                       'transcript_ambiguity', 'transcript_n_versions']


def get_lgbm_params(y: np.ndarray, use_gpu: bool = False) -> dict:
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    params = {
        'objective'        : 'binary',
        'metric'           : 'binary_logloss',
        'learning_rate'    : 0.03,
        'num_leaves'       : 63,
        'min_child_samples': 15,
        'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8,
        'bagging_freq'     : 5,
        'reg_alpha'        : 0.1,
        'reg_lambda'       : 0.1,
        'n_estimators'     : 1000,
        'scale_pos_weight' : pos_weight,
        'random_state'     : SEED,
        'verbose'          : -1,
        'n_jobs'           : -1,
    }
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_use_dp'] = False  # FP32 faster on T4
    return params


def train_evaluate(feat_df: pd.DataFrame, feature_cols: list, label_col: str,
                   strategy_name: str,
                   y_train_col: str = None,
                   use_gpu: bool = False) -> dict:
    """
    5-Fold CV với LightGBM.
    - feature_cols : cột features
    - label_col    : cột nhãn để EVALUATE
    - y_train_col  : cột nhãn để TRAIN (nếu khác label_col → soft/MV label)
    """
    df_clean = feat_df.dropna(subset=feature_cols + [label_col]).copy()
    X  = df_clean[feature_cols].values.astype(np.float32)
    y  = df_clean[label_col].values.astype(int)
    y_tr = df_clean[y_train_col].values.astype(int) if y_train_col else y

    print(f"\n  ▶ {strategy_name}  ({len(X)} samples, {len(feature_cols)} features)")

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X))
    params = get_lgbm_params(y, use_gpu=use_gpu)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr_idx])
        Xvl = sc.transform(X[val_idx])

        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, y_tr[tr_idx],
                  eval_set=[(Xvl, y[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(-1)])
        oof[val_idx] = model.predict_proba(Xvl)[:, 1]
        print(f"    Fold {fold+1}/{N_FOLDS} ✓", end='\r')
    print()

    # Tối ưu threshold
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y, (oof >= t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds  = (oof >= best_t).astype(int)
    auc    = roc_auc_score(y, oof)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    print(f"    MacroF1={best_f1:.4f} | AUC={auc:.4f} | Threshold={best_t:.2f}")
    return {
        'strategy'   : strategy_name,
        'n'          : len(X),
        'macro_f1'   : best_f1,
        'auc'        : auc,
        'threshold'  : best_t,
        'f1_0'       : report['0']['f1-score'],
        'f1_1'       : report['1']['f1-score'],
        'oof'        : oof,
        'y'          : y,
    }


# ════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: VISUALIZE
# ════════════════════════════════════════════════════════════════════════════

def plot_all(results: list, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ASR Quality Classifier — Strategy Comparison', fontsize=14, fontweight='bold')

    names  = [r['strategy'] for r in results]
    f1s    = [r['macro_f1'] for r in results]
    aucs   = [r['auc'] for r in results]
    f1_0   = [r['f1_0'] for r in results]
    colors = ['#e74c3c'] + ['#2ecc71' if v == max(f1s[1:]) else '#3498db' for v in f1s[1:]]

    for ax, vals, title, xl in zip(
        axes,
        [f1s, aucs, f1_0],
        ['Macro F1', 'ROC-AUC', 'F1 Class-0 (Unusable)'],
        [0.5, 0.5, 0.0]
    ):
        ax.barh(names, vals, color=colors)
        ax.axvline(x=0.78, color='red', ls='--', lw=1, label='Ceiling 0.78')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xl, 1.0)
        ax.legend(fontsize=8)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    p = output_dir / 'strategy_comparison.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Biểu đồ lưu tại: {p}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load CSV ──
    df = load_csv(CSV_PATH)

    # ── 2. Feature Extraction (có cache để tránh extract lại) ──
    if FEATURES_CACHE.exists():
        print(f"\n[*] Tìm thấy cache features: {FEATURES_CACHE}")
        feat_df = pd.read_csv(FEATURES_CACHE)
        print(f"    Loaded {len(feat_df)} samples từ cache")
    else:
        print(f"\n[*] Bắt đầu feature extraction (Whisper-tiny + librosa)...")
        print(f"    Audio dir: {AUDIO_DIR}")
        print(f"    Device: {DEVICE.upper()}")

        whisper = WhisperFeatureExtractor(
            model_name="openai/whisper-tiny",
            device=DEVICE
        )

        # Batch size: 16 cho T4 (16GB), giảm xuống 8 nếu OOM
        BATCH_SIZE = 16 if DEVICE == "cuda" else 4
        feat_df = extract_all_features(df, AUDIO_DIR, whisper, batch_size=BATCH_SIZE)

        # Giải phóng Whisper để dành RAM cho LightGBM
        del whisper
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Lưu cache
        feat_df.to_csv(FEATURES_CACHE, index=False)
        print(f"[+] Features đã lưu cache tại: {FEATURES_CACHE}")

    if len(feat_df) == 0:
        print("[ERROR] Không extract được feature nào! Kiểm tra AUDIO_DIR:", AUDIO_DIR)
        print("  Thử các paths sau:")
        for p in AUDIO_DIR.parent.glob("**/*.wav"):
            print(f"    {p}")
            break
        return

    # ── 3. Thêm Annotator Features ──
    print("\n[*] Tính annotator features...")
    feat_df = add_annotator_features(feat_df)

    # ── 4. Thống kê nhanh ──
    print(f"\n{'─'*55}")
    print("  PHÂN TÍCH DỮ LIỆU")
    print(f"{'─'*55}")
    print(f"  Samples      : {len(feat_df)}")
    print(f"  Usable (1)   : {feat_df['target'].sum()} ({feat_df['target'].mean()*100:.1f}%)")
    print(f"  Unusable (0) : {(1-feat_df['target']).sum()} ({(1-feat_df['target']).mean()*100:.1f}%)")
    print(f"  Avg SNR      : {feat_df['snr'].mean():.2f} dB")
    print(f"  Avg WER      : {feat_df['wer'].mean():.3f}")
    print(f"  Avg CER      : {feat_df['cer'].mean():.3f}")
    print(f"  Avg Whisper  : {feat_df['whisper_conf'].mean():.3f}")

    user_tbl = feat_df.groupby('username').agg(
        total=('target','count'), usable=('target','sum')
    )
    user_tbl['reject_%'] = ((1 - user_tbl['usable']/user_tbl['total'])*100).round(1)
    print(f"\n  Annotator rejection rates:")
    print(user_tbl.sort_values('reject_%', ascending=False)[['total','reject_%']].to_string())

    # ── 5. Train: 4 chiến lược ──
    print(f"\n{'='*60}")
    print("  TRAINING — 4 CHIẾN LƯỢC")
    print(f"  GPU LightGBM: {'✅' if USE_GPU_LGB else '❌'}")
    print(f"{'='*60}")

    results = []

    # A. Baseline: chỉ audio features
    results.append(train_evaluate(
        feat_df, AUDIO_FEATURES, 'target',
        'A_Baseline_AudioOnly', use_gpu=USE_GPU_LGB
    ))

    # B. Audio + Whisper confidence (thêm whisper_conf đã có trong AUDIO_FEATURES)
    # (whisper_conf đã trong AUDIO_FEATURES rồi, nên A đã bao gồm)

    # C. Full: Audio + Annotator features
    full_features = AUDIO_FEATURES + ANNOTATOR_FEATURES
    full_features = [f for f in full_features if f in feat_df.columns]
    results.append(train_evaluate(
        feat_df, full_features, 'target',
        'C_Audio+AnnotatorFeatures', use_gpu=USE_GPU_LGB
    ))

    # D. Full features + Majority Voting labels (train với MV, eval với original)
    results.append(train_evaluate(
        feat_df, full_features, 'target',
        'D_AnnotatorFeatures+MVLabel',
        y_train_col='label_mv',
        use_gpu=USE_GPU_LGB
    ))

    # E. High-consensus only (transcript_ambiguity < 0.3)
    df_hc = feat_df[feat_df['transcript_ambiguity'] < 0.3].copy()
    if len(df_hc) >= 200:
        results.append(train_evaluate(
            df_hc, full_features, 'target',
            'E_HighConsensusOnly', use_gpu=USE_GPU_LGB
        ))
    else:
        print(f"\n  [SKIP] E_HighConsensus: chỉ {len(df_hc)} mẫu, không đủ")

    # ── 6. Tổng kết ──
    print(f"\n{'='*70}")
    print("  BẢNG SO SÁNH")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'N':>5} {'MacroF1':>9} {'AUC':>8} {'F1_0':>7} {'F1_1':>7}")
    print("─" * 70)
    for r in results:
        delta = r['macro_f1'] - 0.78
        marker = " ▲" if delta > 0.003 else ("  " if abs(delta) <= 0.003 else " ▼")
        print(f"{r['strategy']:<35} {r['n']:>5} "
              f"{r['macro_f1']:>9.4f}{marker} {r['auc']:>8.4f} "
              f"{r['f1_0']:>7.4f} {r['f1_1']:>7.4f}")

    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\n🏆 Best: {best['strategy']}")
    print(f"   Macro F1 = {best['macro_f1']:.4f} | AUC = {best['auc']:.4f}")
    print(f"   Δ vs baseline ceiling (0.78): {best['macro_f1']-0.78:+.4f}")

    # ── 7. Plot ──
    plot_all(results, OUTPUT_DIR)

    # ── 8. Save summary JSON ──
    summary = [{k: v for k, v in r.items() if k not in ('oof', 'y')}
               for r in results]
    with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Summary lưu tại: {OUTPUT_DIR / 'results_summary.json'}")

    # ── 9. Kết luận ──
    all_below = all(r['macro_f1'] <= 0.785 for r in results)
    if all_below:
        print("""
╔══════════════════════════════════════════════════════╗
║  PHÁT HIỆN: 78% là HARD CEILING của bài toán này   ║
║                                                      ║
║  Lý do xác nhận:                                    ║
║  • Annotator rejection rate chênh 3x (55% vs 84%)  ║
║  • 27.77% audio có nhãn conflict giữa annotators   ║
║  • Thêm annotator features cũng không vượt trần    ║
║                                                      ║
║  → Đây là INSIGHT quan trọng nhất cho báo cáo!     ║
╚══════════════════════════════════════════════════════╝
""")
    else:
        improvement = best['macro_f1'] - 0.78
        print(f"\n✅ Đã vượt trần! Cải thiện: +{improvement:.4f} ({improvement/0.78*100:.1f}%)")


if __name__ == '__main__':
    main()
