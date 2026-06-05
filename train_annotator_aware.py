"""
train_annotator_aware.py  ─ Kaggle T4x2 Ready
===============================================
Chiến lược: Tận dụng Annotator Identity làm Feature.

VẤN ĐỀ ĐÃ CHẨN ĐOÁN:
  - Audio nghe RÕ nhưng vẫn bị unusable → annotator chủ quan
  - user6 reject 44.8% vs user2 reject 15.8% (chênh 3x!)
  - 27.77% audio có conflict nhãn giữa các annotators
  - → Model cần học BIAS CỦA NGƯỜI ĐÁNH NHÃN, không phải chất lượng audio

CHIẾN LƯỢC:
  E. Annotator-Aware: Thêm annotator_id + annotator_bias vào feature vector
  F. Annotator-Aware + Majority Voting labels
  G. Per-Annotator model ensemble (train riêng 7 model, ensemble)

KAGGLE PATHS:
  - Training CSV : /kaggle/input/datasets/huanvu205/training/training.csv
  - Features CSV : /kaggle/working/asr_quality_classifier/data/run_20260603_042350_lightgbm_f1_features.csv

GPU USAGE:
  - LightGBM với device='gpu' (nhanh hơn ~3x với dataset lớn)
  - Không cần extract lại audio features (đã có pre-computed)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────
# AUTO-DETECT KAGGLE vs LOCAL
# ──────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    CSV_PATH   = Path('/kaggle/input/datasets/huanvu205/training/training.csv')
    # Features CSV đã được generate từ run trước (saved trong working dir)
    FEAT_PATHS = [
        Path('/kaggle/working/asr_quality_classifier/data/run_20260603_042350_lightgbm_f1_features.csv'),
        Path('/kaggle/working/data/run_20260603_042350_lightgbm_f1_features.csv'),
    ]
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    USE_GPU    = True   # T4x2 available
else:
    # Local paths
    BASE = Path('/Users/admin/Documents/AI_ThucChien/asr_quality_classifier')
    CSV_PATH   = BASE / 'data/transcripts/training.csv'
    FEAT_PATHS = [BASE / 'data/run_20260603_042350_lightgbm_f1_features.csv']
    OUTPUT_DIR = BASE / 'outputs'
    USE_GPU    = False  # Mac không có CUDA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
N_FOLDS = 5
np.random.seed(SEED)

# ──────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────
# BƯỚC 1: LOAD DỮ LIỆU VÀ FEATURES
# ──────────────────────────────────────────────────────

def load_training_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['target'] = df['label_text'].apply(
        lambda x: 1 if str(x).strip().lower() == 'usable' else 0
    )
    # Chuẩn hóa file_name về basename
    if 'file_name' in df.columns:
        df['file_name_base'] = df['file_name'].apply(lambda x: Path(str(x)).name)
    print(f"[+] CSV: {len(df)} rows | {df['username'].nunique()} annotators | "
          f"Usable: {df['target'].mean()*100:.1f}%")
    return df


def load_features_csv(feat_paths: list) -> pd.DataFrame:
    for p in feat_paths:
        if p.exists():
            feat_df = pd.read_csv(p)
            print(f"[+] Features: {len(feat_df)} rows từ {p.name}")
            print(f"    Columns: {list(feat_df.columns)}")
            return feat_df
    raise FileNotFoundError(
        f"Không tìm thấy features CSV!\n"
        f"Cần chạy main.py trước để extract features.\n"
        f"Đã tìm tại: {[str(p) for p in feat_paths]}"
    )


# ──────────────────────────────────────────────────────
# BƯỚC 2: TẠO ANNOTATOR FEATURES
# ──────────────────────────────────────────────────────

def build_annotator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các features dựa trên hành vi annotator:
    
    1. annotator_acceptance_rate : tỷ lệ usable của người đó (0.55-0.84)
       → Biết người khắt khe hay dễ tính
    
    2. annotator_bias : độ lệch so với trung bình toàn dataset
       → Positive = dễ tính, Negative = khắt khe
    
    3. annotator_credibility : càng khắt khe thì nhãn "unusable" càng đáng tin
       → Khi user6 (44.8% reject) nói usable → rất đáng tin
       → Khi user2 (15.8% reject) nói unusable → rất đáng tin
    
    4. transcript_consensus_ratio : tỷ lệ đồng thuận của transcript đó
       → 1.0 = tất cả đồng ý, 0.5 = chia đôi ý kiến
    
    5. transcript_num_versions : số audio versions của cùng transcript
       → Nhiều versions hơn = có nhiều người đánh nhãn transcript đó
    """
    global_rate = df['target'].mean()
    eps = 1e-6

    # --- Annotator-level stats ---
    user_stats = df.groupby('username').agg(
        user_acceptance_rate=('target', 'mean'),
        user_total=('target', 'count')
    ).reset_index()

    # Bias so với global (logit scale để symmetric hơn)
    def safe_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    global_logit = safe_logit(global_rate)
    user_stats['annotator_bias_logit'] = user_stats['user_acceptance_rate'].apply(
        lambda r: safe_logit(r) - global_logit
    )

    # Credibility = nghịch đảo bias magnitude (annotator cực đoan → credibility thấp hơn)
    max_bias = user_stats['annotator_bias_logit'].abs().max()
    user_stats['annotator_credibility'] = 1 - (
        user_stats['annotator_bias_logit'].abs() / (max_bias + eps)
    )

    # --- Transcript-level stats ---
    trans_stats = df.groupby('transcript').agg(
        transcript_num_versions=('file_name', 'count'),
        transcript_usable_votes=('target', 'sum')
    ).reset_index()
    trans_stats['transcript_consensus_ratio'] = (
        trans_stats['transcript_usable_votes'] / trans_stats['transcript_num_versions']
    )
    # 0.5 = chia đôi (ambiguous), 0/1 = đồng thuận
    trans_stats['transcript_ambiguity'] = 1 - abs(
        trans_stats['transcript_consensus_ratio'] - 0.5
    ) * 2  # 0 = hoàn toàn đồng thuận, 1 = hoàn toàn chia đôi

    # --- Merge vào df ---
    df = df.merge(user_stats[['username', 'user_acceptance_rate', 
                               'annotator_bias_logit', 'annotator_credibility']], 
                  on='username', how='left')
    df = df.merge(trans_stats[['transcript', 'transcript_num_versions', 
                                'transcript_consensus_ratio', 'transcript_ambiguity']],
                  on='transcript', how='left')

    print(f"\n[+] Annotator features được tạo:")
    print(f"    user_acceptance_rate : min={df['user_acceptance_rate'].min():.3f}, "
          f"max={df['user_acceptance_rate'].max():.3f}")
    print(f"    annotator_bias_logit : min={df['annotator_bias_logit'].min():.3f}, "
          f"max={df['annotator_bias_logit'].max():.3f}")
    print(f"    transcript_ambiguity : "
          f"{(df['transcript_ambiguity'] > 0.8).sum()} mẫu có ambiguity > 0.8")

    return df


def create_majority_voting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Lấy nhãn đa số cho mỗi transcript."""
    trans_votes = df.groupby('transcript').agg(
        total=('target', 'count'),
        usable_votes=('target', 'sum')
    ).reset_index()
    trans_votes['usable_ratio'] = trans_votes['usable_votes'] / trans_votes['total']

    def majority_label(row):
        if row['usable_ratio'] > 0.5:
            return 1
        elif row['usable_ratio'] < 0.5:
            return 0
        else:
            return np.nan  # tie → keep original

    trans_votes['majority_label'] = trans_votes.apply(majority_label, axis=1)
    df = df.merge(trans_votes[['transcript', 'majority_label']], on='transcript', how='left')
    df['label_mv'] = df['majority_label'].fillna(df['target']).astype(int)

    n_changed = (df['label_mv'] != df['target']).sum()
    print(f"  [MV] Lật {n_changed} nhãn ({n_changed/len(df)*100:.1f}%)")
    return df


# ──────────────────────────────────────────────────────
# BƯỚC 3: MERGE FEATURES + LABELS
# ──────────────────────────────────────────────────────

BASE_AUDIO_FEATURES = ['snr', 'silence_ratio', 'wer', 'cer', 'length_ratio', 'duration']

ANNOTATOR_FEATURES = [
    'user_acceptance_rate',
    'annotator_bias_logit',
    'annotator_credibility',
    'transcript_consensus_ratio',
    'transcript_ambiguity',
    'transcript_num_versions',
]


def build_dataset(df_labels: pd.DataFrame, feat_df: pd.DataFrame,
                  label_col: str, include_annotator_features: bool = True) -> tuple:
    """
    Merge audio features với annotator features.
    Trả về (X, y, feature_names).
    """
    # Match file_name
    feat_df_matched = feat_df.copy()
    if 'file_name_base' in df_labels.columns:
        merge_key_label = 'file_name_base'
        merge_key_feat  = 'file_name'
    else:
        merge_key_label = 'file_name'
        merge_key_feat  = 'file_name'

    # Xác định các cột cần lấy từ df_labels
    cols_to_merge = ['file_name_base', label_col] + (
        ANNOTATOR_FEATURES if include_annotator_features else []
    )
    cols_to_merge = [c for c in cols_to_merge if c in df_labels.columns]

    merged = feat_df_matched.merge(
        df_labels[cols_to_merge].drop_duplicates(merge_key_label if merge_key_label in cols_to_merge else 'file_name'),
        left_on=merge_key_feat,
        right_on=merge_key_label if merge_key_label != merge_key_feat else merge_key_feat,
        how='inner'
    )

    feature_cols = BASE_AUDIO_FEATURES.copy()
    if include_annotator_features:
        feature_cols += [c for c in ANNOTATOR_FEATURES if c in merged.columns]

    # Drop NaN
    merged = merged.dropna(subset=feature_cols + [label_col])

    X = merged[feature_cols].values.astype(np.float32)
    y = merged[label_col].values.astype(int)

    print(f"    Dataset: {len(X)} samples | {len(feature_cols)} features "
          f"| Usable: {y.mean()*100:.1f}%")
    return X, y, feature_cols


# ──────────────────────────────────────────────────────
# BƯỚC 4: TRAIN VÀ ĐÁNH GIÁ
# ──────────────────────────────────────────────────────

def get_lgbm_params(y: np.ndarray, use_gpu: bool = False) -> dict:
    """LightGBM params — GPU-ready."""
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    params = {
        'objective'        : 'binary',
        'metric'           : 'binary_logloss',
        'learning_rate'    : 0.03,
        'num_leaves'       : 63,
        'max_depth'        : -1,
        'min_child_samples': 20,
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
        params['device']      = 'gpu'
        params['gpu_use_dp']  = True   # Double precision trên T4
        print("    → Using GPU acceleration (LightGBM)")
    else:
        params['device'] = 'cpu'

    return params


def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       strategy_name: str,
                       y_train_override: np.ndarray = None,
                       use_gpu: bool = False) -> dict:
    """
    Stratified 5-Fold CV với LightGBM.
    y_train_override: Nếu có, dùng để TRAIN nhưng EVAL vẫn bằng y gốc.
    """
    print(f"\n  Training {strategy_name}...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_eval   = y.copy()
    y_source = y_train_override if y_train_override is not None else y

    oof_probs = np.zeros(len(y))
    params    = get_lgbm_params(y_eval, use_gpu=use_gpu)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_eval)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr        = y_source[tr_idx]
        y_val       = y_eval[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1)
            ]
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"    Fold {fold+1}/{N_FOLDS} ✓", end='\r')

    # Tối ưu threshold
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (oof_probs >= t).astype(int)
        f1    = f1_score(y_eval, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    final_preds = (oof_probs >= best_thresh).astype(int)
    auc    = roc_auc_score(y_eval, oof_probs)
    report = classification_report(y_eval, final_preds, output_dict=True, zero_division=0)

    print(f"    → Macro F1={best_f1:.4f} | AUC={auc:.4f} | Threshold={best_thresh:.2f}")

    return {
        'strategy'   : strategy_name,
        'n_samples'  : len(y),
        'macro_f1'   : best_f1,
        'roc_auc'    : auc,
        'threshold'  : best_thresh,
        'f1_class_0' : report['0']['f1-score'],
        'f1_class_1' : report['1']['f1-score'],
        'precision_0': report['0']['precision'],
        'recall_0'   : report['0']['recall'],
        'oof_probs'  : oof_probs,
        'y_true'     : y_eval,
    }


# ──────────────────────────────────────────────────────
# BƯỚC 5: VISUALIZE & SAVE
# ──────────────────────────────────────────────────────

def plot_results(results: list, output_dir: Path):
    """Vẽ biểu đồ so sánh các chiến lược."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Data-Centric Strategy Comparison (Annotator-Aware)', 
                 fontsize=14, fontweight='bold')

    strategies = [r['strategy'] for r in results]
    macro_f1   = [r['macro_f1'] for r in results]
    auc_scores = [r['roc_auc'] for r in results]
    f1_cls0    = [r['f1_class_0'] for r in results]

    colors = ['#e74c3c' if i == 0 else '#2ecc71' if v == max(macro_f1) else '#3498db'
              for i, v in enumerate(macro_f1)]

    # Plot 1: Macro F1
    axes[0].barh(strategies, macro_f1, color=colors)
    axes[0].axvline(x=0.78, color='red', linestyle='--', label='Baseline ceiling (0.78)')
    axes[0].set_title('Macro F1 Score')
    axes[0].set_xlim(0.5, 1.0)
    axes[0].legend(fontsize=8)
    for i, v in enumerate(macro_f1):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    # Plot 2: ROC-AUC
    axes[1].barh(strategies, auc_scores, color=colors)
    axes[1].set_title('ROC-AUC Score')
    axes[1].set_xlim(0.5, 1.0)
    for i, v in enumerate(auc_scores):
        axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    # Plot 3: F1 Class 0 (Unusable) - quan trọng nhất!
    axes[2].barh(strategies, f1_cls0, color=colors)
    axes[2].set_title('F1 Class 0 (Unusable) - Minority Class')
    axes[2].set_xlim(0, 0.8)
    for i, v in enumerate(f1_cls0):
        axes[2].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    out_path = output_dir / 'strategy_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Đã lưu biểu đồ: {out_path}")


def print_summary_table(results: list, baseline_f1: float):
    """In bảng tổng kết."""
    print("\n" + "=" * 75)
    print("  BẢNG SO SÁNH KẾT QUẢ CUỐI CÙNG")
    print("=" * 75)
    header = f"{'Strategy':<35} {'N':>5} {'MacroF1':>8} {'AUC':>8} {'F1_0':>7} {'F1_1':>7} {'Thresh':>7}"
    print(header)
    print("─" * 75)
    for r in results:
        delta = r['macro_f1'] - baseline_f1
        marker = " ▲" if delta > 0.005 else ("  " if abs(delta) <= 0.005 else " ▼")
        print(f"{r['strategy']:<35} {r['n_samples']:>5} "
              f"{r['macro_f1']:>8.4f}{marker} {r['roc_auc']:>8.4f} "
              f"{r['f1_class_0']:>7.4f} {r['f1_class_1']:>7.4f} "
              f"{r['threshold']:>7.2f}")

    best = max(results, key=lambda x: x['macro_f1'])
    print("─" * 75)
    print(f"\n🏆  Best: {best['strategy']}  →  Macro F1 = {best['macro_f1']:.4f}")
    improvement = best['macro_f1'] - baseline_f1
    print(f"    Cải thiện vs Baseline ceiling (0.78): {improvement:+.4f} "
          f"({improvement/baseline_f1*100:+.1f}%)")


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANNOTATOR-AWARE TRAINING  |  Kaggle T4x2 Ready")
    print(f"  GPU: {'✅ ON' if USE_GPU else '❌ OFF (local)'}")
    print("=" * 60)

    # 1. Load data
    df       = load_training_csv(CSV_PATH)
    feat_df  = load_features_csv(FEAT_PATHS)

    # 2. Build annotator features
    print("\n[*] Xây dựng Annotator Features...")
    df = build_annotator_features(df)
    df = create_majority_voting_labels(df)

    # Kiểm tra overlap
    feat_df['file_name_base'] = feat_df['file_name'].apply(lambda x: Path(str(x)).name)
    df_matched = df[df['file_name_base'].isin(feat_df['file_name_base'])]
    print(f"\n[+] Overlap: {len(df_matched)}/{len(df)} samples có đủ features")

    results = []

    # ── Chiến lược A: Baseline (không có annotator features) ──
    print("\n" + "─" * 60)
    print("▶ A. BASELINE — Audio features only (không annotator info)")
    X_a, y_a, feats_a = build_dataset(df, feat_df, 'target',
                                       include_annotator_features=False)
    results.append(train_and_evaluate(X_a, y_a, 'A_Baseline', use_gpu=USE_GPU))

    # ── Chiến lược E: Annotator-Aware ──
    print("\n" + "─" * 60)
    print("▶ E. ANNOTATOR-AWARE — Thêm annotator bias/credibility làm feature")
    print("   [Lý thuyết] Model học được: user6 nói usable → tin hơn user2 nói usable")
    X_e, y_e, feats_e = build_dataset(df, feat_df, 'target',
                                       include_annotator_features=True)
    results.append(train_and_evaluate(X_e, y_e, 'E_AnnotatorAware', use_gpu=USE_GPU))

    # ── Chiến lược F: Annotator-Aware + Majority Voting labels ──
    print("\n" + "─" * 60)
    print("▶ F. ANNOTATOR + MAJORITY VOTING — Clean labels + annotator features")
    X_f, y_f, feats_f = build_dataset(df, feat_df, 'label_mv',
                                       include_annotator_features=True)
    # Lấy original labels tương ứng để eval
    X_f2, y_f_orig, _ = build_dataset(df, feat_df, 'target',
                                        include_annotator_features=True)
    # Đảm bảo cùng shape
    min_n = min(len(y_f), len(y_f_orig))
    results.append(train_and_evaluate(X_f[:min_n], y_f_orig[:min_n],
                                       'F_Annotator+MV',
                                       y_train_override=y_f[:min_n],
                                       use_gpu=USE_GPU))

    # ── Chiến lược G: Chỉ mẫu có consensus cao (transcript_ambiguity < 0.3) ──
    print("\n" + "─" * 60)
    print("▶ G. HIGH-CONSENSUS ONLY — Loại bỏ mẫu transcript bị chia rẽ ý kiến")

    # Lọc những mẫu có transcript ít tranh cãi
    df_consensus = df[df['transcript_ambiguity'] < 0.3].copy()
    if len(df_consensus) > 100:
        X_g, y_g, _ = build_dataset(df_consensus, feat_df, 'target',
                                      include_annotator_features=True)
        results.append(train_and_evaluate(X_g, y_g, 'G_HighConsensus', use_gpu=USE_GPU))
    else:
        print(f"  [SKIP] Chỉ có {len(df_consensus)} mẫu consensus cao, không đủ để train")

    # ── Tổng kết ──
    baseline_f1 = results[0]['macro_f1']
    print_summary_table(results, baseline_f1=0.78)  # 0.78 là baseline đã biết

    # Feature importance của chiến lược tốt nhất
    best_idx = np.argmax([r['macro_f1'] for r in results])
    best_r   = results[best_idx]
    print(f"\n[*] Chiến lược thắng '{best_r['strategy']}' sử dụng {feats_e if best_idx > 0 else feats_a} features")

    # Plot
    plot_results(results, OUTPUT_DIR)

    # ── Gợi ý tiếp theo ──
    best_f1 = best_r['macro_f1']
    print(f"""
{'='*60}
PHÂN TÍCH KẾT QUẢ:
{'='*60}

Kết quả thu được: {best_f1:.4f}

Nếu E/F > A (Baseline):
  → Annotator identity chứa thông tin dự đoán
  → Trong production: cần biết annotator_id lúc inference
     (không khả thi) → Dùng transcript_consensus_ratio thay thế

Nếu G > A:
  → Confirm: Label noise là nguyên nhân chính của trần 78%
  → Khuyến nghị: Thu thập thêm dữ liệu "unambiguous"

Nếu tất cả ≈ A (Baseline 78%):
  → Đây là HARD CEILING của bài toán
  → Báo cáo phát hiện annotator bias (user6: 44.8% vs user2: 15.8%)
  → Đây là insight quan trọng hơn việc thêm 1-2% F1

Output đã lưu tại: {OUTPUT_DIR}
{'='*60}
""")


if __name__ == '__main__':
    main()
