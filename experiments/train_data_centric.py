"""
train_data_centric.py
=====================
Chiến lược Data-Centric để vượt trần 78% Macro F1.

Vấn đề đã chẩn đoán:
  - 27.77% audio có nhãn mâu thuẫn (conflict)
  - Annotator bias: user6 reject 44.8% vs user2 reject 15.8% (chênh 3x)
  - 63.5% xung đột là 50/50 → label noise thật sự

3 Chiến lược được so sánh:
  A. Baseline: Nhãn gốc (raw label)
  B. Majority Voting: Lấy nhãn đa số cho mỗi transcript
  C. Soft Label (Label Smoothing): Chuẩn hóa nhãn theo hành vi annotator
  D. Clean-only: Chỉ train trên mẫu có 100% đồng thuận
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Phụ thuộc ---
try:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("[!] Thiếu thư viện. Chạy: pip install lightgbm scikit-learn pandas numpy")
    exit(1)

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
CSV_PATH = Path("data/transcripts/training.csv")
FEATURES_CSV = Path("data/run_20260603_042350_lightgbm_f1_features.csv")
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


# ─────────────────────────────────────────────
# BƯỚC 1: TẢI VÀ PHÂN TÍCH DỮ LIỆU
# ─────────────────────────────────────────────
def load_and_analyze(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['target'] = df['label_text'].apply(lambda x: 1 if str(x).strip().lower() == 'usable' else 0)
    print(f"[+] Đã tải {len(df)} mẫu, {df['username'].nunique()} annotators")
    return df


# ─────────────────────────────────────────────
# BƯỚC 2: TẠO CÁC PHIÊN BẢN NHÃN KHÁC NHAU
# ─────────────────────────────────────────────

def create_majority_voting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chiến lược B: Majority Voting per transcript.
    
    Với mỗi transcript, lấy nhãn đa số từ tất cả audio versions.
    Ưu tiên: 
      - Nếu đa số vote Usable   → nhãn = 1
      - Nếu đa số vote Unusable → nhãn = 0  
      - Nếu 50/50               → giữ nguyên nhãn gốc (ambiguous)
    """
    transcript_votes = df.groupby('transcript').agg(
        total=('target', 'count'),
        usable_votes=('target', 'sum')
    ).reset_index()
    transcript_votes['usable_ratio'] = transcript_votes['usable_votes'] / transcript_votes['total']
    
    # Majority label: >0.5 → usable=1, <0.5 → unusable=0, ==0.5 → NaN (ambiguous)
    def majority_label(row):
        if row['usable_ratio'] > 0.5:
            return 1
        elif row['usable_ratio'] < 0.5:
            return 0
        else:
            return np.nan  # 50/50, không chắc chắn
    
    transcript_votes['majority_label'] = transcript_votes.apply(majority_label, axis=1)
    
    # Merge lại vào df gốc
    df_mv = df.merge(
        transcript_votes[['transcript', 'majority_label', 'usable_ratio']],
        on='transcript', how='left'
    )
    
    # Điền NaN (50/50) bằng nhãn gốc
    df_mv['label_mv'] = df_mv['majority_label'].fillna(df_mv['target']).astype(int)
    
    n_changed = (df_mv['label_mv'] != df_mv['target']).sum()
    print(f"  [MajorityVote] Đã lật {n_changed} nhãn ({n_changed/len(df)*100:.1f}%) so với nhãn gốc")
    return df_mv


def create_annotator_normalized_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chiến lược C: Soft Label - Chuẩn hóa theo hành vi annotator.
    
    Ý tưởng: Nhãn của một annotator khắt khe (user6, reject 44.8%) 
    nên được tin tưởng hơn khi họ nói "Usable" so với annotator dễ tính (user2).
    
    Công thức:
      p_usable_adjusted = sigmoid( logit(usable_ratio_transcript) 
                                   - annotator_bias )
    
    annotator_bias = log(acceptance_rate / (1-acceptance_rate)) 
                   - log(global_acceptance_rate / (1-global_acceptance_rate))
    → Positive bias: annotator dễ tính → điều chỉnh giảm nhãn usable
    → Negative bias: annotator khắt khe → điều chỉnh tăng nhãn usable
    """
    global_rate = df['target'].mean()  # ~0.74
    
    # Tính acceptance rate mỗi annotator
    user_rates = df.groupby('username')['target'].mean().reset_index()
    user_rates.columns = ['username', 'user_acceptance_rate']
    
    # Tính bias (logit scale)
    eps = 1e-6
    def safe_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    global_logit = safe_logit(global_rate)
    user_rates['annotator_bias'] = user_rates['user_acceptance_rate'].apply(
        lambda r: safe_logit(r) - global_logit
    )
    
    df_soft = df.merge(user_rates[['username', 'user_acceptance_rate', 'annotator_bias']], 
                       on='username', how='left')
    
    # Soft label = target hiệu chỉnh theo bias annotator
    # Nếu annotator dễ tính (bias > 0) và label = 1 → credibility thấp hơn
    # Nếu annotator khắt khe (bias < 0) và label = 1 → credibility cao hơn
    def compute_soft_label(row):
        raw_logit = safe_logit(row['target'] * 0.9 + 0.05)  # label smoothing nhẹ
        adjusted_logit = raw_logit - 0.5 * row['annotator_bias']
        return 1 / (1 + np.exp(-adjusted_logit))  # sigmoid
    
    df_soft['soft_label'] = df_soft.apply(compute_soft_label, axis=1)
    
    # Hard threshold để có nhãn cứng để so sánh
    df_soft['label_soft_hard'] = (df_soft['soft_label'] > 0.5).astype(int)
    
    n_changed = (df_soft['label_soft_hard'] != df_soft['target']).sum()
    print(f"  [SoftLabel]    Đã điều chỉnh {n_changed} nhãn ({n_changed/len(df)*100:.1f}%) so với nhãn gốc")
    return df_soft


def create_clean_only_labels(df: pd.DataFrame) -> tuple:
    """
    Chiến lược D: Chỉ giữ lại mẫu có transcript 100% đồng thuận.
    """
    transcript_stats = df.groupby('transcript').agg(
        total=('target', 'count'),
        usable_votes=('target', 'sum')
    ).reset_index()
    transcript_stats['usable_ratio'] = transcript_stats['usable_votes'] / transcript_stats['total']
    
    clean_transcripts = transcript_stats[
        (transcript_stats['usable_ratio'] == 1.0) | (transcript_stats['usable_ratio'] == 0.0)
    ]['transcript']
    
    df_clean = df[df['transcript'].isin(clean_transcripts)].copy()
    print(f"  [CleanOnly]    Giữ lại {len(df_clean)}/{len(df)} mẫu ({len(df_clean)/len(df)*100:.1f}%)")
    return df_clean


# ─────────────────────────────────────────────
# BƯỚC 3: LOAD FEATURES
# ─────────────────────────────────────────────

def load_features(features_csv: Path) -> pd.DataFrame:
    """Load pre-computed features từ file CSV."""
    feat_df = pd.read_csv(features_csv)
    print(f"[+] Đã tải {len(feat_df)} mẫu features: {list(feat_df.columns)}")
    return feat_df


def merge_labels_with_features(df_labels: pd.DataFrame, feat_df: pd.DataFrame, 
                                label_col: str) -> tuple:
    """
    Merge nhãn với features theo file_name.
    Trả về X (features), y (labels), và index hợp lệ.
    """
    merged = feat_df.merge(
        df_labels[['file_name', label_col]].drop_duplicates('file_name'), 
        on='file_name', how='inner'
    )
    
    feature_cols = ['snr', 'silence_ratio', 'wer', 'cer', 'length_ratio', 'duration']
    X = merged[feature_cols].values
    y = merged[label_col].values
    
    return X, y, merged['file_name'].values


# ─────────────────────────────────────────────
# BƯỚC 4: TRAIN VÀ ĐÁNH GIÁ
# ─────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray, 
                       label_col: str, 
                       soft_labels: np.ndarray = None,
                       strategy_name: str = "Baseline") -> dict:
    """
    Stratified 5-Fold CV với LightGBM.
    Nếu soft_labels được cung cấp, dùng để train nhưng evaluate bằng y (nhãn gốc).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Nhãn hard để eval (luôn dùng nhãn gốc để so sánh công bằng)
    y_eval = y.copy()
    
    # Nhãn để train (có thể là soft)
    y_train_source = soft_labels if soft_labels is not None else y
    
    oof_probs = np.zeros(len(y))
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_estimators': 300,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': SEED,
        'verbose': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_eval)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y_train_source[train_idx]  # soft or hard labels for training
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, 
                  eval_set=[(X_val, y_eval[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), 
                              lgb.log_evaluation(-1)])
        
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Tối ưu threshold
    best_threshold, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (oof_probs >= thresh).astype(int)
        f1 = f1_score(y_eval, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    final_preds = (oof_probs >= best_threshold).astype(int)
    auc = roc_auc_score(y_eval, oof_probs)
    
    report = classification_report(y_eval, final_preds, output_dict=True)
    
    return {
        'strategy': strategy_name,
        'n_samples': len(y),
        'macro_f1': best_f1,
        'roc_auc': auc,
        'threshold': best_threshold,
        'f1_class_0': report['0']['f1-score'],
        'f1_class_1': report['1']['f1-score'],
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'oof_probs': oof_probs,
        'y_true': y_eval,
    }


# ─────────────────────────────────────────────
# MAIN: CHẠY TẤT CẢ CÁC CHIẾN LƯỢC
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DATA-CENTRIC EXPERIMENT: Vượt trần 78% Macro F1")
    print("=" * 60)
    
    # Load data
    df = load_and_analyze(CSV_PATH)
    feat_df = load_features(FEATURES_CSV)
    
    # Merge file_name với labels
    # CSV gốc có cột 'file_name' theo dạng 'data2/folder/file.wav'
    # Features CSV có file_name theo dạng 'clone8.wav'
    # → Cần extract basename
    if 'file_name' not in df.columns and 'audio_path' in df.columns:
        df['file_name'] = df['audio_path'].apply(lambda x: Path(str(x)).name)
    elif 'file_name' in df.columns:
        df['file_name'] = df['file_name'].apply(lambda x: Path(str(x)).name)
    
    print(f"\n[*] Kiểm tra overlap giữa labels và features...")
    label_files = set(df['file_name'].unique())
    feat_files = set(feat_df['file_name'].unique())
    overlap = label_files & feat_files
    print(f"    Labels: {len(label_files)} files | Features: {len(feat_files)} files | Overlap: {len(overlap)} files")
    
    if len(overlap) == 0:
        print("\n[!] CẢNH BÁO: Không có overlap giữa file names!")
        print("    Sample labels:", list(label_files)[:5])
        print("    Sample features:", list(feat_files)[:5])
        print("\n[*] Thử match theo thứ tự index (nếu cùng thứ tự)...")
        # Fallback: match theo vị trí nếu cùng số lượng
        if len(df) == len(feat_df):
            df['file_name'] = feat_df['file_name'].values
            print(f"    → Matched {len(df)} mẫu theo index")
        else:
            print("[ERROR] Không thể match. Kiểm tra lại cột file_name trong CSV.")
            return
    
    results = []
    
    print("\n" + "─" * 60)
    print("▶ Chiến lược A: BASELINE (nhãn gốc)")
    print("─" * 60)
    X_base, y_base, _ = merge_labels_with_features(df, feat_df, 'target')
    result_a = train_and_evaluate(X_base, y_base, 'target', strategy_name="A_Baseline")
    results.append(result_a)
    
    print("\n" + "─" * 60)
    print("▶ Chiến lược B: MAJORITY VOTING")
    print("─" * 60)
    df_mv = create_majority_voting_labels(df)
    X_mv, y_mv, _ = merge_labels_with_features(df_mv, feat_df, 'label_mv')
    y_mv_orig, _, _ = merge_labels_with_features(df_mv, feat_df, 'target')
    result_b = train_and_evaluate(X_mv, y_mv_orig, 'label_mv',
                                   strategy_name="B_MajorityVoting")
    # Eval bằng nhãn MV nhưng cũng report trên nhãn gốc
    result_b_with_mv = train_and_evaluate(X_mv, y_mv, 'label_mv',
                                           strategy_name="B_MajorityVoting_EvalMV")
    results.append(result_b)
    results.append(result_b_with_mv)
    
    print("\n" + "─" * 60)
    print("▶ Chiến lược C: SOFT LABEL (annotator-normalized)")
    print("─" * 60)
    df_soft = create_annotator_normalized_labels(df)
    X_soft, y_soft_hard, _ = merge_labels_with_features(df_soft, feat_df, 'label_soft_hard')
    # Lấy soft labels tương ứng
    soft_merged = feat_df.merge(
        df_soft[['file_name', 'soft_label', 'target']].drop_duplicates('file_name'),
        on='file_name', how='inner'
    )
    y_soft_vals = soft_merged['soft_label'].values
    y_orig_vals = soft_merged['target'].values
    X_soft_clean = soft_merged[['snr', 'silence_ratio', 'wer', 'cer', 'length_ratio', 'duration']].values
    result_c = train_and_evaluate(X_soft_clean, y_orig_vals, 'soft_label',
                                   soft_labels=y_soft_vals,
                                   strategy_name="C_SoftLabel")
    results.append(result_c)
    
    print("\n" + "─" * 60)
    print("▶ Chiến lược D: CLEAN-ONLY (chỉ mẫu 100% đồng thuận)")
    print("─" * 60)
    df_clean = create_clean_only_labels(df)
    X_clean, y_clean, _ = merge_labels_with_features(df_clean, feat_df, 'target')
    result_d = train_and_evaluate(X_clean, y_clean, 'target',
                                   strategy_name="D_CleanOnly")
    results.append(result_d)
    
    # ─── TỔNG KẾT ───
    print("\n" + "=" * 70)
    print("  BẢNG SO SÁNH KẾT QUẢ CÁC CHIẾN LƯỢC")
    print("=" * 70)
    print(f"{'Strategy':<30} {'N':<6} {'MacroF1':>8} {'AUC':>8} {'F1_cls0':>8} {'F1_cls1':>8} {'Thresh':>7}")
    print("─" * 70)
    for r in results:
        print(f"{r['strategy']:<30} {r['n_samples']:<6} "
              f"{r['macro_f1']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['f1_class_0']:>8.4f} {r['f1_class_1']:>8.4f} "
              f"{r['threshold']:>7.2f}")
    
    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\n🏆 Chiến lược tốt nhất: {best['strategy']}")
    print(f"   Macro F1 = {best['macro_f1']:.4f} | AUC = {best['roc_auc']:.4f}")
    
    # So sánh với baseline
    baseline_f1 = results[0]['macro_f1']
    improvement = best['macro_f1'] - baseline_f1
    print(f"\n   Cải thiện vs Baseline: {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")
    
    # ─── PHÂN TÍCH NGUYÊN NHÂN NHIỄU ───
    print("\n" + "=" * 60)
    print("  PHÂN TÍCH ANNOTATOR BIAS (Thông tin bổ sung)")
    print("=" * 60)
    
    global_rate = df['target'].mean()
    user_stats = df.groupby('username')['target'].mean().reset_index()
    user_stats.columns = ['username', 'acceptance_rate']
    user_stats['bias_vs_avg'] = user_stats['acceptance_rate'] - global_rate
    user_stats['label_effect'] = user_stats['bias_vs_avg'].apply(
        lambda b: f"Dễ tính (+{b*100:.1f}%)" if b > 0 else f"Khắt khe ({b*100:.1f}%)"
    )
    user_stats = user_stats.sort_values('acceptance_rate')
    
    print(f"\nGlobal acceptance rate: {global_rate*100:.1f}%\n")
    for _, row in user_stats.iterrows():
        print(f"  {row['username']}: {row['acceptance_rate']*100:.1f}% → {row['label_effect']}")
    
    print(f"""
📋 KHUYẾN NGHỊ TIẾP THEO:
  1. Nếu chiến lược D (CleanOnly) tốt hơn → Confirm: Nhiễu là vấn đề số 1
     → Thu thập thêm dữ liệu "unambiguous" thay vì thêm model phức tạp
     
  2. Nếu chiến lược C (SoftLabel) tốt hơn → Annotator bias là vấn đề số 1  
     → Cân nhắc re-labeling với consensus protocol (ít nhất 3 annotators/audio)
     
  3. Nếu không chiến lược nào vượt Baseline đáng kể:
     → Trần 78% là HARD CEILING của bài toán với features hiện tại
     → Cần thêm features mới (DNSMOS, spectral, pitch variance)
""")


if __name__ == '__main__':
    main()
