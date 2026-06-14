"""
phase2_baseline.py — Phase 2: Tabular LightGBM Baseline.

Extracts 37 acoustic + 6 cross-modal features in parallel, caches them,
and trains a LightGBM classifier using 5-fold GroupKFold.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import config, logger
from src.data_loader import load_data, get_kfold_splits
from src.audio_features import process_audio_file, get_acoustic_feature_keys, get_crossmodal_feature_keys
from src.classifier import TabularLGBMClassifier
from src.evaluator import compute_metrics, plot_confusion_matrix, run_error_analysis

def extract_features_parallel(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts features in parallel using ThreadPoolExecutor."""
    n_samples = len(df)
    logger.info(f"Starting parallel feature extraction for {n_samples} files...")
    
    # Pre-allocate arrays
    ac_keys = get_acoustic_feature_keys()
    cm_keys = get_crossmodal_feature_keys()
    
    ac_feats = np.zeros((n_samples, len(ac_keys)), dtype=np.float32)
    cm_feats = np.zeros((n_samples, len(cm_keys)), dtype=np.float32)
    
    # Process helper
    def worker(idx: int, path: str, transcript: str):
        ac_arr, cm_arr = process_audio_file(path, transcript, config)
        return idx, ac_arr, cm_arr

    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(worker, i, row["absolute_path"], row["transcript"])
            for i, row in df.iterrows()
        ]
        
        for future in tqdm(as_completed(futures), total=n_samples, desc="Extracting features"):
            idx, ac_arr, cm_arr = future.result()
            ac_feats[idx] = ac_arr
            cm_feats[idx] = cm_arr
            
    return ac_feats, cm_feats

def load_or_extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Loads features from cache if available, else extracts and caches them."""
    cache_dir = config.paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    ac_cache_path = cache_dir / "acoustic_features.npy"
    cm_cache_path = cache_dir / "crossmodal_features.npy"
    
    if ac_cache_path.exists() and cm_cache_path.exists():
        logger.info("Loading cached acoustic and cross-modal features from cache...")
        ac_feats = np.load(ac_cache_path)
        cm_feats = np.load(cm_cache_path)
        # Verify shape
        if len(ac_feats) == len(df):
            return ac_feats, cm_feats
        logger.warning("Cache size mismatch. Re-extracting features...")
        
    # Extract features
    ac_feats, cm_feats = extract_features_parallel(df)
    
    # Save cache
    np.save(ac_cache_path, ac_feats)
    np.save(cm_cache_path, cm_feats)
    logger.info(f"Features saved to cache at {cache_dir}")
    
    return ac_feats, cm_feats

def run_baseline() -> Dict[str, Any]:
    logger.info("Starting Phase 2 Baseline Experiment...")
    
    # 1. Load data
    df = load_data(config, sync_azure=False)
    
    # 2. Extract or load features
    ac_feats, cm_feats = load_or_extract_features(df)
    
    # Concatenate features for tabular model: shape [N, 37 + 6 = 43]
    X_all = np.concatenate([ac_feats, cm_feats], axis=1)
    y_all = df["binary_label"].values
    
    # 3. K-fold CV setup
    splits = get_kfold_splits(df, config, group_col="folder")
    
    oof_probs = np.zeros(len(df))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"--- Training Tabular Fold {fold+1}/{config.training.n_folds} ---")
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        # Instantiate & Train
        clf = TabularLGBMClassifier(config)
        clf.fit(X_train, y_train, X_val, y_val)
        
        # OOF Predict
        oof_probs[val_idx] = clf.predict_proba(X_val)
        models.append(clf)
        
        # Save model checkpoint
        model_path = config.paths.model_dir / f"baseline_fold{fold+1}.pkl"
        clf.save(str(model_path))

    # 4. Global Evaluation
    # Threshold sweep on OOF
    best_threshold = 0.5
    best_f1 = -1.0
    thresholds = np.arange(0.05, 0.95, 0.01)
    
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        macro_f1 = compute_metrics(y_all, preds, oof_probs)["macro_f1"]
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = t
            
    oof_preds = (oof_probs >= best_threshold).astype(int)
    metrics = compute_metrics(y_all, oof_preds, oof_probs)
    
    logger.info(f"--- Phase 2 Baseline Results (OOF) ---")
    logger.info(f"Best Decision Threshold: {best_threshold:.2f}")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize():<12}: {v:.4f}")
        
    # Save confusion matrix plot
    plot_path = config.paths.output_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(y_all, oof_preds, str(plot_path), title="Baseline LightGBM Confusion Matrix")
    
    # Save OOF predictions to CSV for error analysis
    df_results = df.copy()
    df_results["oof_prob"] = oof_probs
    df_results["oof_pred"] = oof_preds
    df_results.to_csv(config.paths.output_dir / "baseline_oof_predictions.csv", index=False)
    
    # Run error analysis
    err_path = config.paths.output_dir / "baseline_errors.csv"
    run_error_analysis(df, y_all, oof_preds, oof_probs, str(err_path))
    
    results = {
        "model_name": "LightGBM",
        "mode": "handcrafted_tabular",
        **metrics,
        "best_threshold": best_threshold
    }
    return results

if __name__ == "__main__":
    run_baseline()
