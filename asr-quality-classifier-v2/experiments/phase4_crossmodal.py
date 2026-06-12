"""
phase4_crossmodal.py — Phase 4: Cross-Modal Fusion.

Trains ASRQualityClassifier in "full" mode (WavLM + PhoBERT + Cross-Attention + Handcrafted features).
"""

import os
import numpy as np
import pandas as pd
import torch

from src.config import config, logger
from src.data_loader import load_data, get_kfold_splits
from src.audio_features import get_acoustic_feature_keys, get_crossmodal_feature_keys
from src.classifier import ASRQualityClassifier
from src.trainer import ASRDataset, train_neural_model, evaluate_model, sweep_threshold
from src.evaluator import compute_metrics, plot_confusion_matrix

def run_crossmodal() -> Dict[str, Any]:
    logger.info("Starting Phase 4 Cross-Modal Fusion Experiment...")
    
    # 1. Load data
    df = load_data(config, sync_azure=False)
    
    # Load cached handcrafted features
    cache_dir = config.paths.cache_dir
    ac_cache_path = cache_dir / "acoustic_features.npy"
    cm_cache_path = cache_dir / "crossmodal_features.npy"
    
    if not (ac_cache_path.exists() and cm_cache_path.exists()):
        raise FileNotFoundError("Cached features not found. Please run phase2_baseline.py first to extract and cache features.")
        
    ac_feats = np.load(ac_cache_path)
    cm_feats = np.load(cm_cache_path)
    
    # 2. CV Splits
    splits = get_kfold_splits(df, config, group_col="folder")
    
    oof_logits = np.zeros(len(df))
    y_all = df["binary_label"].values
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"--- Training Cross-Modal Fold {fold+1}/{config.training.n_folds} ---")
        
        # Datasets
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = ASRDataset(
            df_train, 
            config, 
            ac_feats[train_idx], 
            cm_feats[train_idx]
        )
        val_ds = ASRDataset(
            df_val, 
            config, 
            ac_feats[val_idx], 
            cm_feats[val_idx]
        )
        
        # Instantiate model in full mode
        model = ASRQualityClassifier(config, mode="full")
        
        # Path to save fold model checkpoint
        checkpoint_path = config.paths.model_dir / f"crossmodal_fold{fold+1}.pt"
        
        # Train model
        model, metrics_info = train_neural_model(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            save_path=str(checkpoint_path)
        )
        
        # Evaluate model on validation set to collect logits
        checkpoint = torch.load(str(checkpoint_path), map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        from torch.utils.data import DataLoader
        from src.trainer import collate_fn
        
        # Load val loader
        val_loader = DataLoader(
            val_ds, 
            batch_size=config.training.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Get val logits
        model.eval()
        val_logits = []
        with torch.no_grad():
            for batch in val_loader:
                wfs = batch["waveforms"]
                txs = batch["texts"]
                ac = batch["acoustic_feats"].to(config.device)
                cm = batch["crossmodal_feats"].to(config.device)
                logits = model(wfs, txs, ac, cm)
                val_logits.append(logits.cpu().numpy())
                
        oof_logits[val_idx] = np.concatenate(val_logits)
        
    # 3. Global OOF Evaluation
    best_threshold, _ = sweep_threshold(oof_logits, y_all, config)
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))
    oof_preds = (oof_probs >= best_threshold).astype(int)
    
    metrics = compute_metrics(y_all, oof_preds, oof_probs)
    
    logger.info(f"--- Phase 4 Cross-Modal Results (OOF) ---")
    logger.info(f"Best Decision Threshold: {best_threshold:.2f}")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize():<12}: {v:.4f}")
        
    # Save confusion matrix plot
    plot_path = config.paths.output_dir / "crossmodal_confusion_matrix.png"
    plot_confusion_matrix(y_all, oof_preds, str(plot_path), title="Cross-Modal Fusion Confusion Matrix")
    
    # Save OOF predictions to CSV for error analysis
    df_results = df.copy()
    df_results["oof_prob"] = oof_probs
    df_results["oof_pred"] = oof_preds
    df_results.to_csv(config.paths.output_dir / "crossmodal_oof_predictions.csv", index=False)
    
    results = {
        "model_name": "CrossModalFusion",
        "mode": "full",
        **metrics,
        "best_threshold": best_threshold
    }
    return results

if __name__ == "__main__":
    run_crossmodal()
