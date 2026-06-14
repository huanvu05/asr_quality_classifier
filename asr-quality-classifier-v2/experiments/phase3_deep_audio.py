"""
phase3_deep_audio.py — Phase 3: Deep Audio Branch.

Trains ASRQualityClassifier in "audio_only" mode (WavLM + Handcrafted Acoustic Features).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import torch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config, logger
from src.data_loader import load_data, get_kfold_splits
from src.audio_features import get_acoustic_feature_keys, get_crossmodal_feature_keys
from src.classifier import ASRQualityClassifier
from src.trainer import ASRDataset, train_neural_model, evaluate_model, sweep_threshold
from src.evaluator import compute_metrics, plot_confusion_matrix

def run_deep_audio() -> Dict[str, Any]:
    logger.info("Starting Phase 3 Deep Audio Branch Experiment...")
    
    # 1. Load data
    df = load_data(config, sync_azure=False)
    
    # Load cached handcrafted features from cache directory
    cache_dir = config.paths.cache_dir
    ac_cache_path = cache_dir / "acoustic_features.npy"
    cm_cache_path = cache_dir / "crossmodal_features.npy"
    wavlm_cache_path = cache_dir / "wavlm_embeddings.npy"
    
    if not (ac_cache_path.exists() and cm_cache_path.exists()):
        raise FileNotFoundError("Cached handcrafted features not found. Please run 'python experiments/phase2_baseline.py' first to extract them.")
        
    if not wavlm_cache_path.exists():
        raise FileNotFoundError("Cached WavLM embeddings not found. Please run 'python experiments/extract_embeddings_multi_gpu.py' first.")
        
    ac_feats = np.load(ac_cache_path)
    cm_feats = np.load(cm_cache_path)
    wavlm_feats = np.load(wavlm_cache_path)
    
    # 2. CV Splits
    splits = get_kfold_splits(df, config, group_col="folder")
    
    oof_logits = np.zeros(len(df))
    y_all = df["binary_label"].values
    
    # We will accumulate test predictions across folds or run on validation set
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"--- Training Deep Audio Fold {fold+1}/{config.training.n_folds} ---")
        
        # Datasets
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = ASRDataset(
            df_train, 
            config, 
            ac_feats[train_idx], 
            cm_feats[train_idx],
            precomputed_audio_pools=wavlm_feats[train_idx]
        )
        val_ds = ASRDataset(
            df_val, 
            config, 
            ac_feats[val_idx], 
            cm_feats[val_idx],
            precomputed_audio_pools=wavlm_feats[val_idx]
        )
        
        # Instantiate model in audio_only mode
        model = ASRQualityClassifier(config, mode="audio_only").to(config.device)
        
        # Path to save fold model checkpoint
        checkpoint_path = config.paths.model_dir / f"deep_audio_fold{fold+1}.pt"
        
        # Train model
        model, metrics_info = train_neural_model(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            save_path=str(checkpoint_path)
        )
        
        # Evaluate model on validation set to collect logits
        # Re-load best checkpoint to evaluate
        checkpoint = torch.load(str(checkpoint_path), map_location=config.device, weights_only=False)
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
                
                audio_pools = batch.get("audio_pools")
                if audio_pools is not None:
                    audio_pools = audio_pools.to(config.device)
                    
                logits = model(wfs, txs, ac, cm, precomputed_audio_pool=audio_pools)
                val_logits.append(logits.cpu().numpy())
                
        oof_logits[val_idx] = np.concatenate(val_logits)
        
    # 3. Global OOF Evaluation
    best_threshold, _ = sweep_threshold(oof_logits, y_all, config)
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))
    oof_preds = (oof_probs >= best_threshold).astype(int)
    
    metrics = compute_metrics(y_all, oof_preds, oof_probs)
    
    logger.info(f"--- Phase 3 Deep Audio Results (OOF) ---")
    logger.info(f"Best Decision Threshold: {best_threshold:.2f}")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize():<12}: {v:.4f}")
        
    # Save confusion matrix plot
    plot_path = config.paths.output_dir / "deep_audio_confusion_matrix.png"
    plot_confusion_matrix(y_all, oof_preds, str(plot_path), title="Deep Audio (WavLM) Confusion Matrix")
    
    # Save OOF predictions to CSV for error analysis
    df_results = df.copy()
    df_results["oof_prob"] = oof_probs
    df_results["oof_pred"] = oof_preds
    df_results.to_csv(config.paths.output_dir / "deep_audio_oof_predictions.csv", index=False)
    
    results = {
        "model_name": "DeepAudio",
        "mode": "audio_only",
        **metrics,
        "best_threshold": best_threshold
    }
    return results

if __name__ == "__main__":
    # Ensure helper file exists
    # Create trainer_eval_helper.py to contain helper functions if needed, or define locally
    run_deep_audio()
