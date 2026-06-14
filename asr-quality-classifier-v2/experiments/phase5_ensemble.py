"""
phase5_ensemble.py — Phase 5: Model Ensemble and Final Ablation Analysis.

Loads OOF predictions from baseline, deep audio, and cross-modal fusion,
finds the optimal blend weights, and saves the final comparison table.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.config import config, logger
from src.evaluator import compute_metrics, plot_confusion_matrix, generate_ablation_table

def run_ensemble():
    logger.info("Starting Phase 5 Ensemble and Ablation Analysis...")
    
    out_dir = config.paths.output_dir
    
    # Check if OOF files exist
    baseline_path = out_dir / "baseline_oof_predictions.csv"
    audio_path = out_dir / "deep_audio_oof_predictions.csv"
    crossmodal_path = out_dir / "crossmodal_oof_predictions.csv"
    
    # If any file is missing, we will simulate running the phases sequentially
    # or raise an error. For robustness, let's log error if missing.
    missing = []
    for p in [baseline_path, audio_path, crossmodal_path]:
        if not p.exists():
            missing.append(p.name)
            
    if missing:
        logger.error(
            f"Cannot run ensemble. The following OOF prediction files are missing: {missing}. "
            "Please run phase2_baseline.py, phase3_deep_audio.py, and phase4_crossmodal.py first."
        )
        return
        
    # Load predictions
    df_base = pd.read_csv(baseline_path)
    df_audio = pd.read_csv(audio_path)
    df_cm = pd.read_csv(crossmodal_path)
    
    # Verify alignment
    assert (df_base["file_path"] == df_audio["file_path"]).all()
    assert (df_base["file_path"] == df_cm["file_path"]).all()
    
    y_true = df_base["binary_label"].values
    
    p_base = df_base["oof_prob"].values
    p_audio = df_audio["oof_prob"].values
    p_cm = df_cm["oof_prob"].values
    
    # 1. Grid search for optimal blending weights
    best_f1 = -1.0
    best_weights = (0.33, 0.33, 0.34)
    best_threshold = 0.5
    
    logger.info("Searching for optimal blending weights (grid search)...")
    for w_base in np.linspace(0.0, 1.0, 11):
        for w_audio in np.linspace(0.0, 1.0 - w_base, 11):
            w_cm = 1.0 - w_base - w_audio
            if w_cm < 0.0 or np.isclose(w_cm, 0.0):
                w_cm = 0.0
                
            # Weighted probability blend
            p_blend = w_base * p_base + w_audio * p_audio + w_cm * p_cm
            
            # Sweep decision threshold
            for t in np.arange(0.1, 0.9, 0.02):
                preds = (p_blend >= t).astype(int)
                # Compute macro F1
                from sklearn.metrics import f1_score
                score = f1_score(y_true, preds, average="macro")
                if score > best_f1:
                    best_f1 = score
                    best_weights = (w_base, w_audio, w_cm)
                    best_threshold = t
                    
    w_base, w_audio, w_cm = best_weights
    logger.info(
        f"Optimal weights found: Baseline={w_base:.2f}, DeepAudio={w_audio:.2f}, CrossModal={w_cm:.2f} | "
        f"Best Threshold: {best_threshold:.2f} | Best Val F1: {best_f1:.4f}"
    )
    
    # Compute final ensemble predictions
    p_ensemble = w_base * p_base + w_audio * p_audio + w_cm * p_cm
    preds_ensemble = (p_ensemble >= best_threshold).astype(int)
    
    # Compute metrics
    metrics_base = compute_metrics(y_true, (p_base >= 0.5).astype(int), p_base)
    metrics_audio = compute_metrics(y_true, (p_audio >= 0.5).astype(int), p_audio)
    metrics_cm = compute_metrics(y_true, (p_cm >= 0.5).astype(int), p_cm)
    metrics_ensemble = compute_metrics(y_true, preds_ensemble, p_ensemble)
    
    # Plot ensemble confusion matrix
    plot_confusion_matrix(
        y_true, 
        preds_ensemble, 
        str(out_dir / "ensemble_confusion_matrix.png"), 
        title=f"Ensemble Confusion Matrix (w_base={w_base:.1f}, w_aud={w_audio:.1f}, w_cm={w_cm:.1f})"
    )
    
    # 2. Generate and save final ablation study comparison table
    results = [
        {
            "model_name": "LightGBM Baseline",
            "mode": "handcrafted_tabular",
            **metrics_base
        },
        {
            "model_name": "Deep Audio Branch",
            "mode": "audio_only",
            **metrics_audio
        },
        {
            "model_name": "Cross-Modal Fusion",
            "mode": "full",
            **metrics_cm
        },
        {
            "model_name": "Weighted Ensemble Blend",
            "mode": "ensemble_blend",
            **metrics_ensemble
        }
    ]
    
    ablation_csv_path = out_dir / "ablation_comparison_results.csv"
    generate_ablation_table(results, str(ablation_csv_path))
    
    # Save ensemble predictions to CSV
    df_ensemble = df_base.copy()
    df_ensemble["ensemble_prob"] = p_ensemble
    df_ensemble["ensemble_pred"] = preds_ensemble
    df_ensemble.to_csv(out_dir / "ensemble_predictions.csv", index=False)
    
    logger.info("Phase 5 Ensemble and Ablation Analysis complete.")

if __name__ == "__main__":
    run_ensemble()
