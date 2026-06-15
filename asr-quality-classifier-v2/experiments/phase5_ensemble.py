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
    
    # Define potential OOF files
    oof_files = {
        "Baseline": out_dir / "baseline_oof_predictions.csv",
        "DeepAudio": out_dir / "deep_audio_oof_predictions.csv",
        "CrossModal": out_dir / "crossmodal_oof_predictions.csv"
    }
    
    available_preds = {}
    df_base = None
    y_true = None
    
    for name, path in oof_files.items():
        if path.exists():
            df = pd.read_csv(path)
            available_preds[name] = df["oof_prob"].values
            if df_base is None:
                df_base = df
                y_true = df["binary_label"].values
            else:
                assert (df_base["file_path"] == df["file_path"]).all()
        else:
            logger.warning(f"OOF predictions for {name} not found. It will be excluded from the ensemble.")
            
    if not available_preds:
        logger.error("No OOF prediction files found. Cannot run ensemble.")
        return
        
    if len(available_preds) == 1:
        logger.error("Only one OOF prediction file found. Ensemble requires at least two models.")
        return
    
    # 1. Grid search for optimal blending weights dynamically
    best_f1 = -1.0
    best_weights = {}
    best_threshold = 0.5
    
    # We will just do a simple search depending on available models
    models = list(available_preds.keys())
    
    logger.info(f"Searching for optimal blending weights for: {models}...")
    
    if len(models) == 2:
        for w1 in np.linspace(0.0, 1.0, 21):
            w2 = 1.0 - w1
            p_blend = w1 * available_preds[models[0]] + w2 * available_preds[models[1]]
            
            for t in np.arange(0.1, 0.9, 0.02):
                preds = (p_blend >= t).astype(int)
                from sklearn.metrics import f1_score
                score = f1_score(y_true, preds, average="macro")
                if score > best_f1:
                    best_f1 = score
                    best_weights = {models[0]: w1, models[1]: w2}
                    best_threshold = t
    elif len(models) == 3:
        for w1 in np.linspace(0.0, 1.0, 11):
            for w2 in np.linspace(0.0, 1.0 - w1, 11):
                w3 = 1.0 - w1 - w2
                if w3 < 0.0 or np.isclose(w3, 0.0): w3 = 0.0
                
                p_blend = w1 * available_preds[models[0]] + w2 * available_preds[models[1]] + w3 * available_preds[models[2]]
                
                for t in np.arange(0.1, 0.9, 0.02):
                    preds = (p_blend >= t).astype(int)
                    from sklearn.metrics import f1_score
                    score = f1_score(y_true, preds, average="macro")
                    if score > best_f1:
                        best_f1 = score
                        best_weights = {models[0]: w1, models[1]: w2, models[2]: w3}
                        best_threshold = t
                        
    logger.info(f"Optimal weights found: {best_weights} | Best Threshold: {best_threshold:.2f} | Best Val F1: {best_f1:.4f}")
    
    # Compute final ensemble predictions
    p_ensemble = np.zeros_like(y_true, dtype=float)
    for name, w in best_weights.items():
        p_ensemble += w * available_preds[name]
        
    preds_ensemble = (p_ensemble >= best_threshold).astype(int)
    
    # Generate Results Table
    results = []
    for name, p_val in available_preds.items():
        metrics = compute_metrics(y_true, (p_val >= 0.5).astype(int), p_val)
        results.append({"model_name": name, "mode": "individual", **metrics})
        
    metrics_ensemble = compute_metrics(y_true, preds_ensemble, p_ensemble)
    results.append({"model_name": "Weighted Ensemble Blend", "mode": "ensemble_blend", **metrics_ensemble})
    
    # Plot ensemble confusion matrix
    weight_str = "_".join([f"{k}={v:.1f}" for k, v in best_weights.items()])
    plot_confusion_matrix(
        y_true, 
        preds_ensemble, 
        str(out_dir / "ensemble_confusion_matrix.png"), 
        title=f"Ensemble Confusion Matrix ({weight_str})"
    )
    
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
