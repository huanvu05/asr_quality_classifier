"""
evaluator.py — Metric computation, confusion matrix plotting, ablation comparisons, and error analysis.
"""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config, logger

def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Computes binary classification metrics: Precision, Recall, F1, Accuracy, AUC.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average="binary", 
        zero_division=0
    )
    macro_f1 = f1_score_macro(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "macro_f1": float(macro_f1),
        "auc": float(auc)
    }

def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper to compute macro F1 score."""
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average="macro", 
        zero_division=0
    )
    return float(f_macro)

def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: str,
    title: str = "Confusion Matrix"
):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # Class labels
    labels = ["Unusable (0)", "Usable (1)"]
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14}
    )
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {save_path}")

def generate_ablation_table(
    results: List[Dict[str, Any]], 
    save_path: str
) -> pd.DataFrame:
    """
    Converts ablation results list to a pandas DataFrame, prints it, and saves as CSV.
    """
    df_results = pd.DataFrame(results)
    
    # Order columns logically
    cols = ["model_name", "mode", "accuracy", "precision", "recall", "f1_score", "macro_f1", "auc"]
    cols = [c for c in cols if c in df_results.columns]
    df_results = df_results[cols]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_results.to_csv(save_path, index=False)
    
    print("\n" + "=" * 80)
    print("                      ABLATION STUDY COMPARISON TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80 + "\n")
    
    logger.info(f"Ablation table saved to {save_path}")
    return df_results

def run_error_analysis(
    df: pd.DataFrame, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    save_path: str
) -> pd.DataFrame:
    """
    Finds false positives and false negatives, saves them to a CSV report.
    """
    df_err = df.copy()
    df_err["y_true"] = y_true
    df_err["y_pred"] = y_pred
    df_err["y_prob"] = y_prob
    
    # Filter for errors
    df_errors = df_err[df_err["y_true"] != df_err["y_pred"]].copy()
    
    df_errors["error_type"] = np.where(
        (df_errors["y_true"] == 0) & (df_errors["y_pred"] == 1),
        "False Positive",
        "False Negative"
    )
    
    # Sort errors by confidence/margin
    # For FP: high probability is worse
    # For FN: low probability is worse
    df_errors["error_margin"] = np.where(
        df_errors["error_type"] == "False Positive",
        df_errors["y_prob"],
        1.0 - df_errors["y_prob"]
    )
    df_errors = df_errors.sort_values(by="error_margin", ascending=False)
    
    # Save output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_errors.to_csv(save_path, index=False)
    
    logger.info(f"Error analysis completed. Found {len(df_errors)} errors out of {len(df)} samples.")
    logger.info(f"Top 5 most confident errors saved to {save_path}")
    
    return df_errors
