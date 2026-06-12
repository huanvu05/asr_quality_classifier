"""
test_evaluator.py — Tests for metrics and evaluation utilities.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluator import (
    compute_metrics,
    plot_confusion_matrix,
    generate_ablation_table,
    run_error_analysis
)

def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 50% acc, precision 0.5, recall 0.5
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    assert np.isclose(metrics["accuracy"], 0.5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 0.5)
    assert np.isclose(metrics["f1_score"], 0.5)
    assert np.isclose(metrics["macro_f1"], 0.5)
    assert metrics["auc"] > 0.5

def test_plot_confusion_matrix(tmp_path):
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    save_path = tmp_path / "cm.png"
    
    plot_confusion_matrix(y_true, y_pred, str(save_path))
    assert save_path.exists()

def test_generate_ablation_table(tmp_path):
    results = [
        {"model_name": "M1", "mode": "full", "accuracy": 0.8, "macro_f1": 0.75, "auc": 0.82},
        {"model_name": "M2", "mode": "audio_only", "accuracy": 0.7, "macro_f1": 0.65, "auc": 0.72}
    ]
    save_path = tmp_path / "ablation.csv"
    
    df = generate_ablation_table(results, str(save_path))
    assert save_path.exists()
    assert len(df) == 2
    assert "model_name" in df.columns

def test_run_error_analysis(tmp_path):
    data = {
        "file_path": ["a.wav", "b.wav"],
        "transcript": ["t1", "t2"],
        "binary_label": [1, 0]
    }
    df = pd.DataFrame(data)
    y_true = np.array([1, 0])
    y_pred = np.array([0, 1])  # both wrong
    y_prob = np.array([0.1, 0.9])
    
    save_path = tmp_path / "errors.csv"
    df_err = run_error_analysis(df, y_true, y_pred, y_prob, str(save_path))
    
    assert save_path.exists()
    assert len(df_err) == 2
    assert "error_type" in df_err.columns
    # Check error types
    assert df_err.loc[df_err["file_path"] == "a.wav", "error_type"].values[0] == "False Negative"
    assert df_err.loc[df_err["file_path"] == "b.wav", "error_type"].values[0] == "False Positive"
