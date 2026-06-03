import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from src.config import config

class Evaluator:
    """
    Handles Cross-Validation, Threshold Optimization, and Performance Reporting.
    """
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    def optimize_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Sweeps thresholds to find the one that maximizes Macro F1-score.
        """
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            score = f1_score(y_true, y_pred, average='macro')
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        
        return float(best_threshold)

    def evaluate_oof(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float):
        """
        Generates comprehensive metrics and plots.
        """
        y_pred = (y_probs >= threshold).astype(int)
        
        print("\n" + "="*30)
        print("FINAL EVALUATION REPORT")
        print("="*30)
        print(f"Optimal Threshold: {threshold:.4f}")
        print(classification_report(y_true, y_pred))
        
        # ROC AUC
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC: {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Out-of-Fold)')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def run_cv(self, X: pd.DataFrame, y: pd.Series, model_factory):
        """
        Performs Stratified K-Fold Cross-Validation.
        """
        oof_probs = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = model_factory()
            model.fit(X_train, y_train)
            
            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            print(f"Fold {fold+1} completed.")
            
        return oof_probs
