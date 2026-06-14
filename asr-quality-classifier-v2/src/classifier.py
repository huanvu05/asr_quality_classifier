"""
classifier.py — Main neural model and LightGBM classifier wrapper.

Supports ablation modes: "full", "audio_only", "text_only", "crossmodal_only".
"""

import os
import joblib
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from src.config import Config, logger
from src.audio_encoder import AudioEncoder
from src.text_encoder import TextEncoder
from src.cross_attention import CrossAttentionAlignment

class ASRQualityClassifier(nn.Module):
    """
    Unified deep cross-modal classifier.
    Combines WavLM, PhoBERT, Cross-Attention alignment, and Handcrafted features.
    """

    def __init__(self, config: Config, mode: str = "full"):
        super().__init__()
        self.config = config
        self.mode = mode
        
        # Initialize frozen encoders
        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)
        
        # Initialize cross-attention head
        self.alignment_head = CrossAttentionAlignment(config)
        
        # Determine MLP input dimension based on ablation mode
        self.input_dim = self._get_input_dim()
        logger.info(f"Classifier initialized in '{mode}' mode. MLP Input Dim: {self.input_dim}")
        
        # Construct MLP classifier
        layers = []
        curr_dim = self.input_dim
        for h_dim in config.model.mlp_hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.model.mlp_dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))  # Output raw logit
        
        self.mlp = nn.Sequential(*layers)

    def _get_input_dim(self) -> int:
        # full: alignment (256) + audio_pool (768) + text_pool (768) + acoustic (37) + crossmodal (6) = 1835
        # audio_only: audio_pool (768) + acoustic (37) = 805
        # text_only: text_pool (768) = 768
        # crossmodal_only: alignment (256) + crossmodal (6) = 262
        if self.mode == "full":
            return (
                self.config.model.proj_dim + 
                self.config.audio.audio_embed_dim + 
                self.config.text.text_embed_dim + 
                self.config.model.n_acoustic_features + 
                self.config.model.n_crossmodal_features
            )
        elif self.mode == "audio_only":
            return (
                self.config.audio.audio_embed_dim + 
                self.config.model.n_acoustic_features
            )
        elif self.mode == "text_only":
            return self.config.text.text_embed_dim
        elif self.mode == "crossmodal_only":
            return (
                self.config.model.proj_dim + 
                self.config.model.n_crossmodal_features
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(
        self,
        waveforms: List[np.ndarray],
        texts: List[str],
        acoustic_feats: torch.Tensor,
        crossmodal_feats: torch.Tensor,
        precomputed_audio_pool: Optional[torch.Tensor] = None,
        precomputed_text_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            waveforms: List of raw audio NumPy arrays.
            texts: List of transcript strings.
            acoustic_feats: Handcrafted features [Batch, 37]
            crossmodal_feats: Length features [Batch, 6]
            precomputed_audio_pool: Offline WavLM embeddings [Batch, 768]. If provided, skips audio_encoder.
            precomputed_text_pool: Offline PhoBERT embeddings [Batch, 768]. If provided, skips text_encoder.
        Returns:
            logits: [Batch] (raw logits before sigmoid)
        """
        device = self.config.device
        ac_tensor = acoustic_feats.to(device)
        cm_tensor = crossmodal_feats.to(device)
        
        # 1. Obtain Audio Embeddings
        if precomputed_audio_pool is not None:
            audio_pool = precomputed_audio_pool.to(device)
            audio_seq, audio_mask = None, None # Alignment not supported with precomputed pools yet
        else:
            audio_seq, audio_mask, audio_pool = self.audio_encoder(waveforms)
            
        # 2. Obtain Text Embeddings
        if precomputed_text_pool is not None:
            text_pool = precomputed_text_pool.to(device)
            text_seq, text_mask = None, None
        else:
            text_seq, text_mask, text_pool = self.text_encoder(texts)
        
        # 3. Extract alignment vector if needed
        if self.mode in ["full", "crossmodal_only"]:
            if audio_seq is None or text_seq is None:
                raise ValueError("Cross-attention alignment requires raw waveforms and texts, not just precomputed pools.")
            alignment = self.alignment_head(audio_seq, audio_mask, text_seq, text_mask)
            
        # 4. Concatenate features based on mode
        features_to_concat = []
        
        if self.mode == "full":
            features_to_concat = [alignment, audio_pool, text_pool, ac_tensor, cm_tensor]
        elif self.mode == "audio_only":
            features_to_concat = [audio_pool, ac_tensor]
        elif self.mode == "text_only":
            features_to_concat = [text_pool]
        elif self.mode == "crossmodal_only":
            features_to_concat = [alignment, cm_tensor]
            
        x = torch.cat(features_to_concat, dim=1)
        
        # 5. Predict
        logits = self.mlp(x).squeeze(-1)
        return logits


class TabularLGBMClassifier:
    """
    LightGBM tabular classifier wrapper.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Trains the LightGBM model with early stopping."""
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": self.config.training.lgbm_learning_rate,
            "num_leaves": self.config.training.lgbm_num_leaves,
            "feature_fraction": self.config.training.lgbm_feature_fraction,
            "bagging_fraction": self.config.training.lgbm_bagging_fraction,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": self.config.training.seed,
            "scale_pos_weight": 1.0 / self.config.training.pos_class_weight,
            "n_jobs": -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        logger.info("Training LightGBM classifier...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.training.lgbm_n_estimators,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.training.lgbm_early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        logger.info(f"LGBM training complete. Best iteration: {self.model.best_iteration}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (usable = class 1)."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path: str):
        """Saves the LightGBM model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.parent(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"LGBM model saved to {path}")

    def load(self, path: str):
        """Loads a saved LightGBM model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LGBM model not found at {path}")
        self.model = joblib.load(path)
        logger.info(f"LGBM model loaded from {path}")
