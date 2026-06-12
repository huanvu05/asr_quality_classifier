"""
test_trainer.py — Tests for trainer utilities and training loop.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.trainer import (
    ASRDataset,
    collate_fn,
    sweep_threshold,
    train_neural_model
)
from src.classifier import ASRQualityClassifier

def test_asr_dataset_and_collate(dummy_df, test_config):
    df, _, _ = dummy_df
    df["binary_label"] = (df["label"] == 1).astype(int)
    # Ensure absolute path is set
    df["absolute_path"] = df["file_path"].apply(lambda fp: str(test_config.paths.audio_dir / fp))
    
    ac_feats = np.random.randn(len(df), 37)
    cm_feats = np.random.randn(len(df), 6)
    
    ds = ASRDataset(df, test_config, ac_feats, cm_feats)
    assert len(ds) == len(df)
    
    item = ds[0]
    assert "waveform" in item
    assert "text" in item
    assert "acoustic_feat" in item
    assert "crossmodal_feat" in item
    assert "label" in item
    
    # Test collate_fn
    batch = [ds[i] for i in range(2)]
    collated = collate_fn(batch)
    assert len(collated["waveforms"]) == 2
    assert len(collated["texts"]) == 2
    assert collated["acoustic_feats"].shape == (2, 37)
    assert collated["crossmodal_feats"].shape == (2, 6)
    assert collated["labels"].shape == (2,)

def test_sweep_threshold(test_config):
    # Perfect alignment
    logits = np.array([-10.0, -5.0, 5.0, 10.0])
    labels = np.array([0, 0, 1, 1])
    
    best_t, best_f1 = sweep_threshold(logits, labels, test_config)
    assert best_t > 0.0 and best_t < 1.0
    assert best_f1 == 1.0

def test_train_neural_model(dummy_df, test_config, tmp_path):
    df, _, _ = dummy_df
    df["binary_label"] = (df["label"] == 1).astype(int)
    df["absolute_path"] = df["file_path"].apply(lambda fp: str(test_config.paths.audio_dir / fp))
    
    ac_feats = np.random.randn(len(df), 37)
    cm_feats = np.random.randn(len(df), 6)
    
    ds = ASRDataset(df, test_config, ac_feats, cm_feats)
    
    # Run in audio_only mode for quick test
    model = ASRQualityClassifier(test_config, mode="audio_only")
    
    test_config.training.epochs = 1
    test_config.training.batch_size = 2
    
    save_path = tmp_path / "model.pt"
    trained_model, info = train_neural_model(
        model=model,
        train_dataset=ds,
        val_dataset=ds,
        config=test_config,
        save_path=str(save_path)
    )
    
    assert save_path.exists()
    assert info["best_val_f1"] >= 0.0
    assert "history" in info
