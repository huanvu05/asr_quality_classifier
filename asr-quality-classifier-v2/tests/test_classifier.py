"""
test_classifier.py — Tests for deep and tabular classifiers.
"""

import pytest
import numpy as np
import torch
from src.classifier import ASRQualityClassifier, TabularLGBMClassifier

def test_deep_classifier_modes(test_config):
    # Test all ablation modes
    modes = ["full", "audio_only", "text_only", "crossmodal_only"]
    batch_size = 2
    
    # Dummy inputs
    w1 = np.random.normal(0.0, 0.1, 16000)
    w2 = np.random.normal(0.0, 0.1, 16000)
    waveforms = [w1, w2]
    
    texts = ["xin chào", "xin chào Việt Nam"]
    
    ac_feats = torch.randn(batch_size, test_config.model.n_acoustic_features)
    cm_feats = torch.randn(batch_size, test_config.model.n_crossmodal_features)
    
    for mode in modes:
        model = ASRQualityClassifier(test_config, mode=mode)
        logits = model(waveforms, texts, ac_feats, cm_feats)
        
        assert logits.shape == (batch_size,)
        assert not torch.isnan(logits).any()

def test_tabular_lgbm_classifier(test_config, tmp_path):
    # Synthesize tabular data
    X_train = np.random.randn(20, 43)
    y_train = np.random.randint(0, 2, 20)
    X_val = np.random.randn(10, 43)
    y_val = np.random.randint(0, 2, 10)
    
    clf = TabularLGBMClassifier(test_config)
    clf.fit(X_train, y_train, X_val, y_val)
    
    # Test predict
    probs = clf.predict_proba(X_val)
    assert probs.shape == (10,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    preds = clf.predict(X_val, threshold=0.5)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})
    
    # Test serialization
    model_file = tmp_path / "lgb.pkl"
    clf.save(str(model_file))
    assert model_file.exists()
    
    new_clf = TabularLGBMClassifier(test_config)
    new_clf.load(str(model_file))
    new_probs = new_clf.predict_proba(X_val)
    assert np.allclose(probs, new_probs)
