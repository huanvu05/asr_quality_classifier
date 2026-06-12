"""
test_audio_encoder.py — Tests for the frozen audio encoder branch.
"""

import pytest
import numpy as np
import torch
from src.audio_encoder import AudioEncoder

def test_audio_encoder_forward(test_config):
    # Initialize encoder (uses patch mock from conftest)
    encoder = AudioEncoder(test_config)
    assert encoder.device == "cpu"
    
    # Create mock batch of 2 raw waveforms
    w1 = np.random.normal(0.0, 0.1, 16000)
    w2 = np.random.normal(0.0, 0.1, 32000)
    
    seq_embeds, attn_mask, pooled_embeds = encoder([w1, w2])
    
    # Check shapes (mock returns last_hidden_state of shape [2, 50, 768])
    assert seq_embeds.shape == (2, 50, 768)
    assert attn_mask.shape == (2, 50)
    assert pooled_embeds.shape == (2, 768)
    
    # Check that mask contains 1s and 0s
    assert (attn_mask >= 0.0).all() and (attn_mask <= 1.0).all()
    # Mask of the second sample (longer) should have more or equal 1s than the first
    assert attn_mask[1].sum() >= attn_mask[0].sum()
