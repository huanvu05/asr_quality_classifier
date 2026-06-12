"""
test_text_encoder.py — Tests for the frozen text encoder branch.
"""

import pytest
import pandas as pd
import torch
from src.text_encoder import TextEncoder

def test_text_encoder_forward(test_config):
    encoder = TextEncoder(test_config)
    assert encoder.device == "cpu"
    
    texts = ["xin chào", "tôi là robot học máy", None]
    
    seq_embeds, attn_mask, pooled_embeds = encoder(texts)
    
    # Check shapes (mock returns last_hidden_state of shape [3, 20, 768])
    # Note conftest fixture mock is batch size 2, but when mocked model is called, 
    # mock returns shapes matching what we setup.
    # Let's verify our mock outputs shape:
    assert seq_embeds.shape[2] == 768
    assert attn_mask.shape[0] == len(texts)
    assert pooled_embeds.shape == (len(texts), 768)
