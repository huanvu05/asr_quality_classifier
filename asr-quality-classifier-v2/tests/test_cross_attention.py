"""
test_cross_attention.py — Tests for the cross-attention alignment layer.
"""

import pytest
import torch
from src.cross_attention import CrossAttentionAlignment

def test_cross_attention_forward(test_config):
    align_head = CrossAttentionAlignment(test_config)
    
    batch_size = 4
    seq_len_audio = 50
    seq_len_text = 20
    
    # Dummy inputs
    audio_seq = torch.randn(batch_size, seq_len_audio, test_config.audio.audio_embed_dim)
    audio_mask = torch.ones(batch_size, seq_len_audio)
    # Mask out some audio padding
    audio_mask[0, 45:] = 0.0
    
    text_seq = torch.randn(batch_size, seq_len_text, test_config.text.text_embed_dim)
    text_mask = torch.ones(batch_size, seq_len_text)
    # Mask out some text padding
    text_mask[1, 15:] = 0.0
    
    alignment = align_head(audio_seq, audio_mask, text_seq, text_mask)
    
    # Result should be [Batch, proj_dim] -> [4, 256]
    assert alignment.shape == (batch_size, test_config.model.proj_dim)
    assert not torch.isnan(alignment).any()
