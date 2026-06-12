"""
cross_attention.py — Cross-attention module for audio-text alignment.

Aligns frame-level audio features with token-level text features using
nn.MultiheadAttention. Q=audio, K=V=text.
"""

import torch
import torch.nn as nn
from src.config import Config

class CrossAttentionAlignment(nn.Module):
    """
    Cross-attention alignment head.
    Projects audio and text into a shared space, performs cross-attention,
    and pools the alignment to generate a unified representation.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.proj_dim = config.model.proj_dim
        
        # Audio projection (from WavLM 768 to proj_dim)
        self.audio_proj = nn.Linear(config.audio.audio_embed_dim, self.proj_dim)
        
        # Text projection (from PhoBERT 768 to proj_dim)
        self.text_proj = nn.Linear(config.text.text_embed_dim, self.proj_dim)
        
        # Multihead cross-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=config.model.n_heads,
            dropout=config.model.attn_dropout,
            batch_first=True
        )
        
        # Layer normalization & activation
        self.layer_norm = nn.LayerNorm(self.proj_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        audio_seq: torch.Tensor,
        audio_mask: torch.Tensor,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes alignment.
        Args:
            audio_seq: [Batch, Seq_Len_Audio, 768]
            audio_mask: [Batch, Seq_Len_Audio] (1.0 for valid, 0.0 for padding)
            text_seq: [Batch, Seq_Len_Text, 768]
            text_mask: [Batch, Seq_Len_Text] (1.0 for valid, 0.0 for padding)
        Returns:
            alignment_vector: [Batch, proj_dim] (mean pooled over audio sequence)
        """
        # 1. Project to shared dimension
        q = self.activation(self.audio_proj(audio_seq))  # [Batch, Seq_Len_Audio, proj_dim]
        k = self.activation(self.text_proj(text_seq))    # [Batch, Seq_Len_Text, proj_dim]
        v = k                                            # [Batch, Seq_Len_Text, proj_dim]
        
        # 2. Key padding mask (True for padding, False for valid)
        # PyTorch MHA expects True for positions to be masked out
        key_padding_mask = (text_mask == 0.0)            # [Batch, Seq_Len_Text]
        
        # 3. MHA cross-attention
        # attn_output: [Batch, Seq_Len_Audio, proj_dim]
        attn_output, _ = self.mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask
        )
        
        # 4. Residual connection & Norm
        attn_output = self.layer_norm(attn_output + q)
        
        # 5. Masked mean pooling over audio sequence
        # Expand audio mask: [Batch, Seq_Len_Audio, 1]
        expanded_audio_mask = audio_mask.unsqueeze(-1)
        masked_output = attn_output * expanded_audio_mask
        
        sum_output = masked_output.sum(dim=1)
        sum_mask = expanded_audio_mask.sum(dim=1).clamp(min=1e-9)
        
        alignment_vector = sum_output / sum_mask          # [Batch, proj_dim]
        return alignment_vector
