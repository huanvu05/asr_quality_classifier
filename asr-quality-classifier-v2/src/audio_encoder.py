"""
audio_encoder.py — WavLM frozen audio encoder.

Extracts frame-level sequence embeddings [Batch, Seq_Len, 768] and 
global pooled embeddings [Batch, 768] without updating weights.
"""

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel
from typing import List, Tuple, Dict, Any
import numpy as np

from src.config import Config, logger

class AudioEncoder(nn.Module):
    """
    Frozen WavLM audio encoder module.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.audio.audio_encoder_name
        self.device = config.device
        
        logger.info(f"Loading WavLM feature extractor: {self.model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name, 
            cache_dir=str(config.paths.cache_dir)
        )
        
        logger.info(f"Loading WavLM model: {self.model_name}")
        self.model = WavLMModel.from_pretrained(
            self.model_name, 
            cache_dir=str(config.paths.cache_dir)
        )
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        self.to(self.device)

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the downsampled sequence length after WavLM CNN encoder.
        WavLM downsampling factor is exactly 320.
        Formula: (L - kernel_size) // stride + 1 for each conv layer.
        """
        # Exact Wav2Vec2/WavLM feature encoder kernel/strides:
        # convs = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
        lengths = input_lengths.clone()
        convs = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
        for kernel, stride in convs:
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return torch.clamp(lengths, min=1)

    def forward(
        self, 
        waveforms: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts features for a list of raw waveforms.
        Returns:
            sequence_embeddings: [Batch, Seq_Len, 768] (padded)
            attention_mask: [Batch, Seq_Len] (1 for valid, 0 for padded)
            pooled_embeddings: [Batch, 768] (masked mean pool)
        """
        # Feature extractor preprocesses list of arrays
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.config.audio.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs["input_values"].to(self.device)
        attention_mask_in = inputs.get("attention_mask")
        
        # Get actual raw lengths for each item
        input_lengths = torch.tensor(
            [len(w) for w in waveforms], 
            dtype=torch.long, 
            device=self.device
        )
        output_lengths = self._get_output_lengths(input_lengths)
        
        # Limit max audio frames to avoid memory issues
        max_allowed_frames = self.config.audio.max_audio_frames
        output_lengths = torch.clamp(output_lengths, max=max_allowed_frames)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_values)
                # Raw sequence embeddings: [Batch, Raw_Seq_Len, 768]
                last_hidden_state = outputs.last_hidden_state
                
                # Truncate sequence length if it exceeds max_allowed_frames
                batch_size, raw_seq_len, hidden_dim = last_hidden_state.shape
                seq_len = min(raw_seq_len, max_allowed_frames)
                last_hidden_state = last_hidden_state[:, :seq_len, :]
                
                # Generate attention mask for the downsampled sequence
                attention_mask = torch.zeros(
                    (batch_size, seq_len), 
                    dtype=torch.float32, 
                    device=self.device
                )
                for i in range(batch_size):
                    valid_len = min(output_lengths[i].item(), seq_len)
                    attention_mask[i, :valid_len] = 1.0
                
                # Masked mean pooling for global embedding
                # Expand mask: [Batch, Seq_Len, 1]
                expanded_mask = attention_mask.unsqueeze(-1)
                masked_embeds = last_hidden_state * expanded_mask
                
                sum_embeds = masked_embeds.sum(dim=1)
                sum_mask = expanded_mask.sum(dim=1).clamp(min=1e-9)
                pooled_embeddings = sum_embeds / sum_mask
                
                return last_hidden_state, attention_mask, pooled_embeddings
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory during WavLM forward pass. Clearing cache.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
