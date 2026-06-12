"""
text_encoder.py — PhoBERT frozen text encoder.

Extracts token-level sequence embeddings [Batch, Seq_Len_Text, 768] and
global pooled embeddings [Batch, 768] without updating weights.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

from src.config import Config, logger

class TextEncoder(nn.Module):
    """
    Frozen PhoBERT text encoder module.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.text.text_encoder_name
        self.device = config.device
        
        logger.info(f"Loading PhoBERT tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=str(config.paths.cache_dir)
        )
        
        logger.info(f"Loading PhoBERT model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            cache_dir=str(config.paths.cache_dir)
        )
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        self.to(self.device)

    def forward(
        self, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes and extracts embeddings for a list of Vietnamese texts.
        Returns:
            sequence_embeddings: [Batch, Seq_Len_Text, 768] (padded)
            attention_mask: [Batch, Seq_Len_Text] (1 for valid, 0 for padded)
            pooled_embeddings: [Batch, 768] (masked mean pool)
        """
        # Ensure input is list of clean strings
        cleaned_texts = [str(t) if not pd_isna(t) else "" for t in texts]
        
        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.config.text.max_token_length,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device).float()
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask.long())
            # [Batch, Seq_Len_Text, 768]
            last_hidden_state = outputs.last_hidden_state
            
            # Masked mean pooling for global embedding
            expanded_mask = attention_mask.unsqueeze(-1)
            masked_embeds = last_hidden_state * expanded_mask
            
            sum_embeds = masked_embeds.sum(dim=1)
            sum_mask = expanded_mask.sum(dim=1).clamp(min=1e-9)
            pooled_embeddings = sum_embeds / sum_mask
            
            return last_hidden_state, attention_mask, pooled_embeddings

# Helper helper to mock pandasisna check inside class
def pd_isna(val) -> bool:
    try:
        import pandas as pd
        return pd.isna(val)
    except ImportError:
        return val is None or val != val
