import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from src.config import config

class SequenceAudioDataset(Dataset):
    def __init__(self, embeddings: list, labels: list):
        # embeddings shape: [N, MAX_SEQ_LENGTH, EMBEDDING_DIM]
        self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class AttentionHeadClassifier(nn.Module):
    """
    SOTA Attention-based Classification Head.
    Instead of mean-pooling, it learns to dynamically weight the most important acoustic frames.
    """
    def __init__(self, input_dim: int = config.EMBEDDING_DIM, hidden_dim: int = 256):
        super(AttentionHeadClassifier, self).__init__()
        
        # Bi-directional LSTM to capture temporal context of errors (e.g., glitch preceding silence)
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention_fc = nn.Linear(hidden_dim * 2, 1) # *2 because of bidirectional
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Raw logit output
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # 1. Temporal Context
        lstm_out, _ = self.bilstm(x) # [batch_size, seq_len, hidden_dim * 2]
        
        # 2. Attention Weights
        # Calculate attention score for each frame
        attn_scores = self.attention_fc(lstm_out) # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # Normalize scores across the sequence length
        
        # 3. Context Vector (Weighted sum of frames based on attention)
        # We multiply the LSTM output by the attention weights and sum over the sequence
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # [batch_size, hidden_dim * 2]
        
        # 4. Classification
        logits = self.classifier(context_vector) # [batch_size, 1]
        return logits
