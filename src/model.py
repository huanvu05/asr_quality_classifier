import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import config

class AudioDataset(Dataset):
    def __init__(self, embeddings: list, labels: list):
        self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class AudioMLP(nn.Module):
    """
    Multi-Layer Perceptron to classify Deep Audio Embeddings.
    """
    def __init__(self, input_dim: int = config.EMBEDDING_DIM):
        super(AudioMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1) # Output raw logit
        )

    def forward(self, x):
        return self.network(x)
