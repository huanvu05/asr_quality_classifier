import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from src.config import config

class AudioDataset(Dataset):
    def __init__(self, embeddings: list, labels: list):
        # embeddings shape: [N, 1024]
        self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class AudioDNN(nn.Module):
    """
    DNN to classify 1024D vectors (512 Mean + 512 Max).
    Uses heavy dropout and weight decay as requested.
    """
    def __init__(self, input_dim: int = config.EMBEDDING_DIM):
        super(AudioDNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout as requested
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1) # Raw logit output
        )

    def forward(self, x):
        return self.network(x)
