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
        
        # Reduced complexity to prevent severe overfitting.
        # Increased Dropout to 0.5
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        return self.network(x)
