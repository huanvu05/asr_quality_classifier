import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from src.config import config

class SequenceAudioDataset(Dataset):
    def __init__(self, embeddings: list, labels: list):
        # embeddings shape: [N, MAX_SEQ_LENGTH, 768]
        self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class AttentionHeadClassifier(nn.Module):
    """
    SOTA Attention-based Classification Head.
    Tự động quét qua chuỗi 400 frames và dồn trọng số vào frame chứa tiếng vấp/ồn.
    """
    def __init__(self, input_dim: int = config.EMBEDDING_DIM, hidden_dim: int = 128):
        super(AttentionHeadClassifier, self).__init__()
        
        # 1. Bi-directional LSTM để hiểu ngữ cảnh trước/sau của âm thanh
        self.bilstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        
        # 2. Attention Layer
        # Input là 2 * hidden_dim do Bi-LSTM
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 3. Mạng phân loại cuối cùng (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout mạnh
            nn.Linear(64, 1) # Output Logit
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len=400, input_dim=768]
        
        # 1. Quét qua LSTM
        lstm_out, _ = self.bilstm(x) 
        # lstm_out shape: [batch_size, 400, 256]
        
        # 2. Tính điểm Attention cho từng frame
        attn_scores = self.attention_fc(lstm_out) 
        # attn_scores shape: [batch_size, 400, 1]
        
        # Softmax để ra trọng số (0 -> 1)
        attn_weights = F.softmax(attn_scores, dim=1) 
        
        # 3. Nhân trọng số Attention vào LSTM output (Nhấn mạnh frame quan trọng)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) 
        # context_vector shape: [batch_size, 256]
        
        # 4. Phân loại
        logits = self.classifier(context_vector) 
        return logits
