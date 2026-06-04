import os
import random
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class Config:
    # Environment
    AZURE_SAS_TOKEN: str = field(default_factory=lambda: os.getenv("AZURE_SAS_TOKEN", ""))
    AZURE_STORAGE_URL: str = field(default_factory=lambda: os.getenv("AZURE_STORAGE_URL", ""))
    
    # Paths
    BASE_DIR: str = "."
    DATA_DIR: str = "data"
    AUDIO_DIR: str = "data/audio/data2"
    MODELS_DIR: str = "models"
    CSV_PATH: str = "data/transcripts/training.csv"
    
    # Audio & SOTA Encoder Settings
    SAMPLE_RATE: int = 16000
    ENCODER_MODEL_NAME: str = "facebook/wav2vec2-large-xlsr-53"  # SOTA for Acoustic Representations
    EMBEDDING_DIM: int = 1024  # Của bản Large
    MAX_SEQ_LENGTH: int = 300  # Trimming/Padding sequence length for Attention
    
    # DNN Hyperparameters
    SEED: int = 42
    N_FOLDS: int = 5
    BATCH_SIZE: int = 64
    EPOCHS: int = 150
    LEARNING_RATE: float = 3e-4 # Tăng mạnh LR từ 1e-4 lên 3e-4
    WEIGHT_DECAY: float = 1e-4
    
    # Imbalance
    POS_WEIGHT: float = 900 / 2600.0  
    
    # Cho phép sử dụng lại GPU. 
    # Bắt buộc người dùng phải chọn T4 GPU trên Kaggle.
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        self.set_seed(self.SEED)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
