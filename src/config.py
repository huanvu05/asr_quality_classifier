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
    ENCODER_MODEL_NAME: str = "openai/whisper-base" 
    CHUNK_DURATION_S: float = 2.0  # Chia khúc 2 giây
    # Whisper-base có 512D. Chúng ta dùng Mean Pooling + Max Pooling = 1024D
    EMBEDDING_DIM: int = 1024 
    
    # DNN Hyperparameters
    SEED: int = 42
    TEST_SIZE: float = 0.2  # 80/20 Train-Val Split
    BATCH_SIZE: int = 128   # Tăng Batch Size để vắt kiệt T4 x2
    EPOCHS: int = 30        # Giảm Epoch vì model hội tụ quá nhanh
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 5e-3 # Tăng gấp 10 lần Weight Decay (L2 penalty) để bóp nghẹt overfitting
    
    # Imbalance
    POS_WEIGHT: float = 2.87 # ~2596/904
    
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
