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
    
    # --- ĐẠI TU KIẾN TRÚC ---
    SAMPLE_RATE: int = 16000
    
    # Đổi sang Wav2Vec2 (Bản Base 768 chiều để tối ưu tốc độ/bộ nhớ trên T4)
    # Bản XLSR-53 1024D quá nặng có thể gây OOM trên Colab
    ENCODER_MODEL_NAME: str = "facebook/wav2vec2-base" 
    EMBEDDING_DIM: int = 768 
    
    # Chuẩn hóa độ dài Sequence cho Attention (Ví dụ: 300 frames ~ 6 giây)
    MAX_SEQ_LENGTH: int = 400 
    
    # DNN Hyperparameters
    SEED: int = 42
    TEST_SIZE: float = 0.2  
    BATCH_SIZE: int = 32   # Giảm Batch Size xuống 32 vì Sequence 400x768 tốn nhiều RAM hơn
    EPOCHS: int = 40       
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    
    # Imbalance
    POS_WEIGHT: float = 2.87
    
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
