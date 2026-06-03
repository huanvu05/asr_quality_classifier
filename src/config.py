import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

@dataclass
class Config:
    """
    Configuration for Deep Audio Classification Pipeline.
    """
    # Environment
    AZURE_SAS_TOKEN: str = field(default_factory=lambda: os.getenv("AZURE_SAS_TOKEN", ""))
    AZURE_STORAGE_URL: str = field(default_factory=lambda: os.getenv("AZURE_STORAGE_URL", ""))
    
    # Paths
    BASE_DIR: str = "."
    DATA_DIR: str = "data"
    AUDIO_DIR: str = "data/audio"
    MODELS_DIR: str = "models"
    
    # Audio & Encoder Settings
    SAMPLE_RATE: int = 16000
    ENCODER_MODEL_NAME: str = "openai/whisper-base" # Tăng lên Base để lấy embedding 512 chiều tốt hơn
    EMBEDDING_DIM: int = 512 # Của Whisper-base
    
    # DNN Hyperparameters
    SEED: int = 42
    N_FOLDS: int = 5
    BATCH_SIZE: int = 64
    EPOCHS: int = 40
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    
    # Class Imbalance (Total 3500: 2600 Usable (1), 900 Unusable (0))
    # PyTorch BCEWithLogitsLoss pos_weight = negative_samples / positive_samples
    # Chú ý: Ở đây ta muốn mô hình chú ý đến class 0 (Unusable). 
    # Nhưng label 1 (Usable) là Positive. Để model cẩn thận hơn khi dự đoán 1 (nghĩa là phạt nặng nếu đoán 1 mà sự thật là 0),
    # ta có thể cấu hình pos_weight hoặc dùng Class Weights cho CrossEntropyLoss.
    # Trong bài toán này, dự đoán đúng class 0 rất quan trọng.
    POS_WEIGHT: float = 900 / 2600.0  # ~0.346
    
    # FORCE CPU: The Colab P100 GPU is strictly incompatible with the current PyTorch version.
    # Running on CPU guarantees success.
    DEVICE: str = "cpu"

    def __post_init__(self):
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        self.set_seed(self.SEED)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
