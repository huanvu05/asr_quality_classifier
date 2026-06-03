import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

@dataclass
class Config:
    """
    Configuration class for ASR Quality Classifier.
    Manages hyperparameters, paths, and environment secrets.
    """
    # Environment & Security
    AZURE_SAS_TOKEN: str = field(default_factory=lambda: os.getenv("AZURE_SAS_TOKEN", ""))
    AZURE_STORAGE_URL: str = field(default_factory=lambda: os.getenv("AZURE_STORAGE_URL", ""))
    
    # Project Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    AUDIO_DIR: str = os.path.join(DATA_DIR, "audio")
    TRANSCRIPT_DIR: str = os.path.join(DATA_DIR, "transcripts")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    # Audio Processing
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # Model Hyperparameters
    SEED: int = 42
    N_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    
    # LightGBM Params (Optimized for imbalanced data)
    LGBM_PARAMS: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,  # Adjusted from 0.05
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": 0.346,  # Corrected: ratio of minority (900) / majority (2600)
        "n_estimators": 500,
        "max_depth": 6,
    })
    
    # Feature Extraction
    WHISPER_MODEL_NAME: str = "openai/whisper-tiny"  # Lightweight for fast inference
    
    # UI / Logging
    VERBOSE: bool = True

    def __post_init__(self):
        # Validate critical credentials
        if not self.AZURE_SAS_TOKEN:
            print("WARNING: AZURE_SAS_TOKEN not found in environment variables.")
        
        # Ensure directories exist
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.TRANSCRIPT_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        # Set global seeds for reproducibility
        self.set_seed(self.SEED)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

config = Config()
