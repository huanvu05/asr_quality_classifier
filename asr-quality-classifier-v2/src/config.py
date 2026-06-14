"""
config.py — Centralized configuration for ASR Quality Classifier v2.

All hyperparameters, paths, and environment variable validation live here.
Never hardcode credentials — always read from environment variables.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # asr-quality-classifier-v2/
_IS_KAGGLE = Path("/kaggle/working").exists()
_IS_COLAB = Path("/content").exists() and not _IS_KAGGLE


@dataclass
class PathConfig:
    """File-system paths. Automatically resolves Kaggle / Colab / local."""

    root: Path = field(default_factory=lambda: _ROOT)

    # Data directories
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    audio_dir: Path = field(default_factory=lambda: _ROOT / "data" / "audio")
    transcript_dir: Path = field(default_factory=lambda: _ROOT / "data" / "transcripts")

    # Model & output directories
    model_dir: Path = field(default_factory=lambda: _ROOT / "models")
    output_dir: Path = field(default_factory=lambda: _ROOT / "outputs")
    cache_dir: Path = field(default_factory=lambda: _ROOT / ".cache")

    # CSV labels file
    labels_csv: Path = field(
        default_factory=lambda: _ROOT / "data" / "transcripts" / "training.csv"
    )

    def __post_init__(self):
        # Override for cloud environments
        if _IS_KAGGLE:
            # Check if user manually mounted the dataset via Kaggle UI
            input_base = Path("/kaggle/input")
            found_csv = list(input_base.glob("**/training.csv"))
            
            if found_csv:
                # Dataset is mounted manually
                self.labels_csv = found_csv[0]
                self.data_dir = self.labels_csv.parent.parent if self.labels_csv.parent.name == "transcripts" else self.labels_csv.parent
                
                # Look for data2 directory specifically
                data2_dir = self.data_dir / "audio" / "data2"
                self.audio_dir = data2_dir if data2_dir.exists() else self.data_dir / "audio"
                logger.info(f"Auto-detected Kaggle dataset at: {self.data_dir}, Audio at: {self.audio_dir}")
            else:
                # Fallback to Azure download location
                self.data_dir = Path("/kaggle/working/data")
                self.labels_csv = self.data_dir / "transcripts" / "training.csv"
                self.audio_dir = self.data_dir / "audio"
                
            self.output_dir = Path("/kaggle/working/outputs")
            self.model_dir = Path("/kaggle/working/models")
            self.cache_dir = Path("/kaggle/working/.cache")

        elif _IS_COLAB:
            self.output_dir = Path("/content/outputs")
            self.model_dir = Path("/content/models")
            self.cache_dir = Path("/content/.cache")

        # Ensure output directories exist
        for d in [self.model_dir, self.output_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio processing hyperparameters."""

    sample_rate: int = 16_000          # Target sample rate (Hz)
    max_duration_sec: float = 30.0     # Clip audio longer than this
    min_duration_sec: float = 0.1      # Discard audio shorter than this

    # Acoustic feature extraction
    n_fft: int = 1024
    hop_length: int = 256
    n_mfcc: int = 13
    silence_top_db: float = 40.0       # dB threshold for silence detection
    frame_length: int = 1024

    # Audio encoder
    audio_encoder_name: str = "microsoft/wavlm-base-plus"
    audio_encoder_layer: int = -1      # Which hidden layer to pool (-1 = last)
    audio_embed_dim: int = 768         # WavLM-base hidden dim
    max_audio_frames: int = 500        # Max sequence length after encoder


@dataclass
class TextConfig:
    """Text processing hyperparameters."""

    text_encoder_name: str = "vinai/phobert-base-v2"
    text_embed_dim: int = 768          # PhoBERT hidden dim
    max_token_length: int = 256        # Max tokens for PhoBERT


@dataclass
class ModelConfig:
    """Cross-modal model architecture hyperparameters."""

    # Projection dimensions
    proj_dim: int = 256

    # Cross-attention
    n_heads: int = 4
    attn_dropout: float = 0.1

    # MLP classifier
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    mlp_dropout: float = 0.3

    # Handcrafted acoustic features (see audio_features.py)
    n_acoustic_features: int = 37      # Updated after EDA

    # Cross-modal features
    n_crossmodal_features: int = 6


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    seed: int = 42
    n_folds: int = 5
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # Class imbalance: 2596 usable / 904 unusable ≈ 2.87
    pos_class_weight: float = 2.87

    # Decision threshold sweep
    threshold_start: float = 0.05
    threshold_end: float = 0.95
    threshold_step: float = 0.01

    # LightGBM
    lgbm_n_estimators: int = 2000
    lgbm_learning_rate: float = 0.02
    lgbm_num_leaves: int = 63
    lgbm_feature_fraction: float = 0.7
    lgbm_bagging_fraction: float = 0.8
    lgbm_early_stopping_rounds: int = 200


@dataclass
class AzureConfig:
    """Azure Blob Storage configuration. Reads from environment variables only."""

    sas_token: Optional[str] = field(default=None, init=False)
    account_url: str = "https://asr.blob.core.windows.net"
    container_name: str = "training"
    upload_prefix: str = "vu_van_huan/asr-quality-classifier-v2"

    def __post_init__(self):
        self.sas_token = os.getenv("AZURE_SAS_TOKEN")
        if self.sas_token is None:
            logger.warning(
                "AZURE_SAS_TOKEN is not set. "
                "Azure upload/download will be disabled. "
                "Set this environment variable before running data download."
            )

    @property
    def is_available(self) -> bool:
        return self.sas_token is not None


# ---------------------------------------------------------------------------
# Master config (singleton-like access)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Master configuration object. Compose all sub-configs here."""

    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)

    # Computed device
    device: str = field(default="cpu", init=False)
    n_gpus: int = field(default=0, init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.n_gpus = torch.cuda.device_count()
            logger.info(f"GPU detected: {self.n_gpus} × {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Apple MPS detected.")
        else:
            self.device = "cpu"
            logger.info("Running on CPU.")

        self._set_seeds()

    def _set_seeds(self):
        import random
        import numpy as np

        random.seed(self.training.seed)
        np.random.seed(self.training.seed)
        torch.manual_seed(self.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training.seed)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ASR Quality Classifier v2 — Config Summary",
            "=" * 60,
            f"  Device        : {self.device} (GPUs: {self.n_gpus})",
            f"  Audio encoder : {self.audio.audio_encoder_name}",
            f"  Text encoder  : {self.text.text_encoder_name}",
            f"  Seed          : {self.training.seed}",
            f"  Folds         : {self.training.n_folds}",
            f"  Azure ready   : {self.azure.is_available}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Module-level singleton
config = Config()
