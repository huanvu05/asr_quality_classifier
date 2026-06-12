"""
conftest.py — Global test configuration and mocks for offline testing.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch

from src.config import config as global_config

@pytest.fixture(autouse=True)
def mock_transformers():
    """Mocks HuggingFace transformers models and tokenizers to run offline."""
    
    # 1. Mock Feature Extractor
    mock_extractor = MagicMock()
    mock_extractor.return_value = {
        "input_values": torch.zeros((2, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((2, 16000), dtype=torch.float32)
    }
    
    # 2. Mock WavLM Model
    mock_wavlm_outputs = MagicMock()
    # Let's say batch=2, seq_len=50, dim=768
    mock_wavlm_outputs.last_hidden_state = torch.zeros((2, 50, 768), dtype=torch.float32)
    
    mock_wavlm = MagicMock()
    mock_wavlm.return_value = mock_wavlm_outputs
    mock_wavlm.to.return_value = mock_wavlm
    
    # 3. Mock Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros((2, 20), dtype=torch.long),
        "attention_mask": torch.ones((2, 20), dtype=torch.float32)
    }
    
    # 4. Mock PhoBERT Model
    mock_phobert_outputs = MagicMock()
    mock_phobert_outputs.last_hidden_state = torch.zeros((2, 20, 768), dtype=torch.float32)
    
    mock_phobert = MagicMock()
    mock_phobert.return_value = mock_phobert_outputs
    mock_phobert.to.return_value = mock_phobert

    # Patches
    with patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=mock_extractor), \
         patch("transformers.WavLMModel.from_pretrained", return_value=mock_wavlm), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_phobert):
        yield

@pytest.fixture
def dummy_df(tmp_path):
    """Creates a dummy dataframe and temporary files for testing."""
    df_data = {
        "username": ["user1", "user2", "user3", "user4"],
        "file_path": ["folder1/clone1.wav", "folder1/clone2.wav", "folder2/clone3.wav", "folder2/clone4.wav"],
        "folder": ["folder1", "folder1", "folder2", "folder2"],
        "file_name": ["clone1.wav", "clone2.wav", "clone3.wav", "clone4.wav"],
        "transcript": ["xin chào Việt Nam", "xin chào Việt Nam", "tôi là robot", "tôi là robot"],
        "kind": ["work", "work", "work", "work"],
        "label": [1, 2, 1, 1],  # 1=usable, 2=unusable
        "label_text": ["usable", "unusable", "usable", "usable"],
        "label_source": ["user1", "user2", "user3", "user4"],
        "updated_at": ["2026-05-25", "2026-05-25", "2026-05-26", "2026-05-26"]
    }
    df = pd.DataFrame(df_data)
    
    # Write empty files so path verification succeeds
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    
    for fp in df["file_path"]:
        full_p = audio_dir / fp
        full_p.parent.mkdir(parents=True, exist_ok=True)
        # Create a tiny 1-second dummy wav file (16000 samples of noise/zeros)
        import soundfile as sf
        sf.write(str(full_p), np.zeros(16000), 16000)
        
    labels_csv = tmp_path / "training.csv"
    df.to_csv(labels_csv, index=False)
    
    return df, labels_csv, audio_dir

@pytest.fixture
def test_config(tmp_path, dummy_df):
    """Generates a test config instance with temporary paths."""
    _, labels_csv, audio_dir = dummy_df
    
    cfg = copy_config(global_config)
    cfg.paths.root = tmp_path
    cfg.paths.data_dir = tmp_path
    cfg.paths.audio_dir = audio_dir
    cfg.paths.labels_csv = labels_csv
    cfg.paths.model_dir = tmp_path / "models"
    cfg.paths.output_dir = tmp_path / "outputs"
    cfg.paths.cache_dir = tmp_path / ".cache"
    
    # Ensure they exist
    cfg.paths.model_dir.mkdir(exist_ok=True)
    cfg.paths.output_dir.mkdir(exist_ok=True)
    cfg.paths.cache_dir.mkdir(exist_ok=True)
    
    # Overwrite device to CPU for testing
    cfg.device = "cpu"
    cfg.n_gpus = 0
    
    return cfg

def copy_config(cfg):
    """Helper to perform shallow copies of dataclass configs."""
    import copy
    new_cfg = copy.copy(cfg)
    new_cfg.paths = copy.copy(cfg.paths)
    new_cfg.audio = copy.copy(cfg.audio)
    new_cfg.text = copy.copy(cfg.text)
    new_cfg.model = copy.copy(cfg.model)
    new_cfg.training = copy.copy(cfg.training)
    new_cfg.azure = copy.copy(cfg.azure)
    return new_cfg
