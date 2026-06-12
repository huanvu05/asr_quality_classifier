"""
test_data_loader.py — Tests for loading and splitting data.
"""

import pytest
import pandas as pd
from src.data_loader import load_data, get_kfold_splits, get_train_val_test_splits, resolve_audio_path

def test_resolve_audio_path(tmp_path):
    # Setup test file
    folder = tmp_path / "data2" / "test_folder"
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / "test.wav"
    file_path.touch()
    
    # Test resolution
    resolved = resolve_audio_path("test_folder/test.wav", tmp_path)
    assert resolved is not None
    assert resolved.exists()

def test_load_data(test_config):
    df = load_data(test_config, sync_azure=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "binary_label" in df.columns
    assert "absolute_path" in df.columns
    # Check that label 1 maps to 1, and label 2 maps to 0
    assert df.loc[df["file_name"] == "clone1.wav", "binary_label"].values[0] == 1
    assert df.loc[df["file_name"] == "clone2.wav", "binary_label"].values[0] == 0

def test_get_kfold_splits(test_config):
    df = load_data(test_config, sync_azure=False)
    splits = get_kfold_splits(df, test_config, group_col="folder")
    assert len(splits) == test_config.training.n_folds
    
    for train_idx, val_idx in splits:
        train_folders = set(df.iloc[train_idx]["folder"])
        val_folders = set(df.iloc[val_idx]["folder"])
        # No overlap between train and val folders
        assert len(train_folders.intersection(val_folders)) == 0

def test_get_train_val_test_splits(test_config):
    # To test train/val/test splits, we need a larger dataset with more distinct groups
    # Create a dummy df with 10 groups
    data = {
        "file_path": [f"f{i}/c.wav" for i in range(10)],
        "folder": [f"f{i}" for i in range(10)],
        "file_name": ["c.wav"] * 10,
        "transcript": ["t"] * 10,
        "label": [1] * 10,
        "binary_label": [1] * 10,
        "absolute_path": [f"f{i}/c.wav" for i in range(10)]
    }
    df = pd.DataFrame(data)
    
    df_train, df_val, df_test = get_train_val_test_splits(
        df, 
        test_config, 
        group_col="folder",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2
    )
    
    # Assert sizes
    assert len(df_train) == 6
    assert len(df_val) == 2
    assert len(df_test) == 2
    
    # Assert group integrity
    train_g = set(df_train["folder"])
    val_g = set(df_val["folder"])
    test_g = set(df_test["folder"])
    assert len(train_g.intersection(val_g)) == 0
    assert len(train_g.intersection(test_g)) == 0
