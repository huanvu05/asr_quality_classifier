"""
data_loader.py — Data loading, path resolution, Azure download, and split logic.

All splits must group by 'folder' (or 'transcript') to prevent data leakage.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from azure.storage.blob import BlobServiceClient

from src.config import Config, logger

def resolve_audio_path(file_path: str, audio_dir: Path) -> Optional[Path]:
    """
    Resolves the absolute path to an audio file on disk.
    Checks audio_dir directly and audio_dir/data2.
    Also handles paths recursively if mounted from Kaggle.
    """
    # 1. Direct check
    path1 = audio_dir / file_path
    if path1.exists():
        return path1
    
    # 2. Check under "data2" (Azure structure)
    path2 = audio_dir / "data2" / file_path
    if path2.exists():
        return path2
        
    # 3. Aggressive recursive search (for Kaggle datasets where structure might be flattened)
    # E.g. file_path is "folder1/audio.wav", but Kaggle mounted it as "/kaggle/input/dataset/folder1/audio.wav"
    # We search the parent of audio_dir just in case.
    search_base = audio_dir.parent
    try:
        # Assuming file_path looks like "folder_name/file.wav"
        folder_name, file_name = os.path.split(file_path)
        found = list(search_base.glob(f"**/{folder_name}/{file_name}"))
        if found:
            return found[0]
        
        # If still not found, search just by file_name
        found = list(search_base.glob(f"**/{file_name}"))
        if found:
            return found[0]
    except Exception:
        pass
        
    return None

def download_file_from_azure(
    blob_name: str, 
    dest_path: Path, 
    container_client
) -> bool:
    """Downloads a single blob to dest_path."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = container_client.get_blob_client(blob_name)
        with open(dest_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        return True
    except Exception as e:
        logger.error(f"Failed to download {blob_name} to {dest_path}: {e}")
        return False

def sync_data_from_azure(config: Config):
    """
    Synchronizes the audio data and transcripts from Azure Blob Storage.
    Uses ThreadPoolExecutor for concurrent downloads of missing files.
    """
    if not config.azure.is_available:
        logger.info("Azure SAS token not configured. Skipping Azure synchronization.")
        return

    logger.info("Initializing Azure Blob Storage connection...")
    try:
        blob_service_client = BlobServiceClient(
            account_url=config.azure.account_url, 
            credential=config.azure.sas_token
        )
        container_client = blob_service_client.get_container_client(
            config.azure.container_name
        )
        
        # 1. Download training.csv if missing
        labels_csv = config.paths.labels_csv
        if not labels_csv.exists():
            logger.info("Downloading training.csv from Azure...")
            blob_name = f"{config.azure.upload_prefix}/transcripts/training.csv"
            success = download_file_from_azure(blob_name, labels_csv, container_client)
            if not success:
                logger.error("Could not download training.csv. Aborting sync.")
                return
        else:
            logger.info("training.csv already exists locally.")

        # Read the csv to get the list of files to download
        df = pd.read_csv(labels_csv)
        missing_files = []
        for _, row in df.iterrows():
            file_path = row["file_path"]
            # Check if resolved locally
            local_path = resolve_audio_path(file_path, config.paths.audio_dir)
            if local_path is None:
                # Target path under audio_dir / "data2" / file_path
                target_path = config.paths.audio_dir / "data2" / file_path
                missing_files.append((file_path, target_path))

        if not missing_files:
            logger.info("All audio files are present locally. No download needed.")
            return

        logger.info(f"Found {len(missing_files)} missing audio files. Downloading concurrently...")
        
        # Concurrently download missing audio files
        max_workers = 8
        downloaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            for file_path, target_path in missing_files:
                # The blob path in storage
                blob_name = f"{config.azure.upload_prefix}/audio/data2/{file_path}"
                future = executor.submit(
                    download_file_from_azure, 
                    blob_name, 
                    target_path, 
                    container_client
                )
                future_to_file[future] = file_path

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                if future.result():
                    downloaded_count += 1
                    if downloaded_count % 100 == 0 or downloaded_count == len(missing_files):
                        logger.info(f"Downloaded {downloaded_count}/{len(missing_files)} files.")

        logger.info(f"Synchronization complete. Downloaded {downloaded_count} files.")
    except Exception as e:
        logger.error(f"Error during Azure sync: {e}")

def load_data(config: Config, sync_azure: bool = False) -> pd.DataFrame:
    """
    Loads and validates the training labels dataframe.
    Filters out rows with missing audio files.
    """
    if sync_azure:
        sync_data_from_azure(config)

    labels_csv = config.paths.labels_csv
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV file not found at {labels_csv}. "
            "Please configure AZURE_SAS_TOKEN to download the data or place it manually."
        )

    logger.info(f"Loading data from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    
    # Required columns check
    required_cols = ["file_path", "folder", "file_name", "transcript", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in labels CSV.")

    # Convert label: 1 -> 1 (usable), 2 -> 0 (unusable)
    # Let's map explicitly: 1 to 1, any other value (usually 2) to 0.
    df["binary_label"] = (df["label"] == 1).astype(int)

    # Resolve audio paths
    absolute_paths = []
    missing_indices = []
    
    for idx, row in df.iterrows():
        abs_path = resolve_audio_path(row["file_path"], config.paths.audio_dir)
        if abs_path is not None:
            absolute_paths.append(str(abs_path))
        else:
            absolute_paths.append(None)
            missing_indices.append(idx)

    df["absolute_path"] = absolute_paths
    
    if missing_indices:
        logger.warning(
            f"Could not find audio files for {len(missing_indices)} out of {len(df)} rows. "
            f"These rows will be excluded. First few missing: "
            f"{df.iloc[missing_indices[:5]]['file_path'].tolist()}"
        )
        df = df.dropna(subset=["absolute_path"]).reset_index(drop=True)
    
    # Compute duration for each transcript to help filter/validate
    logger.info(f"Dataset successfully loaded. Total rows: {len(df)}.")
    logger.info(f"Class distribution:\n{df['binary_label'].value_counts(normalize=True)}")
    return df

def get_kfold_splits(
    df: pd.DataFrame, 
    config: Config, 
    group_col: str = "folder"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates GroupKFold split indices based on the grouping column.
    """
    logger.info(f"Generating {config.training.n_folds}-fold splits grouped by '{group_col}'...")
    gkf = GroupKFold(n_splits=config.training.n_folds)
    
    groups = df[group_col].values
    X = np.arange(len(df))
    y = df["binary_label"].values
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        splits.append((train_idx, val_idx))
        train_groups = set(df.iloc[train_idx][group_col])
        val_groups = set(df.iloc[val_idx][group_col])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Leakage detected in fold {fold}: {overlap}"
        
    return splits

def get_train_val_test_splits(
    df: pd.DataFrame, 
    config: Config, 
    group_col: str = "folder",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into Train, Val, and Test sets using GroupShuffleSplit
    to prevent groups/folders from leaking between splits.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0)
    
    groups = df[group_col].values
    
    # 1. Split into (Train + Val) and Test
    test_ratio = test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=config.training.seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Split (Train + Val) into Train and Val
    val_ratio_of_train_val = val_size / (train_size + val_size)
    groups_train_val = df_train_val[group_col].values
    
    gss2 = GroupShuffleSplit(
        n_splits=1, 
        test_size=val_ratio_of_train_val, 
        random_state=config.training.seed
    )
    train_idx, val_idx = next(gss2.split(df_train_val, groups=groups_train_val))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    # Validation checks for leakage
    train_groups = set(df_train[group_col])
    val_groups = set(df_val[group_col])
    test_groups = set(df_test[group_col])
    
    assert not train_groups.intersection(val_groups), "Leakage between Train and Val!"
    assert not train_groups.intersection(test_groups), "Leakage between Train and Test!"
    assert not val_groups.intersection(test_groups), "Leakage between Val and Test!"
    
    logger.info(
        f"Train/Val/Test split created: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
    )
    return df_train, df_val, df_test
