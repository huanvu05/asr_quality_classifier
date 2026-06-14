"""
phase1_eda.py — Phase 1: Exploratory Data Analysis.

Analyzes class distributions, duration statistics, word counts, and generates EDA plots.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import librosa

from src.config import config, logger
from src.data_loader import load_data

def run_eda():
    logger.info("Starting Phase 1 EDA...")
    
    # 1. Load label dataframe (without sync to keep EDA local first)
    try:
        df = load_data(config, sync_azure=False)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Make sure the dataset is downloaded to data/ before running.")
        return
        
    # Create output dir
    eda_out = config.paths.output_dir / "eda"
    eda_out.mkdir(parents=True, exist_ok=True)
    
    # 2. Compute basic stats
    total_samples = len(df)
    usable_count = (df["binary_label"] == 1).sum()
    unusable_count = (df["binary_label"] == 0).sum()
    usable_pct = usable_count / total_samples * 100
    unusable_pct = unusable_count / total_samples * 100
    
    unique_folders = df["folder"].nunique()
    unique_transcripts = df["transcript"].nunique()
    
    # Check for conflict folders (folders having both usable and unusable samples)
    folder_labels = df.groupby("folder")["binary_label"].nunique()
    conflict_folders = (folder_labels > 1).sum()
    
    print("\n" + "=" * 50)
    print("              DATASET CHARACTERISTICS")
    print("=" * 50)
    print(f"Total samples         : {total_samples}")
    print(f"Usable (1)            : {usable_count} ({usable_pct:.2f}%)")
    print(f"Unusable (0)          : {unusable_count} ({unusable_pct:.2f}%)")
    print(f"Unique Folders        : {unique_folders}")
    print(f"Unique Transcripts    : {unique_transcripts}")
    print(f"Conflicting Folders   : {conflict_folders}")
    print("=" * 50 + "\n")
    
    # 3. Add text stats
    df["word_count"] = df["transcript"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["transcript"].apply(lambda t: len(str(t)))
    
    # 4. Compute duration for a subset of audio files to save time, or all if quick
    # Since loading audio is slow, we will sample up to 100 files to get duration distribution
    logger.info("Computing duration statistics for a sample of 200 files...")
    sample_df = df.sample(min(200, len(df)), random_state=config.training.seed).copy()
    durations = []
    for path in sample_df["absolute_path"]:
        try:
            d = librosa.get_duration(path=path)
            durations.append(d)
        except Exception:
            durations.append(0.0)
    sample_df["duration"] = durations
    sample_df = sample_df[sample_df["duration"] > 0]
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label_text", palette="Set2")
    plt.title("ASR Quality Class Distribution", fontsize=14, pad=15)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(eda_out / "class_distribution.png", dpi=300)
    plt.close()
    
    # Plot 2: Transcript Word Count Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="word_count", hue="label_text", kde=True, multiple="stack", palette="Set1")
    plt.title("Transcript Word Count Distribution by Class", fontsize=14, pad=15)
    plt.xlabel("Word Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(eda_out / "word_count_distribution.png", dpi=300)
    plt.close()
    
    # Plot 3: Audio Duration Distribution (on sample)
    if len(sample_df) > 0:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=sample_df, x="duration", hue="label_text", kde=True, multiple="stack", palette="Set2")
        plt.title("Audio Duration Distribution (200 sampled files)", fontsize=14, pad=15)
        plt.xlabel("Duration (seconds)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(eda_out / "duration_distribution.png", dpi=300)
        plt.close()
        
    logger.info(f"EDA plots saved to {eda_out}")
    logger.info("Phase 1 EDA completed successfully.")

if __name__ == "__main__":
    run_eda()
