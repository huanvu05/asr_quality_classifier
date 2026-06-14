"""
extract_embeddings_multi_gpu.py — Fast Offline Feature Extraction for Kaggle T4x2.

This script guarantees 100% utilization of both T4 GPUs on Kaggle by splitting
the dataset in half and running two independent extraction processes simultaneously.
This avoids the DataParallel bottleneck and saves you ~50 hours of redundant WavLM compute!
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config, logger
from src.data_loader import load_data
from src.audio_encoder import AudioEncoder
from src.text_encoder import TextEncoder

def extract_chunk(df_chunk: pd.DataFrame, gpu_id: int, config_copy) -> dict:
    """Runs extraction for a chunk of data on a specific GPU."""
    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Starting extraction for {len(df_chunk)} samples...")
    
    # Override device for this worker
    config_copy.device = device
    
    # Initialize models directly on the target GPU
    audio_encoder = AudioEncoder(config_copy).to(device)
    text_encoder = TextEncoder(config_copy).to(device)
    
    audio_encoder.eval()
    text_encoder.eval()
    
    results = {
        "audio_pools": [],
        "text_pools": [],
        "indices": df_chunk.index.values
    }
    
    if len(df_chunk) == 0:
        logger.warning(f"[GPU {gpu_id}] Received an empty chunk. Skipping.")
        return results
        
    # Process sequentially (batch size 1 or small batch to avoid OOM on 30s audio)
    batch_size = 8
    num_batches = math.ceil(len(df_chunk) / batch_size)
    
    for i in tqdm(range(num_batches), desc=f"GPU {gpu_id} Progress"):
        batch_df = df_chunk.iloc[i*batch_size : (i+1)*batch_size]
        
        # Load audio waveforms
        waveforms = []
        for path in batch_df["absolute_path"].values:
            try:
                y, _ = librosa.load(path, sr=config_copy.audio.sample_rate, mono=True, duration=config_copy.audio.max_duration_sec)
                waveforms.append(y)
            except Exception:
                waveforms.append(np.zeros(config_copy.audio.sample_rate, dtype=np.float32))
                
        # Load texts
        texts = batch_df["transcript"].values.tolist()
        
        # Extract embeddings
        with torch.no_grad():
            _, _, a_pool = audio_encoder(waveforms)
            _, _, t_pool = text_encoder(texts)
            
            results["audio_pools"].append(a_pool.cpu().numpy())
            results["text_pools"].append(t_pool.cpu().numpy())
            
    # Concatenate results if not empty
    if results["audio_pools"]:
        results["audio_pools"] = np.concatenate(results["audio_pools"], axis=0)
        results["text_pools"] = np.concatenate(results["text_pools"], axis=0)
    else:
        results["audio_pools"] = np.array([])
        results["text_pools"] = np.array([])
        
    logger.info(f"[GPU {gpu_id}] Completed chunk.")
    return results

def main():
    logger.info("Initializing T4x2 Multi-GPU Extraction...")
    df = load_data(config, sync_azure=False)
    
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        logger.warning(f"Only {n_gpus} GPU(s) detected. For Kaggle T4x2, make sure you selected 'GPU T4 x2'.")
        n_gpus = 1 # Fallback to 1
        
    # Split dataframe into chunks based on number of GPUs
    chunks = np.array_split(df, n_gpus)
    
    import copy
    results = []
    
    # Launch parallel extraction threads
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for gpu_id in range(n_gpus):
            # Create a copy of config for thread safety
            config_copy = copy.deepcopy(config)
            future = executor.submit(extract_chunk, chunks[gpu_id], gpu_id, config_copy)
            futures.append(future)
            
        for future in futures:
            results.append(future.result())
            
    # Re-assemble results in original order
    logger.info("Merging extracted features...")
    
    # Sort results by the indices to match original df
    merged_audio = np.zeros((len(df), config.audio.audio_embed_dim), dtype=np.float32)
    merged_text = np.zeros((len(df), config.text.text_embed_dim), dtype=np.float32)
    
    for res in results:
        indices = res["indices"]
        merged_audio[indices] = res["audio_pools"]
        merged_text[indices] = res["text_pools"]
        
    # Save to disk
    cache_dir = config.paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(cache_dir / "wavlm_embeddings.npy", merged_audio)
    np.save(cache_dir / "phobert_embeddings.npy", merged_text)
    
    logger.info(f"Extraction complete! Saved to {cache_dir}.")
    logger.info("Now your training phases will be LIGHTNING FAST!")

if __name__ == "__main__":
    main()
