import os
import argparse
import datetime
import pandas as pd
import numpy as np
import torch
import joblib
from src.config import config
from src.data_loader import load_labels
from src.features import DeepAudioChunkExtractor
from src.evaluator import Evaluator
import json

def create_metadata(run_id: str, metrics: dict, params: dict):
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "params": params,
        "architecture": "WhisperChunk_MeanMax_DNN"
    }
    meta_path = os.path.join(config.MODELS_DIR, f"{run_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return meta_path

def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_chunk_dnn"
    print(f"Starting Chunk & Pool Pipeline: {run_id}")

    # 1. Load Labels
    labels_df = load_labels(args.labels_path)

    # 2. Extract Features (Chunking + Pooling)
    features_path = os.path.join(config.DATA_DIR, f"{run_id}_embeddings.pkl")
    
    if not args.skip_extraction:
        extractor = DeepAudioChunkExtractor()
        data = extractor.process_dataset(labels_df, features_path)
    else:
        print(f"Loading precomputed chunked embeddings from {args.features_path}")
        data = joblib.load(args.features_path)

    if not data:
        print("Error: No data to process. Exiting.")
        return

    # 3. Prepare Data for PyTorch
    embeddings = []
    labels = []
    
    for item in data:
        emb = item['embedding']
        # Check if shape is exactly 1024 (512 Mean + 512 Max)
        if emb.shape[0] != config.EMBEDDING_DIM:
            print(f"Warning: Unexpected shape {emb.shape} for {item['file_name']}. Skipping.")
            continue
            
        label = 1 if str(item['label']) == '1' else 0
        embeddings.append(emb)
        labels.append(label)
        
    X = np.array(embeddings)
    y = np.array(labels)

    # 4. Training (Single Fold)
    print("\nStarting PyTorch DNN Training (80/20 Split)...")
    evaluator = Evaluator()
    y_val, val_probs = evaluator.run_training(X, y)

    # 5. Threshold Optimization on Validation Set
    optimal_threshold = evaluator.optimize_threshold(y_val, val_probs)
    evaluator.evaluate_results(y_val, val_probs, optimal_threshold)

    # 6. Save Metadata
    metrics = {
        "val_macro_f1": evaluator.optimize_threshold(y_val, val_probs),
        "optimal_threshold": optimal_threshold
    }
    
    params = {
        "encoder": config.ENCODER_MODEL_NAME,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LEARNING_RATE,
        "pos_weight": config.POS_WEIGHT,
        "chunk_duration": config.CHUNK_DURATION_S
    }
    meta_path = create_metadata(run_id, metrics, params)

    print(f"\n[DONE] Pipeline completed. Run ID: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=str, default=config.CSV_PATH)
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction if embeddings exist")
    parser.add_argument("--features-path", type=str, help="Path to precomputed .pkl file")
    args = parser.parse_args()
    main(args)
