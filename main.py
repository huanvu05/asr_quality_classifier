import os
import argparse
import datetime
import pandas as pd
import numpy as np
import torch
import joblib
from src.config import config
from src.data_loader import load_labels
from src.features import DeepAudioExtractor
from src.model import AudioMLP, AudioDataset
from src.evaluator import Evaluator
from torch.utils.data import DataLoader
import json

def create_metadata(run_id: str, metrics: dict, params: dict):
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "params": params,
        "architecture": "WhisperEncoder_PyTorchMLP"
    }
    meta_path = os.path.join(config.MODELS_DIR, f"{run_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return meta_path

def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_deep_audio"
    print(f"Starting Deep Audio Pipeline: {run_id}")

    # 1. Load Labels
    labels_df = load_labels(args.labels_path)

    # 2. Extract Deep Embeddings
    features_path = os.path.join(config.DATA_DIR, f"{run_id}_embeddings.pkl")
    
    if not args.skip_extraction:
        extractor = DeepAudioExtractor()
        data = extractor.process_dataset(labels_df, features_path)
    else:
        print(f"Loading precomputed embeddings from {args.features_path}")
        data = joblib.load(args.features_path)

    if not data:
        print("Error: No data to process. Exiting.")
        return

    # 3. Prepare Data for PyTorch
    embeddings = []
    labels = []
    
    for item in data:
        # Get the 512D embedding
        emb = item['embedding']
        # Map label: '1' -> 1, others -> 0 (Usable vs Unusable)
        label = 1 if str(item['label']) == '1' else 0
        
        embeddings.append(emb)
        labels.append(label)
        
    X = np.array(embeddings)
    y = np.array(labels)

    # 4. Training & Cross-Validation
    print("Starting PyTorch DNN Training...")
    evaluator = Evaluator(n_folds=config.N_FOLDS)
    oof_probs = evaluator.run_cv(X, y)

    # 5. Threshold Optimization
    optimal_threshold = evaluator.optimize_threshold(y, oof_probs)
    evaluator.evaluate_oof(y, oof_probs, optimal_threshold)

    # 6. Train Final Model on Full Dataset
    print("Training final model on full dataset...")
    full_dataset = AudioDataset(X, y)
    full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    final_model = AudioMLP().to(config.DEVICE)
    final_model = evaluator.train_model(final_model, full_loader, full_loader) # Use train as val for final fit to utilize early stopping if needed, or just run max epochs
    
    # 7. Save Artifacts
    model_path = os.path.join(config.MODELS_DIR, f"{run_id}_model.pth")
    torch.save(final_model.state_dict(), model_path)
    
    metrics = {
        "oof_macro_f1": evaluator.optimize_threshold(y, oof_probs), # Store best F1
        "optimal_threshold": optimal_threshold
    }
    
    params = {
        "encoder": config.ENCODER_MODEL_NAME,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LEARNING_RATE,
        "pos_weight": config.POS_WEIGHT
    }
    meta_path = create_metadata(run_id, metrics, params)

    print(f"\n[DONE] Artifacts saved locally in {config.MODELS_DIR}:")
    print(f"- PyTorch Model: {model_path}")
    print(f"- Metadata: {meta_path}")
    print(f"- Confusion Matrix: confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=str, default="data/transcripts/training.csv")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip Whisper extraction if embeddings already exist")
    parser.add_argument("--features-path", type=str, help="Path to precomputed .pkl file if skipping extraction")
    args = parser.parse_args()
    main(args)
