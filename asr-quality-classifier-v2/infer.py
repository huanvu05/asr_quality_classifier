"""
infer.py — Standalone inference CLI.

Usage:
  python infer.py --audio path/to/audio.wav --transcript "văn bản"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to python path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import config, logger
from src.audio_features import process_audio_file
from src.classifier import ASRQualityClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="ASR Data Quality Classifier Inference")
    parser.add_argument(
        "--audio", 
        type=str, 
        required=True, 
        help="Path to raw audio file (.wav)"
    )
    parser.add_argument(
        "--transcript", 
        type=str, 
        required=True, 
        help="Transcript text corresponding to the audio"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found at: {audio_path}")
        sys.exit(1)
        
    # 1. Feature extraction
    logger.info("Extracting handcrafted features...")
    ac_arr, cm_arr = process_audio_file(str(audio_path), args.transcript, config)
    
    # 2. Prepare inputs for neural network
    # Since process_audio_file returns 0s on load failure, we check
    if np.all(ac_arr == 0.0) and np.all(cm_arr == 0.0):
        logger.error("Audio loading or processing failed. Cannot perform inference.")
        sys.exit(1)
        
    # Load raw waveform for WavLM
    import librosa
    try:
        y, _ = librosa.load(
            str(audio_path), 
            sr=config.audio.sample_rate, 
            mono=True, 
            duration=config.audio.max_duration_sec
        )
    except Exception as e:
        logger.error(f"Failed to load audio for model encoder: {e}")
        sys.exit(1)
        
    # Format tensors
    ac_tensor = torch.tensor(ac_arr, dtype=torch.float32).unsqueeze(0)  # batch size 1
    cm_tensor = torch.tensor(cm_arr, dtype=torch.float32).unsqueeze(0)
    waveforms = [y]
    texts = [args.transcript]
    
    # 3. Load trained cross-modal fusion model(s)
    # Check if we have checkpoints
    model_dir = config.paths.model_dir
    checkpoints = list(model_dir.glob("crossmodal_fold*.pt"))
    
    if not checkpoints:
        logger.error(
            f"No trained model checkpoints found in {model_dir}. "
            "Please run train.py to train the models first."
        )
        sys.exit(1)
        
    logger.info(f"Found {len(checkpoints)} cross-modal checkpoints. Running ensemble prediction...")
    
    # Set model to evaluation on appropriate device
    device = config.device
    
    all_probs = []
    
    # Run prediction across all available fold models
    for cp_path in checkpoints:
        # Load checkpoint
        checkpoint = torch.load(str(cp_path), map_location=device)
        best_threshold = checkpoint.get("best_threshold", 0.5)
        
        # Instantiate model in full mode
        model = ASRQualityClassifier(config, mode="full")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            logits = model(waveforms, texts, ac_tensor, cm_tensor)
            prob = torch.sigmoid(logits).item()
            all_probs.append((prob, best_threshold))
            
    # Calculate average probability and average threshold
    avg_prob = np.mean([p for p, _ in all_probs])
    avg_threshold = np.mean([t for _, t in all_probs])
    
    is_usable = avg_prob >= avg_threshold
    label = 1 if is_usable else 0
    label_text = "usable" if is_usable else "unusable"
    
    print("\n" + "=" * 50)
    print("                CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"Audio Path    : {args.audio}")
    print(f"Transcript    : {args.transcript}")
    print(f"Confidence    : {avg_prob:.4f}")
    print(f"Threshold     : {avg_threshold:.2f}")
    print(f"Prediction    : {label} ({label_text.upper()})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
