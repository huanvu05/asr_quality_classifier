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
        
    # 3. Load trained models
    model_dir = config.paths.model_dir
    
    # We use the optimal ensemble: 25% Baseline + 75% Deep Audio
    lgb_checkpoints = list(model_dir.glob("baseline_fold*.pkl"))
    deep_checkpoints = list(model_dir.glob("deep_audio_fold*.pt"))
    
    if not lgb_checkpoints and not deep_checkpoints:
        logger.error(f"No trained model checkpoints found in {model_dir}.")
        sys.exit(1)
        
    logger.info(f"Found {len(lgb_checkpoints)} LGBM and {len(deep_checkpoints)} Deep Audio checkpoints.")
    
    device = config.device
    
    # Evaluate Baseline (LightGBM)
    base_probs = []
    if lgb_checkpoints:
        from src.classifier import TabularLGBMClassifier
        import joblib
        
        # Combine handcrafted and crossmodal for baseline
        x_base = np.concatenate([ac_arr, cm_arr]).reshape(1, -1)
        
        for cp_path in lgb_checkpoints:
            lgb_model = TabularLGBMClassifier(config)
            lgb_model.model = joblib.load(str(cp_path))
            prob = lgb_model.predict_proba(x_base)[0]
            base_probs.append(prob)
            
    avg_base_prob = np.mean(base_probs) if base_probs else 0.0

    # Evaluate Deep Audio (WavLM + MLP)
    deep_probs = []
    if deep_checkpoints:
        ac_tensor = torch.tensor(ac_arr, dtype=torch.float32).unsqueeze(0).to(device)
        cm_tensor = torch.tensor(cm_arr, dtype=torch.float32).unsqueeze(0).to(device)
        waveforms = [y]
        texts = [args.transcript]
        
        for cp_path in deep_checkpoints:
            checkpoint = torch.load(str(cp_path), map_location=device, weights_only=False)
            model = ASRQualityClassifier(config, mode="audio_only")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                logits = model(waveforms, texts, ac_tensor, cm_tensor)
                prob = torch.sigmoid(logits).item()
                deep_probs.append(prob)
                
    avg_deep_prob = np.mean(deep_probs) if deep_probs else 0.0
    
    # 4. Ensemble Prediction
    # Use weights found in Phase 5
    if base_probs and deep_probs:
        final_prob = 0.25 * avg_base_prob + 0.75 * avg_deep_prob
    elif deep_probs:
        final_prob = avg_deep_prob
    else:
        final_prob = avg_base_prob
        
    # The optimal threshold found in Phase 5 was 0.78
    best_threshold = 0.78
    
    is_usable = final_prob >= best_threshold
    label = 1 if is_usable else 0
    label_text = "usable" if is_usable else "unusable"
    
    print("\n" + "=" * 50)
    print("                CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"Audio Path    : {args.audio}")
    print(f"Transcript    : {args.transcript}")
    print(f"LGBM Prob     : {avg_base_prob:.4f} (from {len(lgb_checkpoints)} folds)")
    print(f"Deep Prob     : {avg_deep_prob:.4f} (from {len(deep_checkpoints)} folds)")
    print(f"Final Blend   : {final_prob:.4f} (0.25*LGBM + 0.75*Deep)")
    print(f"Threshold     : {best_threshold:.2f}")
    print(f"Prediction    : {label} ({label_text.upper()})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
