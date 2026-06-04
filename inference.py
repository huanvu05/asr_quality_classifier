import os
import argparse
import joblib
import json
import torch
import pandas as pd
import librosa
from src.config import config
from src.preprocessor import TextPreprocessor, AudioPreprocessor
from src.features import DeepAudioSequenceExtractor
from src.model import AttentionHeadClassifier

class InferenceEngine:
    def __init__(self, model_path: str, metadata_path: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Determine threshold from metadata
        if "metrics" in metadata and "optimal_threshold" in metadata["metrics"]:
            self.threshold = metadata["metrics"]["optimal_threshold"]
        else:
            self.threshold = 0.5
            
        print(f"Loading model from {model_path}...")
        self.device = config.DEVICE
        self.model = AttentionHeadClassifier().to(self.device)
        
        # Load weights, handling potential DataParallel prefix
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        self.extractor = DeepAudioSequenceExtractor()
        self.text_proc = TextPreprocessor()

    def predict(self, audio_path: str, transcript: str):
        # 1. Feature Extraction (Single Sample)
        seq_emb = self.extractor.get_sequence_embedding(audio_path)
        if seq_emb is None:
            return {"error": "Could not process audio"}

        # Normalize text just for output info, if needed
        clean_text = self.text_proc.clean_text(transcript)
        
        # 2. Inference
        input_tensor = torch.tensor(seq_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(input_tensor)
            prob = torch.sigmoid(logit).item()
            
        label = 1 if prob >= self.threshold else 0
        
        return {
            "prediction": "Usable (1)" if label == 1 else "Unusable (0)",
            "probability": float(prob),
            "threshold": self.threshold,
            "metrics": {
                "transcript_normalized": clean_text
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Quality Classifier Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_sota_model.pth")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata.json containing threshold")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--transcript", type=str, required=True, help="Transcript text")
    
    args = parser.parse_args()
    
    engine = InferenceEngine(args.model_path, args.metadata_path)
    result = engine.predict(args.audio, args.transcript)
    
    print("\nInference Result:")
    print(json.dumps(result, indent=4))
