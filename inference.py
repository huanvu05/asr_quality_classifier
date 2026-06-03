import os
import argparse
import joblib
import pandas as pd
import librosa
from src.config import config
from src.preprocessor import TextPreprocessor, AudioPreprocessor
from src.features import FeatureExtractor

class InferenceEngine:
    def __init__(self, pipeline_path: str):
        print(f"Loading pipeline from {pipeline_path}...")
        artifacts = joblib.load(pipeline_path)
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.threshold = artifacts["threshold"]
        self.extractor = FeatureExtractor()
        self.text_proc = TextPreprocessor()

    def predict(self, audio_path: str, transcript: str):
        # 1. Feature Extraction (Single Sample)
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None:
            return {"error": "Could not process audio"}

        snr = self.extractor.estimate_snr(y)
        silence_ratio = self.extractor.get_silence_ratio(y)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # ASR & Similarity (Mocking parts for speed in single inference if needed)
        # In real scenario, we use the same extractor logic
        # For simplicity, we wrap it in a dataframe
        mock_df = pd.DataFrame([{
            "file_name": os.path.basename(audio_path),
            "transcript": transcript,
            "label": 0 # dummy
        }])
        
        features_df = self.extractor.extract_features(mock_df)
        X = features_df.drop(columns=["file_name", "label"])
        
        # 2. Scaling
        X_scaled = self.scaler.transform(X)
        
        # 3. Inference
        prob = self.model.predict_proba(X_scaled)[0, 1]
        label = 1 if prob >= self.threshold else 0
        
        return {
            "prediction": "Usable (1)" if label == 1 else "Unusable (0)",
            "probability": float(prob),
            "threshold": self.threshold,
            "metrics": features_df.iloc[0].to_dict()
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Quality Classifier Inference")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to pipeline_artifacts.pkl")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--transcript", type=str, required=True, help="Transcript text")
    
    args = parser.parse_args()
    
    engine = InferenceEngine(args.pipeline)
    result = engine.predict(args.audio, args.transcript)
    
    print("\nInference Result:")
    import json
    print(json.dumps(result, indent=4))
