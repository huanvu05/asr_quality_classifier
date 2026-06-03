import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
from src.config import config
from src.preprocessor import AudioPreprocessor

class DeepAudioExtractor:
    """
    Extracts deep acoustic latent representations (Embeddings) using Whisper's Encoder.
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading Whisper Encoder ({config.ENCODER_MODEL_NAME}) on {self.device}...")
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)
        # Chỉ load phần Encoder để chạy nhanh và tập trung vào âm thanh
        self.model = WhisperModel.from_pretrained(config.ENCODER_MODEL_NAME).encoder.to(self.device)
        self.model.eval()

    def get_embedding(self, audio_path: str) -> np.ndarray:
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None or len(y) == 0:
            return None
            
        try:
            # 1. Chuyển audio thành Log-Mel Spectrogram 80 dải tần (chuẩn đầu vào của Whisper)
            inputs = self.feature_extractor(y, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            
            # 2. Đưa qua Encoder
            with torch.no_grad():
                # Shape: [1, sequence_length, hidden_dim (512)]
                outputs = self.model(input_features)
                
            # 3. Pooling: Tính trung bình theo chiều thời gian để ra 1 vector duy nhất đại diện cho cả file
            # Shape: [512]
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def process_dataset(self, df: pd.DataFrame, output_path: str):
        """
        Quét qua toàn bộ dataset và lưu embedding ra file Parquet hoặc Pickle.
        Dùng Pickle (joblib) lưu list các numpy array sẽ bảo toàn độ chính xác tốt hơn CSV.
        """
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Deep Audio Embeddings"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.AUDIO_DIR, "data2", file_rel_path)
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.AUDIO_DIR, file_rel_path)
                if not os.path.exists(audio_path): 
                    continue
            
            emb = self.get_embedding(audio_path)
            if emb is not None:
                data.append({
                    "file_name": row['file_name'],
                    "label": row['label'],
                    "embedding": emb # Lưu trữ vector dạng numpy object
                })
        
        # Lưu thành file nén
        import joblib
        joblib.dump(data, output_path)
        print(f"Saved {len(data)} embeddings to {output_path}")
        return data
