import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
from src.config import config
from src.preprocessor import AudioPreprocessor
import joblib

class DeepAudioChunkExtractor:
    """
    SOTA Extractor: Slices audio into 2-second chunks, passes each through Whisper,
    then applies Mean & Max Pooling to preserve localized anomalies.
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading Whisper Encoder ({config.ENCODER_MODEL_NAME}) on {self.device}...")
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)
        # Force FP32 to ensure compatibility across all Kaggle GPUs (T4/P100)
        self.model = WhisperModel.from_pretrained(config.ENCODER_MODEL_NAME).encoder.to(self.device).to(torch.float32)
        self.model.eval()

    def get_chunked_embedding(self, audio_path: str) -> np.ndarray:
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None or len(y) == 0:
            return None
            
        try:
            chunk_length_samples = int(config.CHUNK_DURATION_S * sr)
            chunks = [y[i:i + chunk_length_samples] for i in range(0, len(y), chunk_length_samples)]
            
            chunk_embeddings = []
            
            with torch.no_grad():
                for chunk in chunks:
                    # Bỏ qua các chunk quá ngắn (ví dụ < 0.5s ở cuối file)
                    if len(chunk) < sr * 0.5:
                        continue
                        
                    inputs = self.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
                    input_features = inputs.input_features.to(self.device).to(torch.float32)
                    
                    outputs = self.model(input_features)
                    # Shape: [512] for this specific chunk
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    chunk_embeddings.append(emb)
            
            if not chunk_embeddings:
                return None
                
            # [Num_chunks, 512]
            chunk_matrix = np.array(chunk_embeddings)
            
            # 1. Tích hợp tổng thể: Lấy trung bình cộng (Mean Pooling) -> 512D
            mean_pooled = np.mean(chunk_matrix, axis=0)
            
            # 2. Bắt lỗi cục bộ: Lấy giá trị lớn nhất (Max Pooling) -> 512D
            max_pooled = np.max(chunk_matrix, axis=0)
            
            # Ghép lại thành Siêu Vector 1024D
            final_embedding = np.concatenate([mean_pooled, max_pooled])
            
            return final_embedding
            
        except Exception as e:
            # print(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def process_dataset(self, df: pd.DataFrame, output_path: str):
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Chunked Embeddings"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.DATA_DIR, "audio", "data2", file_rel_path)
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.DATA_DIR, "audio", "training_audio", file_rel_path)
                if not os.path.exists(audio_path):
                    continue
            
            emb = self.get_chunked_embedding(audio_path)
            if emb is not None:
                data.append({
                    "file_name": row['file_name'],
                    "label": row['label'],
                    "embedding": emb # Vector 1024D
                })
        
        joblib.dump(data, output_path)
        print(f"Saved {len(data)} chunked embeddings to {output_path}")
        return data
