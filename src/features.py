import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from src.config import config
from src.preprocessor import AudioPreprocessor
import joblib

class DeepAudioSequenceExtractor:
    """
    SOTA Extractor: Dùng Wav2Vec2 để lấy chuỗi trạng thái nguyên bản (Sequence of Hidden States).
    Bỏ qua text, tập trung 100% vào Acoustic Anomalies.
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading Wav2Vec2 Encoder ({config.ENCODER_MODEL_NAME}) on {self.device}...")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)
        # Ép Float32 chống lỗi CUDA
        self.model = Wav2Vec2Model.from_pretrained(config.ENCODER_MODEL_NAME).to(self.device).to(torch.float32)
        self.model.eval()

    def get_sequence_embedding(self, audio_path: str) -> np.ndarray:
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None or len(y) == 0:
            return None
            
        try:
            inputs = self.feature_extractor(y, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            input_values = inputs.input_values.to(self.device).to(torch.float32)
            
            with torch.no_grad():
                # Lấy output từ lớp cuối cùng: Shape [1, Sequence_Length, 768]
                outputs = self.model(input_values)
                # Bỏ dimension Batch -> [Sequence_Length, 768]
                hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                # CHUẨN HÓA ĐỘ DÀI (PADDING / TRUNCATING)
                # Mạng Self-Attention cần đầu vào cố định chiều dài
                seq_len = hidden_states.shape[0]
                max_len = config.MAX_SEQ_LENGTH
                
                if seq_len < max_len:
                    # Thiếu thì đắp số 0 vào cuối (Padding)
                    pad_width = max_len - seq_len
                    hidden_states = np.pad(hidden_states, ((0, pad_width), (0, 0)), mode='constant')
                elif seq_len > max_len:
                    # Dư thì cắt bớt (Truncating)
                    hidden_states = hidden_states[:max_len, :]
                    
            return hidden_states # Shape CHUẨN: [400, 768]
            
        except Exception as e:
            # print(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def process_dataset(self, df: pd.DataFrame, output_path: str):
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting SOTA Sequence Embeddings"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.DATA_DIR, "audio", "data2", file_rel_path)
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.DATA_DIR, "audio", "training_audio", file_rel_path)
                if not os.path.exists(audio_path):
                    continue
            
            seq_emb = self.get_sequence_embedding(audio_path)
            if seq_emb is not None:
                data.append({
                    "file_name": row['file_name'],
                    "label": row['label'],
                    "embedding": seq_emb # Ma trận 2D [400, 768]
                })
        
        joblib.dump(data, output_path)
        print(f"Saved {len(data)} sequence embeddings to {output_path}")
        return data
