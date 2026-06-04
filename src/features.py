import os
import torch
import torch.nn as nn
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
    Bỏ qua text, tập trung 100% vào Acoustic Anomalies. Hỗ trợ xử lý Batch đa GPU (T4x2).
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading Wav2Vec2 Encoder ({config.ENCODER_MODEL_NAME}) on {self.device}...")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)
        # Ép Float32 chống lỗi CUDA
        model = Wav2Vec2Model.from_pretrained(config.ENCODER_MODEL_NAME).to(self.device).to(torch.float32)
        
        # Hỗ trợ T4x2 GPU (Multi-GPU)
        if torch.cuda.device_count() > 1 and self.device == "cuda":
            print(f"🚀 Kích hoạt trích xuất song song trên {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        self.model.eval()

    def get_sequence_embedding(self, audio_path: str) -> np.ndarray:
        # Giữ lại hàm cũ cho Inference đơn lẻ
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None or len(y) == 0:
            return None
            
        try:
            inputs = self.feature_extractor(y, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            input_values = inputs.input_values.to(self.device).to(torch.float32)
            
            with torch.no_grad():
                outputs = self.model(input_values)
                # Tương thích với DataParallel: outputs có thể là tuple nếu bọc module khác, nhưng với HuggingFace model bọc DataParallel, 
                # nó trả về BaseModelOutput. Truy cập last_hidden_state
                last_hidden = outputs.last_hidden_state
                hidden_states = last_hidden.squeeze(0).cpu().numpy()
                
                seq_len = hidden_states.shape[0]
                max_len = config.MAX_SEQ_LENGTH
                
                if seq_len < max_len:
                    pad_width = max_len - seq_len
                    hidden_states = np.pad(hidden_states, ((0, pad_width), (0, 0)), mode='constant')
                elif seq_len > max_len:
                    hidden_states = hidden_states[:max_len, :]
                    
            return hidden_states 
            
        except Exception as e:
            return None

    def process_dataset(self, df: pd.DataFrame, output_path: str, batch_size=8):
        """Xử lý theo lô (Batch) để tối đa hóa sức mạnh T4x2"""
        data = []
        
        # Thu thập các đường dẫn hợp lệ
        valid_paths = []
        valid_rows = []
        
        print("[*] Lọc các file audio tồn tại...")
        for idx, row in df.iterrows():
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.DATA_DIR, "audio", "data2", file_rel_path)
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.DATA_DIR, "audio", "training_audio", file_rel_path)
                if not os.path.exists(audio_path):
                    continue
            valid_paths.append(audio_path)
            valid_rows.append(row)
            
        print(f"[*] Tiến hành trích xuất {len(valid_paths)} file bằng Wav2Vec2 (Batch Size = {batch_size})")
        
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="Extracting SOTA Embeddings"):
            batch_paths = valid_paths[i:i+batch_size]
            batch_rows = valid_rows[i:i+batch_size]
            
            # Đọc audio
            batch_audios = []
            valid_batch_rows = []
            
            for path, row in zip(batch_paths, batch_rows):
                y, sr = AudioPreprocessor.process_audio(path, target_sr=config.SAMPLE_RATE)
                if y is not None and len(y) > 0:
                    batch_audios.append(y)
                    valid_batch_rows.append(row)
            
            if not batch_audios:
                continue
                
            try:
                # Padding trực tiếp trong feature_extractor
                inputs = self.feature_extractor(
                    batch_audios, 
                    sampling_rate=config.SAMPLE_RATE, 
                    return_tensors="pt", 
                    padding=True
                )
                input_values = inputs.input_values.to(self.device).to(torch.float32)
                
                with torch.no_grad():
                    outputs = self.model(input_values)
                    hidden_states_batch = outputs.last_hidden_state.cpu().numpy()
                    
                # Xử lý cắt/đệm cho từng mẫu trong batch
                for j, hidden_states in enumerate(hidden_states_batch):
                    seq_len = hidden_states.shape[0]
                    max_len = config.MAX_SEQ_LENGTH
                    
                    if seq_len < max_len:
                        pad_width = max_len - seq_len
                        hidden_states = np.pad(hidden_states, ((0, pad_width), (0, 0)), mode='constant')
                    elif seq_len > max_len:
                        hidden_states = hidden_states[:max_len, :]
                        
                    data.append({
                        "file_name": valid_batch_rows[j]['file_name'],
                        "label": valid_batch_rows[j]['label'],
                        "embedding": hidden_states
                    })
                    
            except Exception as e:
                print(f"Lỗi batch {i}: {e}")
                
        joblib.dump(data, output_path)
        print(f"Saved {len(data)} sequence embeddings to {output_path}")
        return data
