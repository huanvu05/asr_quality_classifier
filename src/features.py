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

class DeepAudioExtractorSOTA:
    """
    State-of-the-Art Feature Extractor using Wav2Vec2-Large-XLSR-53.
    Extracts FULL sequence representations for Attention processing, not just a mean vector.
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading Wav2Vec2 Encoder ({config.ENCODER_MODEL_NAME}) on {self.device}...")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)
        self.model = Wav2Vec2Model.from_pretrained(config.ENCODER_MODEL_NAME).to(self.device)
        self.model.eval()

    def get_sequence_embedding(self, audio_path: str) -> np.ndarray:
        y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
        if y is None or len(y) == 0:
            return None
            
        try:
            inputs = self.feature_extractor(y, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                # Extract latent representation from the final layer
                # Shape: [1, sequence_length, 1024]
                outputs = self.model(input_values)
                hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                # We do NOT mean-pool here. We keep the sequence to feed into the Attention Head later.
                # Just pad or truncate to MAX_SEQ_LENGTH to standardize input sizes
                seq_len = hidden_states.shape[0]
                max_len = config.MAX_SEQ_LENGTH
                
                if seq_len < max_len:
                    pad_width = max_len - seq_len
                    # Pad with zeros
                    hidden_states = np.pad(hidden_states, ((0, pad_width), (0, 0)), mode='constant')
                elif seq_len > max_len:
                    # Truncate
                    hidden_states = hidden_states[:max_len, :]
                    
            return hidden_states
        except Exception as e:
            # print(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def process_dataset(self, df: pd.DataFrame, output_path: str):
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting SOTA Sequence Embeddings"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.AUDIO_DIR, file_rel_path)
            
            if not os.path.exists(audio_path): 
                # Try fallback just in case
                audio_path = os.path.join(config.DATA_DIR, "audio", "training_audio", file_rel_path)
                if not os.path.exists(audio_path):
                    continue
            
            seq_emb = self.get_sequence_embedding(audio_path)
            if seq_emb is not None:
                data.append({
                    "file_name": row['file_name'],
                    "label": row['label'],
                    "embedding": seq_emb # Shape: [MAX_SEQ_LENGTH, 1024]
                })
        
        joblib.dump(data, output_path)
        print(f"Saved {len(data)} sequence embeddings to {output_path}")
        return data
