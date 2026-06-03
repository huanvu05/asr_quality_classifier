import os
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm
from src.config import config
from src.preprocessor import TextPreprocessor, AudioPreprocessor

class FeatureExtractor:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ÉP BUỘC TIẾNG VIỆT để tránh Whisper nhận diện nhầm ngôn ngữ
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL_NAME,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
            batch_size=8
        )
        self.text_proc = TextPreprocessor()

    def get_advanced_acoustic_features(self, y: np.ndarray, sr: int) -> dict:
        """
        Trích xuất các đặc trưng phổ sâu để bắt nhiễu và chất lượng âm thanh.
        """
        if y is None or len(y) == 0:
            return {"snr": 0, "silence_ratio": 1, "spectral_flatness": 1, "mfcc_var": 0}
        
        # 1. SNR (Robust version)
        rms = librosa.feature.rms(y=y)[0]
        noise_floor = np.percentile(rms, 10) + 1e-10
        active_speech = rms[rms > (noise_floor * 3)]
        snr = 10 * np.log10(np.mean(active_speech**2) / (noise_floor**2)) if len(active_speech) > 0 else 0
        
        # 2. Spectral Flatness (Độ phẳng phổ - Cao nghĩa là nhiễu trắng nhiều)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 3. MFCC Variance (Âm thanh tự nhiên có độ biến thiên MFCC đặc trưng)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfccs, axis=1))
        
        # 4. Silence Ratio (Absolute)
        silence_ratio = np.sum(rms < 0.001) / len(rms)
        
        return {
            "snr": float(np.clip(snr, 0, 50)),
            "silence_ratio": float(silence_ratio),
            "spectral_flatness": float(flatness),
            "mfcc_var": float(mfcc_var)
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Deep Features"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.AUDIO_DIR, "data2", file_rel_path)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.AUDIO_DIR, file_rel_path)
            
            if not os.path.exists(audio_path): continue

            y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
            if y is None: continue

            # 1. Acoustic Features
            acoustic = self.get_advanced_acoustic_features(y, sr)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 2. ASR Features (Forced Vietnamese)
            try:
                # Ép task='transcribe' và language='vi'
                asr_out = self.asr_pipeline(audio_path, generate_kwargs={"language": "vi", "task": "transcribe"})
                hyp_transcript = asr_out["text"]
                
                clean_gt = self.text_proc.clean_text(str(row['transcript']))
                clean_hyp = self.text_proc.clean_text(hyp_transcript)
                
                current_wer = wer(clean_gt, clean_hyp) if clean_gt else 1.0
                current_cer = cer(clean_gt, clean_hyp) if clean_gt else 1.0
                
                # Word Count Consistency
                gt_words = len(clean_gt.split()) + 1
                hyp_words = len(clean_hyp.split()) + 1
                word_ratio = hyp_words / gt_words
            except:
                current_wer, current_cer, word_ratio = 2.0, 2.0, 0.0

            feat = {
                "file_name": row['file_name'],
                "duration": duration,
                "wer": np.clip(current_wer, 0, 3),
                "cer": np.clip(current_cer, 0, 3),
                "word_ratio": np.clip(word_ratio, 0, 3),
                "label": row['label']
            }
            feat.update(acoustic)
            features_list.append(feat)
            
        return pd.DataFrame(features_list)
