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
    """
    Main feature extraction engine: Acoustic, ASR Confidence, and Cross-Modal Metrics.
    """
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Optimized Whisper Pipeline: FP16 + Batching
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL_NAME,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
            batch_size=8
        )
        self.text_proc = TextPreprocessor()

    def estimate_snr(self, y: np.ndarray) -> float:
        """
        Estimates SNR using an adaptive energy threshold.
        """
        if y is None or len(y) == 0: return 0.0
        
        rms = librosa.feature.rms(y=y)[0]
        # Adaptive threshold using a percentile of RMS energy (simulating noise floor)
        noise_floor = np.percentile(rms, 20) 
        active_mask = rms > (noise_floor * 2) # Heuristic for active speech
        
        active_frames = rms[active_mask]
        silent_frames = rms[~active_mask]
        
        if len(silent_frames) == 0 or np.mean(silent_frames) == 0:
            return 50.0
            
        snr = 10 * np.log10(np.mean(active_frames**2) / (np.mean(silent_frames**2) + 1e-10))
        return float(snr)

    def get_silence_ratio(self, y: np.ndarray) -> float:
        """
        Calculates percentage of silence using adaptive thresholding.
        """
        if y is None or len(y) == 0: return 1.0
        
        rms = librosa.feature.rms(y=y)[0]
        # Adaptive silence threshold
        adaptive_threshold = np.percentile(rms, 30)
        silence_ratio = np.sum(rms < adaptive_threshold) / len(rms)
        return float(silence_ratio)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features for the entire dataset.
        """
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            # Construct path: data/audio/data2/{folder}/{file_name}
            # Based on user's structure: file_path contains "folder/file_name"
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.AUDIO_DIR, "data2", file_rel_path)
            
            gt_transcript = str(row['transcript'])
            
            # Check if file exists before processing
            if not os.path.exists(audio_path):
                # Try without data2 just in case structure varies
                alt_path = os.path.join(config.AUDIO_DIR, file_rel_path)
                if os.path.exists(alt_path):
                    audio_path = alt_path
                else:
                    if config.VERBOSE:
                        print(f"Warning: File not found: {audio_path}")
                    continue

            # Load and preprocess audio
            y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
            
            if y is None:
                continue

            # 1. Acoustic Metrics
            snr = self.estimate_snr(y)
            silence_ratio = self.get_silence_ratio(y)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 2. ASR Features & Similarity
            try:
                # Perform ASR
                asr_result = self.asr_pipeline(audio_path, return_timestamps=False)
                hyp_transcript = asr_result["text"]
                
                # Normalize both texts
                clean_gt = self.text_proc.clean_text(gt_transcript)
                clean_hyp = self.text_proc.clean_text(hyp_transcript)
                
                # WER / CER
                current_wer = wer(clean_gt, clean_hyp) if clean_gt else 1.0
                current_cer = cer(clean_gt, clean_hyp) if clean_gt else 1.0
                
                # Length Ratio
                gt_len = len(clean_gt.split())
                length_ratio = gt_len / duration if duration > 0 else 0
                
            except Exception as e:
                print(f"ASR Error on {audio_path}: {e}")
                current_wer, current_cer, length_ratio = 1.0, 1.0, 0.0

            features_list.append({
                "file_name": row['file_name'],
                "snr": snr,
                "silence_ratio": silence_ratio,
                "wer": current_wer,
                "cer": current_cer,
                "length_ratio": length_ratio,
                "duration": duration,
                "label": row['label']
            })
            
        return pd.DataFrame(features_list)
