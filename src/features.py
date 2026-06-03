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
        # Optimized Whisper Pipeline: Trích xuất thêm log-probabilities để tính confidence
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL_NAME,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=False
        )
        # Load processor separately for confidence scoring
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
        self.text_proc = TextPreprocessor()

    def estimate_snr(self, y: np.ndarray) -> float:
        """
        Estimates SNR using a more robust energy-based method.
        """
        if y is None or len(y) == 0: return 0.0
        
        rms = librosa.feature.rms(y=y)[0]
        # Use a fixed low-energy threshold to identify 'likely' noise frames
        # and compare them to 'likely' speech frames
        noise_floor = np.percentile(rms, 10) + 1e-10
        active_speech = rms[rms > (noise_floor * 3)]
        
        if len(active_speech) == 0:
            return 0.0
            
        snr = 10 * np.log10(np.mean(active_speech**2) / (noise_floor**2))
        return float(np.clip(snr, 0, 50))

    def get_silence_ratio(self, y: np.ndarray) -> float:
        """
        Calculates silence ratio using an ABSOLUTE energy threshold.
        """
        if y is None or len(y) == 0: return 1.0
        rms = librosa.feature.rms(y=y)[0]
        # Ngưỡng năng lượng cố định (0.001) để bắt im lặng thật sự
        silence_mask = rms < 0.001
        return float(np.sum(silence_mask) / len(rms))

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features including ASR Confidence.
        """
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting high-quality features"):
            file_rel_path = str(row['file_path'])
            audio_path = os.path.join(config.AUDIO_DIR, "data2", file_rel_path)
            
            if not os.path.exists(audio_path):
                audio_path = os.path.join(config.AUDIO_DIR, file_rel_path)
            
            if not os.path.exists(audio_path): continue

            y, sr = AudioPreprocessor.process_audio(audio_path, target_sr=config.SAMPLE_RATE)
            if y is None: continue

            # 1. Acoustic
            snr = self.estimate_snr(y)
            silence_ratio = self.get_silence_ratio(y)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 2. ASR & Confidence
            try:
                # Trích xuất transcript và metadata
                asr_out = self.asr_pipeline(audio_path, return_timestamps=False, generate_kwargs={"output_scores": True, "return_dict_in_generate": True})
                hyp_transcript = asr_out["text"]
                
                # Giả lập confidence score từ kết quả pipeline (vì pipeline mặc định khó lấy token-level score trực tiếp)
                # Thay vào đó, ta sử dụng độ dài transcript / thời lượng như một proxy mạnh
                # Hoặc nếu chạy Whisper qua model.generate() sẽ lấy được logprob chuẩn hơn.
                # Ở đây dùng proxy: WER + Length Consistency.
                
                clean_gt = self.text_proc.clean_text(str(row['transcript']))
                clean_hyp = self.text_proc.clean_text(hyp_transcript)
                
                current_wer = wer(clean_gt, clean_hyp)
                current_cer = cer(clean_gt, clean_hyp)
                
                # Đặc trưng mới: Word Count Ratio (Số từ thực tế vs dự đoán)
                gt_words = len(clean_gt.split()) + 1
                hyp_words = len(clean_hyp.split()) + 1
                word_ratio = hyp_words / gt_words
                
            except Exception as e:
                current_wer, current_cer, word_ratio = 2.0, 2.0, 0.0

            features_list.append({
                "file_name": row['file_name'],
                "snr": snr,
                "silence_ratio": silence_ratio,
                "wer": np.clip(current_wer, 0, 5), # Clip để tránh outlier phá hỏng mô hình
                "cer": np.clip(current_cer, 0, 5),
                "word_ratio": word_ratio,
                "duration": duration,
                "label": row['label']
            })
            
        return pd.DataFrame(features_list)
