"""
audio_features.py — Feature extraction from raw audio and transcript text.

Extracts:
1. 37 handcrafted acoustic features (SNR, silence ratio, RMS, spectral metrics, 12 MFCCs).
2. 6 cross-modal features (durations, word/char counts, speaking speed mismatch).
"""

import warnings
import numpy as np
import librosa
from typing import Dict, List, Tuple

from src.config import Config, logger

# Suppress librosa user warnings for clean logs
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def get_acoustic_feature_keys() -> List[str]:
    """Returns the ordered list of 37 acoustic feature keys."""
    keys = [
        "snr",
        "voiced_ratio",
        "clipping_ratio",
        "silence_ratio",
        "speaking_rate",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
    ]
    # 12 MFCC means and 12 MFCC stds
    keys += [f"mfcc{i}_mean" for i in range(1, 13)]
    keys += [f"mfcc{i}_std" for i in range(1, 13)]
    return keys

def get_crossmodal_feature_keys() -> List[str]:
    """Returns the ordered list of 6 crossmodal feature keys."""
    return [
        "char_count",
        "word_count",
        "chars_per_sec",
        "words_per_sec",
        "char_to_word_ratio",
        "duration_mismatch",
    ]

def extract_acoustic_features(
    y: np.ndarray, 
    sr: int, 
    config: Config
) -> Dict[str, float]:
    """
    Extracts exactly 37 acoustic features from loaded audio.
    """
    feats = {}
    n = len(y)
    keys = get_acoustic_feature_keys()
    
    if n == 0:
        return {k: 0.0 for k in keys}
        
    duration = n / sr
    
    # 1. SNR and voiced ratio based on energy thresholds
    frame_length = config.audio.frame_length
    hop_length = config.audio.hop_length
    
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.mean(frames ** 2, axis=0) + 1e-12
    max_energy = frame_energy.max()
    thr = 0.01 * max_energy
    
    s_energy = frame_energy[frame_energy >= thr].mean() if (frame_energy >= thr).any() else frame_energy.mean()
    n_energy = frame_energy[frame_energy < thr].mean() if (frame_energy < thr).any() else 1e-12
    
    feats["snr"] = float(np.clip(10 * np.log10(s_energy / n_energy), -10, 60))
    feats["voiced_ratio"] = float((frame_energy > thr).mean())
    feats["clipping_ratio"] = float(np.mean(np.abs(y) > 0.99))
    
    # 2. Silence ratio (using librosa's split method)
    try:
        intervals = librosa.effects.split(y, top_db=config.audio.silence_top_db)
        voiced_len = sum(e - s for s, e in intervals)
        feats["silence_ratio"] = float(1.0 - voiced_len / n)
    except Exception:
        feats["silence_ratio"] = 0.5  # default if failure
        
    # 3. Speaking rate (transitions of energy threshold)
    trans = np.diff((frame_energy > thr).astype(int))
    feats["speaking_rate"] = float(np.sum(trans > 0) / (duration + 1e-6))
    
    # 4. Spectral features
    S = np.abs(librosa.stft(y, n_fft=config.audio.n_fft, hop_length=hop_length))
    sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    feats["spectral_centroid_mean"] = float(sc.mean())
    feats["spectral_centroid_std"] = float(sc.std())
    feats["spectral_bandwidth_mean"] = float(sb.mean())
    feats["spectral_rolloff_mean"] = float(sro.mean())
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())
    
    # 5. RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    
    # 6. 12 MFCCs (means and stds)
    # Extract 12 MFCCs (index 0 is excluded in some ASR features, but we keep 1 to 12)
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=12, 
        n_fft=config.audio.n_fft, 
        hop_length=hop_length
    )
    for i in range(12):
        feats[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i+1}_std"] = float(mfcc[i].std())
        
    # Check completeness
    for k in keys:
        if k not in feats:
            feats[k] = 0.0
            
    return feats

def extract_crossmodal_features(
    duration: float, 
    transcript: str
) -> Dict[str, float]:
    """
    Extracts exactly 6 cross-modal features linking the audio duration and text.
    """
    transcript = str(transcript) if not pd.isna(transcript) else ""
    words = transcript.strip().split()
    
    char_count = len(transcript)
    word_count = len(words)
    
    chars_per_sec = char_count / (duration + 1e-6)
    words_per_sec = word_count / (duration + 1e-6)
    char_to_word_ratio = char_count / (word_count + 1e-6)
    
    # Expected speaking rate for Vietnamese is roughly 2.5 - 3.5 words/sec.
    # We define duration mismatch as the absolute difference from the expected duration at 3.0 words/sec.
    expected_duration = word_count / 3.0
    duration_mismatch = abs(duration - expected_duration)
    
    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "chars_per_sec": float(chars_per_sec),
        "words_per_sec": float(words_per_sec),
        "char_to_word_ratio": float(char_to_word_ratio),
        "duration_mismatch": float(duration_mismatch),
    }

def process_audio_file(
    file_path: str, 
    transcript: str, 
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an audio file, resamples it, and extracts:
    - 37 acoustic features as a float32 NumPy array.
    - 6 cross-modal features as a float32 NumPy array.
    
    Handles loading errors by returning zero arrays.
    """
    keys_ac = get_acoustic_feature_keys()
    keys_cm = get_crossmodal_feature_keys()
    
    try:
        # Load audio (mono, resampled to config sample rate)
        y, sr = librosa.load(
            file_path, 
            sr=config.audio.sample_rate, 
            mono=True, 
            duration=config.audio.max_duration_sec
        )
        
        # Check duration bounds
        duration = len(y) / sr
        if duration < config.audio.min_duration_sec:
            logger.warning(f"Audio file {file_path} is too short ({duration:.2f}s). Returning zeros.")
            return np.zeros(len(keys_ac), dtype=np.float32), np.zeros(len(keys_cm), dtype=np.float32)

        # Extract acoustic features
        ac_dict = extract_acoustic_features(y, sr, config)
        ac_arr = np.array([ac_dict[k] for k in keys_ac], dtype=np.float32)
        
        # Extract cross-modal features
        cm_dict = extract_crossmodal_features(duration, transcript)
        cm_arr = np.array([cm_dict[k] for k in keys_cm], dtype=np.float32)
        
        return ac_arr, cm_arr
        
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {e}")
        return np.zeros(len(keys_ac), dtype=np.float32), np.zeros(len(keys_cm), dtype=np.float32)
