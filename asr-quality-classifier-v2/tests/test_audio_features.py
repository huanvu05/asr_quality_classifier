"""
test_audio_features.py — Tests for feature extraction.
"""

import pytest
import numpy as np
from src.audio_features import (
    extract_acoustic_features,
    extract_crossmodal_features,
    process_audio_file,
    get_acoustic_feature_keys,
    get_crossmodal_feature_keys
)

def test_feature_keys():
    ac_keys = get_acoustic_feature_keys()
    cm_keys = get_crossmodal_feature_keys()
    
    assert len(ac_keys) == 37
    assert len(cm_keys) == 6
    assert "snr" in ac_keys
    assert "silence_ratio" in ac_keys
    assert "char_count" in cm_keys
    assert "duration_mismatch" in cm_keys

def test_extract_acoustic_features(test_config):
    # Create 1 second of dummy audio (16000 samples)
    y = np.random.normal(0.0, 0.1, 16000)
    sr = 16000
    
    feats = extract_acoustic_features(y, sr, test_config)
    assert isinstance(feats, dict)
    assert len(feats) == 37
    for k in get_acoustic_feature_keys():
        assert k in feats
        assert isinstance(feats[k], float)

def test_extract_crossmodal_features():
    duration = 5.0
    transcript = "xin chào Việt Nam tôi là robot thông minh"  # 8 words, 40 characters (incl spaces)
    
    feats = extract_crossmodal_features(duration, transcript)
    assert isinstance(feats, dict)
    assert len(feats) == 6
    assert feats["char_count"] == 40
    assert feats["word_count"] == 8
    assert np.isclose(feats["chars_per_sec"], 40 / 5.0)
    assert np.isclose(feats["words_per_sec"], 8 / 5.0)
    assert np.isclose(feats["char_to_word_ratio"], 40 / 8.0)
    # Expected duration at 3.0 words/sec is 8 / 3.0 = 2.667 seconds.
    # Mismatch = abs(5.0 - 2.667) = 2.333
    assert np.isclose(feats["duration_mismatch"], abs(5.0 - 8 / 3.0))

def test_process_audio_file(tmp_path, test_config):
    # Create dummy file
    file_path = tmp_path / "dummy.wav"
    import soundfile as sf
    # 1 second wav
    sf.write(str(file_path), np.random.normal(0.0, 0.1, 16000), 16000)
    
    ac_arr, cm_arr = process_audio_file(str(file_path), "xin chào", test_config)
    assert isinstance(ac_arr, np.ndarray)
    assert isinstance(cm_arr, np.ndarray)
    assert ac_arr.shape == (37,)
    assert cm_arr.shape == (6,)
    
    # Test error handling (invalid path)
    err_ac, err_cm = process_audio_file("invalid_path.wav", "xin chào", test_config)
    assert (err_ac == 0.0).all()
    assert (err_cm == 0.0).all()
