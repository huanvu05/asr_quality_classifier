import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd
import numpy as np

# 1. Create mocks for all heavy dependencies
mock_modules = [
    'librosa', 'transformers', 'jiwer', 'azure', 'azure.storage', 
    'azure.storage.blob', 'lightgbm', 'xgboost', 'seaborn', 'matplotlib', 
    'matplotlib.pyplot', 'soundfile', 'num2words', 'torch', 'tqdm', 'sklearn', 
    'sklearn.preprocessing', 'sklearn.model_selection', 'sklearn.metrics', 'joblib'
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Fix tqdm to be an identity function so it's iterable
import tqdm
tqdm.tqdm.side_effect = lambda x, **kwargs: x

# Mock librosa.get_duration and feature.rms
import librosa
librosa.get_duration.return_value = 10.0
librosa.feature.rms.return_value = [np.random.rand(100)]

# Mock transformers pipeline
import transformers
mock_pipe = MagicMock()
mock_pipe.return_value = {"text": "vâng em chào anh ạ"}
transformers.pipeline.return_value = mock_pipe

# 2. Import src modules
sys.path.append(os.getcwd())
from src.data_loader import load_labels
from src.features import FeatureExtractor
from src.config import config

class TestRealDataIntegration(unittest.TestCase):
    def test_load_real_csv(self):
        """Verify we can load the user's training.csv correctly."""
        csv_path = "data/transcripts/training.csv"
        df = load_labels(csv_path)
        self.assertIn('file_path', df.columns)
        self.assertIn('transcript', df.columns)
        print(f"\n[OK] Loaded CSV with {len(df)} samples.")

    def test_path_resolution(self):
        """Verify the feature extractor can find the audio files based on CSV paths."""
        csv_path = "data/transcripts/training.csv"
        df = load_labels(csv_path).head(5) # Test first 5
        
        extractor = FeatureExtractor()
        
        # We need to mock AudioPreprocessor.process_audio to return a valid dummy signal
        # instead of actually trying to load a file with a mocked librosa
        with patch('src.features.AudioPreprocessor.process_audio') as mock_proc:
            mock_proc.return_value = (np.random.rand(16000), 16000)
            
            # This will trigger the path resolution logic in features.py
            features_df = extractor.extract_features(df)
            
            self.assertEqual(len(features_df), 5)
            print(f"[OK] Successfully resolved and 'processed' 5 samples.")
            print("Feature columns:", features_df.columns.tolist())

if __name__ == "__main__":
    unittest.main()
