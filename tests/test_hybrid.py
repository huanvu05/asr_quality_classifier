import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np
import pandas as pd
import joblib

# 1. Mock dependencies for local testing
mock_modules = [
    'librosa', 'xgboost', 'seaborn', 'matplotlib', 'matplotlib.pyplot', 'tqdm'
]
for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Setup explicit librosa mock returns
import librosa
librosa.load.return_value = (np.random.rand(16000), 16000)
librosa.get_duration.return_value = 5.0
librosa.effects.split.return_value = [[0, 8000], [10000, 15000]]
librosa.feature.rms.return_value = np.array([[0.1, 0.5, 0.2, 0.05, 0.6]])
librosa.feature.spectral_rolloff.return_value = np.array([[2000, 2500, 1500, 2200, 1800]])

# 2. Fix TQDM for tests
import tqdm
tqdm.tqdm.side_effect = lambda x, **kwargs: x

# Add root path
sys.path.append(os.getcwd())

# 3. Create dummy data for testing
os.makedirs("data/test_audio", exist_ok=True)
dummy_csv_path = "data/test_labels.csv"
dummy_pkl_path = "data/test_embeddings.pkl"

# Dummy CSV
df = pd.DataFrame({
    'file_path': ['folder1/audio1.wav', 'folder2/audio2.wav', 'audio3.wav'],
    'file_name': ['audio1.wav', 'audio2.wav', 'audio3.wav'],
    'label': [1, 0, 1]
})
df.to_csv(dummy_csv_path, index=False)

# Dummy PKL (Simulating Whisper Base 512D)
dummy_embeddings = [
    {'file_name': 'audio1.wav', 'embedding': np.random.rand(512).astype(np.float32), 'label': 1},
    {'file_name': 'audio2.wav', 'embedding': np.random.rand(512).astype(np.float32), 'label': 0},
    {'file_name': 'audio3.wav', 'embedding': np.random.rand(512).astype(np.float32), 'label': 1}
]
joblib.dump(dummy_embeddings, dummy_pkl_path)

# Create dummy physical files so os.path.exists passes
os.makedirs("data/test_audio/folder1", exist_ok=True)
os.makedirs("data/test_audio/folder2", exist_ok=True)
open("data/test_audio/folder1/audio1.wav", "w").close()
open("data/test_audio/folder2/audio2.wav", "w").close()
open("data/test_audio/audio3.wav", "w").close()


class TestHybridPipelineLogic(unittest.TestCase):
    def test_feature_extraction_dims(self):
        """Test if Component A (512) + Component B (10) = 522 dimensions."""
        from train_hybrid import extract_handcrafted_features, build_hybrid_dataset
        
        # Test Handcrafted features (Component B)
        handcrafted = extract_handcrafted_features("data/test_audio/folder1/audio1.wav")
        self.assertEqual(len(handcrafted), 10)
        self.assertEqual(handcrafted[0], 5.0) # Duration should be 5.0 as mocked
        
        # Test Full Matrix Build
        X, y, files = build_hybrid_dataset(dummy_csv_path, dummy_pkl_path, "data/test_audio")
        
        # We have 3 files
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(y.shape[0], 3)
        
        # Dimension should be 512 + 10 = 522
        self.assertEqual(X.shape[1], 522)
        
        # Ensure correct alignment (Order should match despite how CSV is read)
        self.assertEqual(files[0], 'audio1.wav')
        self.assertEqual(files[1], 'audio2.wav')
        self.assertEqual(files[2], 'audio3.wav')
        
        print("\n[OK] Hybrid Vector Concatenation and Alignment logic passed completely.")

if __name__ == "__main__":
    unittest.main()
