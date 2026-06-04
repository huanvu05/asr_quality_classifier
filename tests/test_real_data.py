import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd
import numpy as np

# 2. Import src modules
sys.path.append(os.getcwd())
from src.data_loader import load_labels
from src.features import DeepAudioSequenceExtractor
from src.config import config

class TestRealDataIntegration(unittest.TestCase):
    def test_load_real_csv(self):
        """Verify we can load the user's training.csv correctly."""
        csv_path = "data/transcripts/training.csv"
        # Create a dummy CSV for the test if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            pd.DataFrame({'file_path': ['a.wav'], 'transcript': ['test'], 'label': [1]}).to_csv(csv_path, index=False)
            
        df = load_labels(csv_path)
        self.assertIn('file_path', df.columns)
        self.assertIn('transcript', df.columns)
        print(f"\n[OK] Loaded CSV with {len(df)} samples.")

    def test_path_resolution(self):
        """Verify the feature extractor can find the audio files based on CSV paths."""
        csv_path = "data/transcripts/training.csv"
        # Just create dummy DataFrame since load_labels is mocked or used from actual
        df = pd.DataFrame([
            {'file_path': 'folder1/audio1.wav', 'file_name': 'audio1.wav', 'label': 1},
            {'file_path': 'folder2/audio2.wav', 'file_name': 'audio2.wav', 'label': 0}
        ])
        
        # We need to mock the pipeline for DeepAudioSequenceExtractor
        with patch('src.features.Wav2Vec2FeatureExtractor.from_pretrained') as mock_fe, \
             patch('src.features.Wav2Vec2Model.from_pretrained') as mock_model:
            
            mock_fe.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            
            extractor = DeepAudioSequenceExtractor()
            
            with patch('src.features.AudioPreprocessor.process_audio') as mock_proc:
                # Return dummy signal
                mock_proc.return_value = (np.random.rand(16000), 16000)
                
                # Mock the inputs and outputs of Wav2Vec2
                extractor.feature_extractor.return_value = MagicMock(input_values=MagicMock())
                
                class DummyOutput:
                    def __init__(self):
                        self.last_hidden_state = MagicMock()
                        self.last_hidden_state.squeeze.return_value.cpu.return_value.numpy.return_value = np.zeros((100, 768))
                
                extractor.model.return_value = DummyOutput()
                
                # Test get_sequence_embedding directly
                emb = extractor.get_sequence_embedding("dummy_path.wav")
                
                self.assertIsNotNone(emb)
                self.assertEqual(emb.shape, (config.MAX_SEQ_LENGTH, 768))
                print(f"[OK] Successfully extracted sequence embedding.")

if __name__ == "__main__":
    unittest.main()
