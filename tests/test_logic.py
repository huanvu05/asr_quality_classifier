import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# 1. Create mocks for all heavy dependencies
mock_modules = [
    'librosa', 'transformers', 'jiwer', 'azure', 'azure.storage', 
    'azure.storage.blob', 'lightgbm', 'xgboost', 'seaborn', 'matplotlib', 
    'matplotlib.pyplot', 'soundfile', 'num2words', 'torch', 'pandas', 'sklearn', 
    'sklearn.model_selection', 'sklearn.metrics', 'sklearn.preprocessing', 'joblib', 'tqdm'
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# 2. Import src modules AFTER mocking
sys.path.append(os.getcwd())
from src.preprocessor import TextPreprocessor
from src.evaluator import Evaluator
from src.config import config

class TestASRQualityLogic(unittest.TestCase):
    def test_text_normalization(self):
        """Tests Vietnamese text cleaning and number conversion."""
        # Manually mock num2words inside the test to be safe
        with patch('num2words.num2words') as mock_n2w:
            mock_n2w.side_effect = lambda x, lang: "một trăm hai mươi ba" if str(x) == "123" else str(x)
            proc = TextPreprocessor()
            test_cases = [
                ("Chào 123", "chào một trăm hai mươi ba"),
                ("Test... câu! hỏi?", "test câu hỏi"),
                ("  Nhiều   khoảng  trống  ", "nhiều khoảng trống")
            ]
            for input_text, expected in test_cases:
                self.assertEqual(proc.clean_text(input_text), expected)

    def test_threshold_optimization(self):
        """Tests if the threshold optimizer finds the best F1."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 1])
        y_probs = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        
        # Patch the f1_score that was imported into src.evaluator
        with patch('src.evaluator.f1_score') as mock_f1:
            mock_f1.side_effect = lambda yt, yp, average=None: 0.9 if np.array_equal(yp, (y_probs >= 0.5).astype(int)) else 0.5
            
            best_thresh = evaluator.optimize_threshold(y_true, y_probs)
            self.assertGreater(best_thresh, 0.4)
            self.assertLessEqual(best_thresh, 0.6)

    def test_config_initialization(self):
        """Tests if config loads defaults correctly."""
        self.assertEqual(config.SAMPLE_RATE, 16000)
        self.assertEqual(config.SEED, 42)

if __name__ == "__main__":
    unittest.main()
