import re
import string
import librosa
import soundfile as sf
from num2words import num2words
from src.config import config

class TextPreprocessor:
    """
    Handles Vietnamese text normalization and number conversion.
    """
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """
        Converts numeric digits into Vietnamese words.
        Example: "99" -> "chín mươi chín"
        """
        def replace_num(match):
            return num2words(match.group(), lang='vi')
        
        return re.sub(r'\d+', replace_num, text)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Standardizes text: lowercase, remove punctuation, strip.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Normalize numbers
        text = TextPreprocessor.normalize_numbers(text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class AudioPreprocessor:
    """
    Standardizes audio to target sample rate and mono channel.
    """
    @staticmethod
    def process_audio(file_path: str, target_sr: int = 16000) -> tuple:
        """
        Loads audio, resamples, and converts to mono.
        Returns: (y, sr)
        """
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"Error processing audio {file_path}: {e}")
            return None, None
