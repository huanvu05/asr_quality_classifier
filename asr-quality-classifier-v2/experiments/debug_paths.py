import os
import sys

def debug_paths():
    base = "/kaggle/input"
    print(f"Scanning {base} for .wav files...")
    wav_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    
    print(f"Found {len(wav_files)} .wav files.")
    if wav_files:
        print("First 10 files:")
        for f in wav_files[:10]:
            print(f"  {f}")
    else:
        print("CRITICAL: NO .WAV FILES FOUND ANYWHERE IN /kaggle/input")
        print("This means the dataset you added to Kaggle DOES NOT contain the audio files.")

if __name__ == "__main__":
    debug_paths()
