# ASR Data Quality Classifier

A modular, production-ready pipeline for classifying ASR data quality into Usable (1) and Unusable (0).

## 🚀 Features
- **Acoustic Metrics**: SNR estimation, Silence Ratio.
- **ASR Similarity**: WER/CER calculation using Whisper-tiny.
- **Cost-Sensitive Learning**: Handled via LightGBM `scale_pos_weight`.
- **Threshold Optimization**: Maximizes Macro F1-score.
- **MLOps Ready**: Automatic artifact packaging and Azure Blob Storage integration.

## 🛠 Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export AZURE_SAS_TOKEN="your_token"
   export AZURE_STORAGE_URL="https://your_account.blob.core.windows.net"
   ```

## 📈 Training
Run the end-to-end training pipeline:
```bash
python main.py --labels-path data/labels.csv --upload
```

## 🔍 Inference
Run inference on a single audio file:
```bash
python inference.py --pipeline models/run_timestamp_pipeline.pkl --audio path/to/audio.wav --transcript "text transcript"
```

## 📂 Project Structure
- `src/`: Core modules (config, loader, processor, features, model, evaluator, register).
- `data/`: Local cache for audio and transcripts.
- `models/`: Local storage for trained artifacts.
- `main.py`: Training orchestrator.
- `inference.py`: Single-sample predictor.
