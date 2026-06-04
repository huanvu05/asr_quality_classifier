import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import config
from src.features import DeepAudioSequenceExtractor
from src.model import AttentionHeadClassifier

def load_sota_model(model_path: str):
    """Loads the trained PyTorch Attention Model."""
    model = AttentionHeadClassifier().to(config.DEVICE)
    
    # Handle DataParallel state dict if it was saved with multiple GPUs
    state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def evaluate_on_test_set(test_csv_path: str, model_path: str, threshold: float):
    print(f"Loading Test Data from: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)
    
    # 1. Check if we need to extract embeddings
    extractor = DeepAudioSequenceExtractor()
    
    y_true = []
    y_probs = []
    
    print("\nStarting Inference on Test Set...")
    with torch.no_grad():
        for idx, row in df_test.iterrows():
            file_rel_path = str(row['file_path'])
            # Lấy nhãn thực tế
            label = 1 if str(row['label']) == '1' else 0
            
            # Xử lý đường dẫn
            audio_path = os.path.join(config.AUDIO_DIR, file_rel_path)
            if not os.path.exists(audio_path): 
                audio_path = os.path.join(config.DATA_DIR, "audio", "training_audio", file_rel_path)
                if not os.path.exists(audio_path):
                    print(f"Warning: File not found {audio_path}")
                    continue
            
            # Trích xuất Wav2Vec2 Sequence Embedding On-the-fly
            seq_emb = extractor.get_sequence_embedding(audio_path)
            if seq_emb is None:
                continue
                
            # Đưa vào model dự đoán
            # Shape cần đưa vào: [1, seq_length, embedding_dim]
            input_tensor = torch.tensor(seq_emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            logit = load_sota_model(model_path)(input_tensor)
            prob = torch.sigmoid(logit).item()
            
            y_probs.append(prob)
            y_true.append(label)

    if len(y_true) == 0:
        print("No valid audio files found to test.")
        return

    # 2. Đánh giá kết quả
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    print("\n" + "="*40)
    print("TEST SET EVALUATION REPORT")
    print("="*40)
    print(f"Tested on {len(y_true)} samples.")
    print(f"Used Threshold: {threshold:.4f}")
    print(classification_report(y_true, y_pred))
    
    auc = roc_auc_score(y_true, y_probs)
    print(f"ROC-AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Unusable(0)', 'Usable(1)'], yticklabels=['Unusable(0)', 'Usable(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Test Set Confusion Matrix\nAUC = {auc:.2f}')
    plt.savefig('test_confusion_matrix.png')
    print("Saved confusion matrix to test_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=str, required=True, help="Path to unseen test CSV")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .pth model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()
    
    evaluate_on_test_set(args.test_csv, args.model_path, args.threshold)
