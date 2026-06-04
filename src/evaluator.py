import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config
from src.model import AudioDataset, AudioDNN
import copy

class Evaluator:
    def __init__(self):
        self.device = config.DEVICE

    def optimize_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            score = f1_score(y_true, y_pred, average='macro')
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        
        return float(best_threshold)

    def evaluate_results(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float):
        y_pred = (y_probs >= threshold).astype(int)
        
        print("\n" + "="*40)
        print("FINAL EVALUATION REPORT (SINGLE FOLD)")
        print("="*40)
        print(f"Optimal Threshold: {threshold:.4f}")
        print(classification_report(y_true, y_pred))
        
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC: {auc:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Validation Set)')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def train_model(self, model, train_loader, val_loader):
        # Tận dụng Multi-GPU (T4 x2)
        if torch.cuda.device_count() > 1 and self.device == "cuda":
            print(f"🚀 Kích hoạt chạy song song trên {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # Standard BCE with explicit positive weight
        pos_weight = torch.tensor([config.POS_WEIGHT]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        checkpoint_path = os.path.join(config.MODELS_DIR, "best_model.pth")
        
        print("\n--- Bắt đầu quá trình huấn luyện DNN (1 Fold) ---")
        for epoch in range(config.EPOCHS):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Lưu model tốt nhất
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), checkpoint_path)
                
                print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} 🌟 (Lưu Checkpoint)")
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                    
        # Load best weights
        if os.path.exists(checkpoint_path):
             model_to_load = model.module if isinstance(model, nn.DataParallel) else model
             model_to_load.load_state_dict(torch.load(checkpoint_path))
             
        return model

    def run_training(self, embeddings: np.ndarray, labels: np.ndarray):
        # 80/20 Stratified Split
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=config.TEST_SIZE, 
            stratify=labels, random_state=config.SEED
        )
        
        print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
        
        train_dataset = AudioDataset(X_train, y_train)
        val_dataset = AudioDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        model = AudioDNN().to(self.device)
        model = self.train_model(model, train_loader, val_loader)
        
        model.eval()
        val_probs = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                probs = torch.sigmoid(model(X_batch)).cpu().numpy()
                val_probs.extend(probs.flatten())
        
        return y_val, np.array(val_probs)
