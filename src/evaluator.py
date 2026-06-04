import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config
from src.model import SequenceAudioDataset, AttentionHeadClassifier
import copy

class FocalLossWithLogits(nn.Module):
    """
    Focal Loss designed to address severe class imbalance by focusing on hard-to-classify examples.
    """
    def __init__(self, alpha=config.POS_WEIGHT, gamma=2.0):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs are raw logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class Evaluator:
    def __init__(self, n_folds: int = config.N_FOLDS):
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
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

    def evaluate_oof(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float):
        y_pred = (y_probs >= threshold).astype(int)
        
        print("\n" + "="*40)
        print("FINAL EVALUATION REPORT (SOTA ATTENTION)")
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
        plt.title('Confusion Matrix (Out-of-Fold)')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def train_model(self, model, train_loader, val_loader, fold_idx=None):
        # Tận dụng tối đa Multi-GPU (T4 x2)
        if torch.cuda.device_count() > 1 and self.device == "cuda":
            print(f"🚀 Kích hoạt chạy song song trên {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # Use Focal Loss instead of standard BCE
        criterion = FocalLossWithLogits(alpha=config.POS_WEIGHT, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        # Tên file checkpoint để lưu vật lý xuống ổ cứng
        checkpoint_name = f"best_model_fold_{fold_idx}.pth" if fold_idx is not None else "best_model_final.pth"
        checkpoint_path = os.path.join(config.MODELS_DIR, checkpoint_name)
        
        print("\n--- Bắt đầu quá trình huấn luyện DNN ---")
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
            
            # Liên tục lưu model tốt nhất vật lý xuống ổ cứng
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
                # Lưu file .pth xuống disk ngay lập tức
                # Xử lý trường hợp dùng DataParallel
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), checkpoint_path)
                
                print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} 🌟 (Lưu Checkpoint: {checkpoint_name})")
            else:
                # In log thường nếu không phải epoch tốt nhất
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                    
        # Load lại trọng số tốt nhất sau khi kết thúc tất cả epochs từ file vật lý cho chắc chắn
        if os.path.exists(checkpoint_path):
             model_to_load = model.module if isinstance(model, nn.DataParallel) else model
             model_to_load.load_state_dict(torch.load(checkpoint_path))
             
        return model

    def run_cv(self, embeddings: np.ndarray, labels: np.ndarray):
        oof_probs = np.zeros(len(labels))
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(embeddings, labels)):
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            train_dataset = SequenceAudioDataset(X_train, y_train)
            val_dataset = SequenceAudioDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            model = AttentionHeadClassifier().to(self.device)
            model = self.train_model(model, train_loader, val_loader, fold_idx=fold+1)
            
            model.eval()
            val_probs = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    probs = torch.sigmoid(model(X_batch)).cpu().numpy()
                    val_probs.extend(probs.flatten())
            
            oof_probs[val_idx] = val_probs
            print(f"Fold {fold+1} completed.")
            
        return oof_probs
