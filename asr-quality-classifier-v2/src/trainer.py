"""
trainer.py — PyTorch dataset, dataloader collator, and training loop.

Optimizes only trainable parameters (alignment head and MLP).
Uses BCEWithLogitsLoss with positive class weighting and early stopping on validation Macro F1.
"""

import os
import time
import copy
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score

from src.config import Config, logger
from src.audio_features import process_audio_file

class ASRDataset(Dataset):
    """
    PyTorch Dataset for audio-transcript pairs.
    Loads raw waveforms on-the-fly to save memory (if no precomputed pools are provided).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        acoustic_feats: np.ndarray,
        crossmodal_feats: np.ndarray,
        precomputed_audio_pools: Optional[np.ndarray] = None,
        precomputed_text_pools: Optional[np.ndarray] = None
    ):
        self.df = df
        self.config = config
        self.audio_paths = df["absolute_path"].values
        self.texts = df["transcript"].values
        self.labels = df["binary_label"].values
        
        self.acoustic_feats = acoustic_feats
        self.crossmodal_feats = crossmodal_feats
        
        self.precomputed_audio_pools = precomputed_audio_pools
        self.precomputed_text_pools = precomputed_text_pools
        
        self.sample_rate = config.audio.sample_rate
        self.max_duration = config.audio.max_duration_sec

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.audio_paths[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Only load raw audio if we don't have offline embeddings
        # This speeds up training 100x when using precomputed pools.
        if self.precomputed_audio_pools is None:
            try:
                y, _ = librosa.load(
                    path, 
                    sr=self.sample_rate, 
                    mono=True, 
                    duration=self.max_duration
                )
            except Exception as e:
                logger.error(f"Error reading audio file {path} inside dataset: {e}")
                y = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence fallback
        else:
            y = np.array([]) # Empty array, will not be used
            
        item = {
            "waveform": y,
            "text": text,
            "acoustic_feat": torch.tensor(self.acoustic_feats[idx], dtype=torch.float32),
            "crossmodal_feat": torch.tensor(self.crossmodal_feats[idx], dtype=torch.float32),
            "label": label
        }
        
        if self.precomputed_audio_pools is not None:
            item["audio_pool"] = torch.tensor(self.precomputed_audio_pools[idx], dtype=torch.float32)
        if self.precomputed_text_pools is not None:
            item["text_pool"] = torch.tensor(self.precomputed_text_pools[idx], dtype=torch.float32)
            
        return item

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-length raw waveforms and text into lists."""
    waveforms = [item["waveform"] for item in batch]
    texts = [item["text"] for item in batch]
    acoustic_feats = torch.stack([item["acoustic_feat"] for item in batch])
    crossmodal_feats = torch.stack([item["crossmodal_feat"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    
    result = {
        "waveforms": waveforms,
        "texts": texts,
        "acoustic_feats": acoustic_feats,
        "crossmodal_feats": crossmodal_feats,
        "labels": labels
    }
    
    if "audio_pool" in batch[0]:
        result["audio_pools"] = torch.stack([item["audio_pool"] for item in batch])
    if "text_pool" in batch[0]:
        result["text_pools"] = torch.stack([item["text_pool"] for item in batch])
        
    return result

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config
) -> float:
    """Trains the model for one epoch."""
    model.train()
    # Encoders remain frozen (eval mode), but we set the container to train
    model.audio_encoder.eval()
    model.text_encoder.eval()
    
    total_loss = 0.0
    device = config.device
    
    for batch in dataloader:
        waveforms = batch["waveforms"]
        texts = batch["texts"]
        acoustic_feats = batch["acoustic_feats"].to(device)
        crossmodal_feats = batch["crossmodal_feats"].to(device)
        labels = batch["labels"].to(device)
        
        audio_pools = batch.get("audio_pools")
        text_pools = batch.get("text_pools")
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            waveforms, texts, acoustic_feats, crossmodal_feats,
            precomputed_audio_pool=audio_pools,
            precomputed_text_pool=text_pools
        )
        loss = criterion(logits, labels)
        
        # Backward & Optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(waveforms)
        
    return total_loss / len(dataloader.dataset)

def evaluate_predictions(
    logits: np.ndarray, 
    labels: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Computes F1 and AUC from logits."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    macro_f1 = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5  # Handle case where only 1 class is present in batch
        
    return {"macro_f1": macro_f1, "auc": auc}

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Config
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates the model over the validation set."""
    model.eval()
    total_loss = 0.0
    device = config.device
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"]
            texts = batch["texts"]
            acoustic_feats = batch["acoustic_feats"].to(device)
            crossmodal_feats = batch["crossmodal_feats"].to(device)
            labels = batch["labels"].to(device)
            
            audio_pools = batch.get("audio_pools")
            text_pools = batch.get("text_pools")
            
            logits = model(
                waveforms, texts, acoustic_feats, crossmodal_feats,
                precomputed_audio_pool=audio_pools,
                precomputed_text_pool=text_pools
            )
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(waveforms)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    val_loss = total_loss / len(dataloader.dataset)
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    return val_loss, all_logits, all_labels

def sweep_threshold(
    logits: np.ndarray, 
    labels: np.ndarray, 
    config: Config
) -> Tuple[float, float]:
    """Sweeps decision thresholds to find the best Macro F1 score."""
    best_f1 = -1.0
    best_threshold = 0.5
    
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    thresholds = np.arange(
        config.training.threshold_start,
        config.training.threshold_end,
        config.training.threshold_step
    )
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(labels, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = t
            
    return best_threshold, best_f1

def train_neural_model(
    model: nn.Module,
    train_dataset: ASRDataset,
    val_dataset: ASRDataset,
    config: Config,
    save_path: str
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Runs full training and validation loop for ASRQualityClassifier.
    Only trains parameter branches with requires_grad=True.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Only optimize parameters that require grad (encoders are frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params, 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # Class weights for BCE loss
    pos_weight = torch.tensor([config.training.pos_class_weight], device=config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    best_model_wts = copy.deepcopy(model.state_dict())
    
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    
    logger.info(f"Starting deep classifier training on {config.device}...")
    for epoch in range(1, config.training.epochs + 1):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        val_loss, val_logits, val_labels = evaluate_model(model, val_loader, criterion, config)
        
        # Sweep threshold to find best validation F1
        best_t_epoch, val_f1 = sweep_threshold(val_logits, val_labels, config)
        val_auc = roc_auc_score(val_labels, 1.0 / (1.0 + np.exp(-val_logits)))
        
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        
        logger.info(
            f"Epoch {epoch:02d}/{config.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} (t={best_t_epoch:.2f}) | Val AUC: {val_auc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Learning rate scheduling based on validation F1
        scheduler.step(val_f1)
        
        # Check if model improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_threshold = best_t_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_wts,
                "best_threshold": best_threshold,
                "best_val_f1": best_val_f1,
                "config": config
            }, save_path)
            logger.info(f"[*] Saved new best checkpoint to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Load best weights
    model.load_state_dict(best_model_wts)
    logger.info(f"Training complete. Best Validation F1: {best_val_f1:.4f} at threshold {best_threshold:.2f}")
    
    return model, {
        "history": history,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "best_threshold": best_threshold
    }
