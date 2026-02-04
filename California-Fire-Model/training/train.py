"""
Training Script for California Fire Model
Clean training loop with proper validation and checkpointing.

Key features:
1. Per-fire validation tracking
2. Early stopping
3. Mixed precision training
4. Proper logging
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'data'))
sys.path.append(str(Path(__file__).parent.parent / 'model'))

from config import (
    TRAINING_CONFIG, MODEL_CONFIG, CHECKPOINT_DIR, LOG_DIR,
    RAW_DATA_DIR, TRAINING_FIRES, TEST_FIRES, VAL_SPLIT
)
from data.dataset import CaliforniaFireDataset, create_train_val_datasets
from model.architecture import CaliforniaFireModel, save_model
from model.losses import get_loss_function, CombinedLoss
from model.metrics import MetricTracker


class Trainer:
    """Handles training loop and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_val_iou = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_mae': [],
            'lr': [],
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', CHECKPOINT_DIR))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both (image, label) and (image, label, metadata) formats
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss, _ = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('gradient_clip', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss, _ = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('gradient_clip', 1.0)
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress logging
            if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"   [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    def validate(self, epoch: int, track_per_fire: bool = True) -> dict:
        """Validate the model."""
        self.model.eval()
        
        tracker = MetricTracker(threshold=0.5)
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both formats
                if len(batch) == 2:
                    images, labels = batch
                    categories = [None] * images.size(0)
                else:
                    images, labels, metadata = batch
                    categories = [m.get('fire_key', 'unknown') for m in metadata] if isinstance(metadata, list) else [None] * images.size(0)
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss, _ = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss, _ = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Track metrics
                for i in range(images.size(0)):
                    cat = categories[i] if track_per_fire and categories[i] else None
                    tracker.update(logits[i:i+1], labels[i:i+1], category=cat)
        
        val_loss = total_loss / len(self.val_loader)
        summary = tracker.get_summary()
        
        return {
            'loss': val_loss,
            'iou': summary.get('iou', 0),
            'mae': summary.get('mae', 1),
            'dice': summary.get('dice', 0),
            'accuracy': summary.get('accuracy', 0),
            'per_fire': tracker.get_category_summary() if track_per_fire else {},
        }
    
    def train(self, epochs: int) -> dict:
        """Full training loop."""
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70)
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print("=" * 70)
        
        patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch, track_per_fire=True)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Log results
            print(f"\n   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f}")
            print(f"   Val IoU:    {val_metrics['iou']:.4f}")
            print(f"   Val MAE:    {val_metrics['mae']:.4f}")
            print(f"   LR:         {current_lr:.2e}")
            print(f"   Time:       {epoch_time:.1f}s")
            
            # Per-fire breakdown
            if val_metrics['per_fire']:
                print("\n   Per-Fire IoU:")
                for fire, metrics in sorted(val_metrics['per_fire'].items()):
                    iou = metrics.get('iou', 0)
                    print(f"      {fire:<20}: {iou:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['lr'].append(current_lr)
            
            # Checkpointing
            improved = False
            
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                improved = True
                
                # Save best model
                best_path = self.checkpoint_dir / 'best_model.pth'
                save_model(
                    self.model, str(best_path),
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={'iou': val_metrics['iou'], 'mae': val_metrics['mae']},
                )
                print(f"\n   [BEST] NEW BEST IoU: {val_metrics['iou']:.4f} - Saved!")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
            
            # Early stopping
            if improved:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                print(f"\n   No improvement for {self.epochs_without_improvement} epoch(s)")
                
                if self.epochs_without_improvement >= patience:
                    print(f"\n[WARNING] EARLY STOPPING triggered!")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'epoch_{epoch+1}.pth'
                save_model(self.model, str(checkpoint_path), epoch=epoch)
                print(f"   [OK] Checkpoint saved: epoch_{epoch+1}.pth")
        
        # Save final model
        final_path = self.checkpoint_dir / 'final_model.pth'
        save_model(self.model, str(final_path), epoch=epochs)
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 70)
        print("[OK] TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Best Val IoU: {self.best_val_iou:.4f}")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Models saved to: {self.checkpoint_dir}")
        print("=" * 70)
        
        return {
            'best_iou': self.best_val_iou,
            'best_loss': self.best_val_loss,
            'history': self.history,
        }


def main(args):
    """Main training function."""
    print("=" * 70)
    print("CALIFORNIA FIRE MODEL - TRAINING")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Data directories
    data_dirs = [
        str(RAW_DATA_DIR / "fires"),
        str(RAW_DATA_DIR / "healthy"),
    ]
    
    # Check for data
    has_data = any(Path(d).exists() and any(Path(d).iterdir()) for d in data_dirs if Path(d).exists())
    
    if not has_data:
        print("\n[ERROR] No data found!")
        print("   Please run: python data/download_fire_data.py")
        print("   Then download from Google Drive and place in data/raw/")
        return
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, _ = create_train_val_datasets(
        data_dirs,
        val_split=VAL_SPLIT,
    )
    
    # Data loaders
    config = TRAINING_CONFIG.copy()
    config.update(vars(args))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    # Model
    print("\nCreating model...")
    model = CaliforniaFireModel(**MODEL_CONFIG).to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {params:.2f}M")
    
    # Loss
    criterion = CombinedLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=2.0,
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        min_lr=1e-7,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )
    
    # Train
    results = trainer.train(epochs=config['epochs'])
    
    print("\nFinal Results:")
    print(f"   Best IoU: {results['best_iou']:.4f}")
    print(f"   Best Loss: {results['best_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train California Fire Model')
    
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=TRAINING_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=TRAINING_CONFIG['num_workers'],
                        help='Data loader workers')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--checkpoint-dir', type=str, default=str(CHECKPOINT_DIR),
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    main(args)
