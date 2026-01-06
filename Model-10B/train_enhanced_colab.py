"""
Enhanced Training for Google Colab - Optimized for A100 + Google Drive
Run this on Colab with GPU enabled
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from dataset import EcoReviveDataset
from refined_model import RefinedEcoReviveModel
from adaptive_losses import EnhancedCompositeLoss
from hard_negative_mining import HardNegativeMiner


# =====================================================================
# GOOGLE COLAB CONFIGURATION
# =====================================================================

# Google Drive paths
CHECKPOINT_PATH = "/content/drive/MyDrive/RefinedModel/CalibratedP2.pth"
SAVE_DIR = "/content/drive/MyDrive/RefinedModel/Enhanced"

# Data directories (LOCAL DISK - much faster than Drive!)
# Run copy_data_to_local.py first to populate these directories
DATA_DIRS = [
    "/content/local_data/EcoRevive_Balanced-Dryland",
    "/content/local_data/EcoRevive_Balanced-Coastal",
    "/content/local_data/EcoRevive_Balanced-Peatland",
    "/content/local_data/EcoRevive_Balanced-Tropical",
    "/content/local_data/EcoRevive_Balanced-Mediterranean",
    "/content/local_data/EcoRevive_Balanced-Grassland",
    "/content/local_data/EcoRevive_Balanced-Wetland",
    "/content/local_data/EcoRevive_Balanced-Forest"
]

# A100 + Local SSD optimized settings
BATCH_SIZE = 32  # Increased for local SSD (A100 can handle it)
NUM_WORKERS = 6  # Higher workers for fast local SSD (was 2 for Drive)
PREFETCH_FACTOR = 4  # More prefetch with fast local I/O
PERSISTENT_WORKERS = True  # Keep workers alive
PIN_MEMORY = True  # Faster GPU transfer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
EPOCHS = 25
LEARNING_RATE = 5e-5
HARD_MINE_FREQUENCY = 5  # Mine hard examples every N epochs
CHECKPOINT_FREQUENCY = 5  # Save periodic checkpoints every N epochs
EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for N epochs

# Mixed precision for A100
USE_AMP = True  # A100 has excellent FP16 performance
GRAD_SCALER = torch.cuda.amp.GradScaler() if USE_AMP else None

# Loss weights (keys must match EnhancedCompositeLoss parameters)
LOSS_WEIGHTS = {
    'focal_weight': 1.0,
    'boundary_weight': 0.8,
    'uncertainty_weight': 0.3,
    'iou_weight': 0.5
}

# Difficulty-based sampling weights
EASY_WEIGHT = 1.0
MEDIUM_WEIGHT = 2.0
HARD_WEIGHT = 3.0


def create_difficulty_weighted_sampler(dataset, difficulty_scores):
    """Create sampler that oversamples hard examples"""
    weights = []
    for idx in range(len(dataset)):
        if idx in difficulty_scores:
            difficulty = difficulty_scores[idx]
            weight = EASY_WEIGHT + (HARD_WEIGHT - EASY_WEIGHT) * difficulty
        else:
            weight = MEDIUM_WEIGHT
        weights.append(weight)
    
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validation with enhanced metrics"""
    model.eval()
    val_loss = 0.0
    loss_components = {'focal': 0, 'boundary': 0, 'uncertainty': 0, 'iou': 0}
    
    performance = {
        'high_degradation': {'count': 0, 'error': 0},
        'medium_degradation': {'count': 0, 'error': 0},
        'low_degradation': {'count': 0, 'error': 0}
    }
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.unsqueeze(1).to(device, non_blocking=True)
            
            # Use autocast for validation too
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss, components = criterion(logits, masks)
            else:
                logits = model(images)
                loss, components = criterion(logits, masks)
            
            val_loss += loss.item()
            for k in loss_components:
                loss_components[k] += components[k]
            
            # Categorize by degradation level
            probs = torch.sigmoid(logits)
            mean_target = masks.mean(dim=[1,2,3])
            mean_pred = probs.mean(dim=[1,2,3])
            error = torch.abs(mean_target - mean_pred)
            
            for i in range(len(mean_target)):
                if mean_target[i] > 0.6:
                    performance['high_degradation']['count'] += 1
                    performance['high_degradation']['error'] += error[i].item()
                elif mean_target[i] > 0.3:
                    performance['medium_degradation']['count'] += 1
                    performance['medium_degradation']['error'] += error[i].item()
                else:
                    performance['low_degradation']['count'] += 1
                    performance['low_degradation']['error'] += error[i].item()
    
    num_batches = len(val_loader)
    avg_loss = val_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    for category in performance:
        if performance[category]['count'] > 0:
            performance[category]['avg_error'] = \
                performance[category]['error'] / performance[category]['count']
        else:
            performance[category]['avg_error'] = 0.0
    
    return avg_loss, avg_components, performance


def train_enhanced(use_hard_mining=True):
    """
    Enhanced training optimized for Google Colab A100
    """
    print("="*80)
    print("🚀 ENHANCED TRAINING - Google Colab A100")
    print("="*80)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Batch Size: {BATCH_SIZE} (A100 optimized)")
    print(f"Workers: {NUM_WORKERS} (Drive I/O optimized)")
    print(f"Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Hard Mining: {'Every {} epochs'.format(HARD_MINE_FREQUENCY) if use_hard_mining else 'Disabled'}")
    print("="*80 + "\n")
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Verify data directories exist
    print("📂 Verifying data directories...")
    valid_dirs = []
    for d in DATA_DIRS:
        if os.path.exists(d):
            num_files = len([f for f in os.listdir(d) if f.endswith('.tif')])
            print(f"   ✅ {os.path.basename(d)}: {num_files} files")
            valid_dirs.append(d)
        else:
            print(f"   ⚠️  Not found: {d}")
    
    if len(valid_dirs) == 0:
        raise ValueError("No valid data directories found!")
    
    print(f"\n   Total: {len(valid_dirs)}/{len(DATA_DIRS)} directories available\n")
    
    # Load data with Drive-optimized settings
    print("📂 Loading datasets...")
    train_dataset = EcoReviveDataset(valid_dirs, augment=True, mode='train')
    val_dataset = EcoReviveDataset(valid_dirs, augment=False, mode='val')
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples\n")
    
    # Load model
    print("🧠 Loading model...")
    model = RefinedEcoReviveModel(device=DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        print(f"   ✅ Loaded from {CHECKPOINT_PATH}")
    else:
        print(f"   ⚠️  Checkpoint not found, starting from scratch")
    
    model = model.to(DEVICE)
    
    # Enable cudnn benchmarking for A100
    torch.backends.cudnn.benchmark = True
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {total_params:.2f}M\n")
    
    # Enhanced loss function
    criterion = EnhancedCompositeLoss(**LOSS_WEIGHTS)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # Validation loader (Drive-optimized)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    # Initial hard mining
    difficulty_scores = {}
    if use_hard_mining:
        print("🔍 Initial hard negative mining...")
        miner = HardNegativeMiner(model, DEVICE)
        difficulty_scores, sample_metrics = miner.compute_difficulty_scores(
            train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        
        miner.save_results(difficulty_scores, sample_metrics, 
                          os.path.join(SAVE_DIR, "initial_difficulty_scores.json"))
        
        easy, medium, hard = miner.categorize_by_difficulty(difficulty_scores)
        print(f"   Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}\n")
    
    # Create dataloader with difficulty weighting
    if difficulty_scores:
        sampler = create_difficulty_weighted_sampler(train_dataset, difficulty_scores)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR
        )
    
    # Training loop
    best_val_loss = float('inf')
    best_medium_error = float('inf')
    epochs_without_improvement = 0
    
    print("="*80)
    print("🏋️ TRAINING START")
    print("="*80 + "\n")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Re-mine hard examples periodically
        if use_hard_mining and epoch > 0 and epoch % HARD_MINE_FREQUENCY == 0:
            print(f"\n🔍 Re-mining hard examples (Epoch {epoch})...")
            miner = HardNegativeMiner(model, DEVICE)
            difficulty_scores, sample_metrics = miner.compute_difficulty_scores(
                train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
            )
            
            sampler = create_difficulty_weighted_sampler(train_dataset, difficulty_scores)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS,
                prefetch_factor=PREFETCH_FACTOR
            )
            print("   ✅ Updated difficulty weights\n")
        
        # Training
        model.train()
        train_loss = 0.0
        train_components = {'focal': 0, 'boundary': 0, 'uncertainty': 0, 'iou': 0}
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.unsqueeze(1).to(DEVICE, non_blocking=True)
            
            # Mixed precision training
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss, components = criterion(logits, masks)
                
                GRAD_SCALER.scale(loss).backward()
                GRAD_SCALER.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                GRAD_SCALER.step(optimizer)
                GRAD_SCALER.update()
            else:
                logits = model(images)
                loss, components = criterion(logits, masks)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item()
            for k in train_components:
                train_components[k] += components[k]
            
            if (batch_idx + 1) % 25 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{EPOCHS} [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        
        # Validation
        val_loss, val_components, performance = validate(model, val_loader, criterion, DEVICE, USE_AMP)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\n{'='*80}")
        print(f"📈 Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time/60:.1f}min")
        print(f"{'='*80}")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"\n   Loss Components:")
        for k, v in val_components.items():
            print(f"      {k}: {v:.4f}")
        print(f"\n   Performance by Degradation Level:")
        print(f"      High (>60%): {performance['high_degradation']['avg_error']:.4f} error")
        print(f"      Medium (30-60%): {performance['medium_degradation']['avg_error']:.4f} error ⭐")
        print(f"      Low (<30%): {performance['low_degradation']['avg_error']:.4f} error")
        print(f"{'='*80}\n")
        
        # Smart checkpointing
        medium_error = performance['medium_degradation']['avg_error']
        model_improved = False
        
        # Condition 1: Save if model improved
        if medium_error < best_medium_error:
            best_medium_error = medium_error
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model_improved = True
            
            # Save BEST model
            best_path = os.path.join(SAVE_DIR, "enhanced_BEST.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'medium_error': medium_error,
                'performance': performance,
            }, best_path)
            print(f"   ⭐ NEW BEST (Medium Error: {medium_error:.4f}) - Saved")
        else:
            epochs_without_improvement += 1
            print(f"   No improvement for {epochs_without_improvement} epoch(s)")
        
        # Condition 2: Save every 5 epochs (periodic checkpoint)
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
            periodic_path = os.path.join(SAVE_DIR, f"enhanced_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'performance': performance,
            }, periodic_path)
            print(f"   📦 Periodic checkpoint saved: epoch_{epoch+1}.pth")
        
        # Condition 3: Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  EARLY STOPPING triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            
            # Save early stopping checkpoint
            early_stop_path = os.path.join(SAVE_DIR, "enhanced_EARLY_STOP.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'performance': performance,
                'reason': 'early_stopping'
            }, early_stop_path)
            print(f"   💾 Early stop checkpoint saved")
            print(f"   Best model was at epoch {epoch + 1 - EARLY_STOPPING_PATIENCE}")
            break
    
    print("\n" + "="*80)
    print("🎉 ENHANCED TRAINING COMPLETE!")
    print("="*80)
    print(f"   Best Medium Degradation Error: {best_medium_error:.4f}")
    print(f"   Saved to: {SAVE_DIR}/enhanced_BEST.pth")
    print("="*80)


if __name__ == "__main__":
    try:
        train_enhanced(use_hard_mining=True)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
