"""
Enhanced Training Loop for Failed Cases
Focuses on medium degradation, edge cases, and hard examples
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


# Configuration
CHECKPOINT_PATH = "phase2_epoch_5_calibrated_v2.pth"  # Starting point
SAVE_DIR = "./enhanced_model_checkpoints"
DATA_DIRS = [
    # Add your data directories here
    "/content/local_data/EcoRevive_Balanced-Forest",
    "/content/local_data/EcoRevive_Balanced-Wetland",
    "/content/local_data/EcoRevive_Balanced-Grassland",
    # ... add more
]

# Training settings
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enhanced training parameters
EPOCHS = 25
LEARNING_RATE = 5e-5  # Lower for stability
HARD_MINE_FREQUENCY = 5  # Mine hard examples every N epochs

# Loss weights
LOSS_WEIGHTS = {
    'focal': 1.0,
    'boundary': 0.8,
    'uncertainty': 0.3,
    'iou': 0.5
}

# Difficulty-based sampling weights
EASY_WEIGHT = 1.0
MEDIUM_WEIGHT = 2.0
HARD_WEIGHT = 3.0


def create_difficulty_weighted_sampler(dataset, difficulty_scores):
    """
    Create sampler that oversamples hard examples
    """
    # Create weight for each sample
    weights = []
    for idx in range(len(dataset)):
        if idx in difficulty_scores:
            difficulty = difficulty_scores[idx]
            # Map difficulty [0,1] to weight [EASY_WEIGHT, HARD_WEIGHT]
            weight = EASY_WEIGHT + (HARD_WEIGHT - EASY_WEIGHT) * difficulty
        else:
            weight = MEDIUM_WEIGHT  # Default for unknown samples
        weights.append(weight)
    
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    return sampler


def validate(model, val_loader, criterion, device):
    """Validation with enhanced metrics"""
    model.eval()
    val_loss = 0.0
    loss_components = {'focal': 0, 'boundary': 0, 'uncertainty': 0, 'iou': 0}
    
    # Track performance by degradation level
    performance = {
        'high_degradation': {'count': 0, 'error': 0},
        'medium_degradation': {'count': 0, 'error': 0},
        'low_degradation': {'count': 0, 'error': 0}
    }
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.unsqueeze(1).to(device, non_blocking=True)
            
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
    
    # Calculate average errors per category
    for category in performance:
        if performance[category]['count'] > 0:
            performance[category]['avg_error'] = \
                performance[category]['error'] / performance[category]['count']
        else:
            performance[category]['avg_error'] = 0.0
    
    return avg_loss, avg_components, performance


def train_enhanced(use_hard_mining=True):
    """
    Main enhanced training function
    """
    print("="*80)
    print("🚀 ENHANCED TRAINING - Focused on Failed Cases")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Hard Mining: {'Enabled (every {} epochs)'.format(HARD_MINE_FREQUENCY) if use_hard_mining else 'Disabled'}")
    print("="*80 + "\n")
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    train_dataset = EcoReviveDataset(DATA_DIRS, augment=True, mode='train')
    val_dataset = EcoReviveDataset(DATA_DIRS, augment=False, mode='val')
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples\n")
    
    # Load model
    print("🧠 Loading model...")
    model = RefinedEcoReviveModel(device=DEVICE)
    model.load_weights(CHECKPOINT_PATH)
    model = model.to(DEVICE)
    print(f"   ✅ Model loaded from {CHECKPOINT_PATH}\n")
    
    # Enhanced loss function
    criterion = EnhancedCompositeLoss(**LOSS_WEIGHTS)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # Initial validation loader (will be updated with difficulty-weighted sampler)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Initial hard mining (optional)
    difficulty_scores = {}
    if use_hard_mining:
        print("🔍 Initial hard negative mining...")
        miner = HardNegativeMiner(model, DEVICE)
        difficulty_scores, sample_metrics = miner.compute_difficulty_scores(
            train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        
        # Save initial mining results
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
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    
    # Training loop
    best_val_loss = float('inf')
    best_medium_error = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Re-mine hard examples periodically
        if use_hard_mining and epoch > 0 and epoch % HARD_MINE_FREQUENCY == 0:
            print(f"\n🔍 Re-mining hard examples (Epoch {epoch})...")
            miner = HardNegativeMiner(model, DEVICE)
            difficulty_scores, sample_metrics = miner.compute_difficulty_scores(
                train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
            )
            
            # Update sampler
            sampler = create_difficulty_weighted_sampler(train_dataset, difficulty_scores)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
            print("   ✅ Updated difficulty weights\n")
        
        # Training
        model.train()
        train_loss = 0.0
        train_components = {'focal': 0, 'boundary': 0, 'uncertainty': 0, 'iou': 0}
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.unsqueeze(1).to(DEVICE, non_blocking=True)
            
            # Forward pass
            logits = model(images)
            loss, components = criterion(logits, masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item()
            for k in train_components:
                train_components[k] += components[k]
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{EPOCHS} [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        
        # Validation
        val_loss, val_components, performance = validate(model, val_loader, criterion, DEVICE)
        
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
        
        # Save checkpoint
        save_path = os.path.join(SAVE_DIR, f"enhanced_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'performance': performance,
        }, save_path)
        print(f"   ✅ Saved: {save_path}")
        
        # Save best model (based on medium degradation error)
        medium_error = performance['medium_degradation']['avg_error']
        if medium_error < best_medium_error:
            best_medium_error = medium_error
            best_path = os.path.join(SAVE_DIR, "enhanced_BEST.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'medium_error': medium_error,
                'performance': performance,
            }, best_path)
            print(f"   ⭐ NEW BEST (Medium Error: {medium_error:.4f}): {best_path}\n")
    
    print("\n" + "="*80)
    print("🎉 ENHANCED TRAINING COMPLETE!")
    print("="*80)
    print(f"   Best Medium Degradation Error: {best_medium_error:.4f}")
    print(f"   Model saved to: {os.path.join(SAVE_DIR, 'enhanced_BEST.pth')}")
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
