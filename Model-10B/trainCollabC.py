import os
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

from dataset import EcoReviveDataset
from refined_model import RefinedEcoReviveModel, CompositeLoss

# Configuration
CHECKPOINT_PATH = "/content/drive/MyDrive/ecorevive_best.pth"
SAVE_DIR = "/content/drive/MyDrive/RefinedModel"

DIRS = [
    "/content/local_data/EcoRevive_Balanced-Dryland",
    "/content/local_data/EcoRevive_Balanced-Coastal",
    "/content/local_data/EcoRevive_Balanced-Peatland",
    "/content/local_data/EcoRevive_Balanced-Tropical",
    "/content/local_data/EcoRevive_Balanced-Mediterranean",
    "/content/local_data/EcoRevive_Balanced-Grassland",
    "/content/local_data/EcoRevive_Balanced-Wetland",
    "/content/local_data/EcoRevive_Balanced-Forest"
]

# A100-optimized settings
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training phases
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4

# CRITICAL: Disable mixed precision to avoid dtype mismatch with BatchNorm
USE_AMP = False  # Set to False to avoid issues

torch.backends.cudnn.benchmark = True


# ==================== SAFE DATASET WRAPPER ====================

class SafeDataset(Dataset):
    """Wrapper around a dataset to catch and skip problematic samples."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.n = len(base_dataset)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        tries = 0
        start_idx = idx % self.n
        while tries < self.n:
            try:
                return self.base[start_idx]
            except Exception as e:
                print(f"⚠️ Data error at idx {start_idx}: {e}")
                start_idx = (start_idx + 1) % self.n
                tries += 1
        raise RuntimeError("SafeDataset: no valid samples found (all failed).")


# ==================== METRICS ====================

def compute_metrics_with_threshold_sweep(logits, targets, thresholds=None):
    """
    Compute IoU, F1, Precision, Recall across multiple thresholds
    Returns best threshold and metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.8, 0.05)
    
    probs = torch.sigmoid(logits)
    
    best_iou = 0.0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        preds = (probs > thresh).float()
        
        # Global metrics (not batch-averaged)
        intersection = (preds * targets).sum().item()
        union = preds.sum().item() + targets.sum().item() - intersection
        
        tp = intersection
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        if iou > best_iou:
            best_iou = iou
            best_threshold = thresh
            best_metrics = {
                'iou': iou,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'threshold': thresh
            }
    
    return best_metrics


def validate(model, val_loader, criterion, device, compute_best_threshold=True):
    """
    Validation with adaptive thresholding
    """
    model.eval()
    val_loss = 0.0
    
    # Accumulate all predictions and targets for global threshold search
    all_logits = []
    all_targets = []
    
    loss_components = {'iou': 0, 'tversky': 0, 'boundary': 0, 'connectivity': 0}
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.unsqueeze(1).to(device, non_blocking=True)
            
            logits = model(images)
            loss, components = criterion(logits, masks)
            
            val_loss += loss.item()
            for k in loss_components:
                loss_components[k] += components[k]
            
            if compute_best_threshold:
                all_logits.append(logits.cpu())
                all_targets.append(masks.cpu())
            
            num_batches += 1
    
    avg_loss = val_loss / num_batches if num_batches > 0 else float('inf')
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    if compute_best_threshold and len(all_logits) > 0:
        # Compute best threshold on entire validation set
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics_with_threshold_sweep(all_logits, all_targets)
    else:
        # Fallback: use fixed threshold
        metrics = {
            'iou': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'threshold': 0.5
        }
    
    return avg_loss, metrics, avg_components


def save_checkpoint(state, path, is_best=False):
    """Safe checkpoint saving with extensive error handling"""
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save to temporary file first
        temp_path = path + ".tmp"
        torch.save(state, temp_path)
        
        # Verify the save worked
        if not os.path.exists(temp_path):
            raise IOError(f"Failed to create temporary file: {temp_path}")
        
        # Move to final location
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)
        
        # Verify final file exists
        if not os.path.exists(path):
            raise IOError(f"Failed to create final checkpoint: {path}")
        
        if is_best:
            print(f"✅ NEW BEST MODEL SAVED: IoU={state.get('val_iou', 0):.4f}")
            print(f"   Location: {path}")
        else:
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✅ Checkpoint saved: {os.path.basename(path)} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Checkpoint save failed!")
        print(f"   Path: {path}")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ==================== MAIN TRAINING ====================

def train_refined():
    print("=" * 80)
    print("🚀 REFINED TRAINING: Pushing to Prithvi-Level Performance")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Mixed Precision: {'Enabled' if USE_AMP else 'Disabled (for stability)'}")
    print(f"Phase 1 (Frozen Backbone): {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}")
    print(f"Phase 2 (Full Training): {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}")
    print("=" * 80 + "\n")
    
    # Verify checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    # Create save directory and verify write access
    os.makedirs(SAVE_DIR, exist_ok=True)
    test_file = os.path.join(SAVE_DIR, ".write_test")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Save directory verified: {SAVE_DIR}\n")
    except Exception as e:
        print(f"❌ Cannot write to save directory: {SAVE_DIR}")
        print(f"   Error: {e}")
        return
    
    # Load data
    valid_dirs = [d for d in DIRS if os.path.exists(d)]
    if len(valid_dirs) == 0:
        print("❌ No data directories found!")
        return
    
    print(f"📂 Loading data from {len(valid_dirs)} directories...")
    raw_dataset = EcoReviveDataset(valid_dirs, augment=True, mode='train')
    dataset = SafeDataset(raw_dataset)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Train: {train_size} | Val: {val_size}\n")
    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    # Initialize model
    print("🧠 Initializing refined model...")
    model = RefinedEcoReviveModel(CHECKPOINT_PATH, device=DEVICE)
    model = model.to(DEVICE)
    
    # Ensure model is in float32
    model.float()
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Total params: {total_params:.2f}M")
    print(f"   Trainable: {trainable_params:.2f}M\n")
    
    # Loss function
    criterion = CompositeLoss(
        iou_weight=1.0,
        tversky_weight=1.0,
        boundary_weight=0.5,
        connectivity_weight=0.3
    )
    
    # Training state
    best_val_iou = 0.0
    best_threshold = 0.5
    global_step = 0
    
    # ==================== PHASE 1: Frozen Backbone ====================
    print("=" * 80)
    print("📍 PHASE 1: Training Spatial Refinement (Backbone Frozen)")
    print("=" * 80 + "\n")
    
    model.freeze_backbone()
    
    optimizer = optim.AdamW([
        {'params': model.spatial_refine.parameters(), 'lr': PHASE1_LR},
        {'params': model.temperature.parameters(), 'lr': PHASE1_LR},
        {'params': model.final_conv.parameters(), 'lr': PHASE1_LR}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-6
    )
    
    phase1_start = time.time()
    
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_components = {'iou': 0, 'tversky': 0, 'boundary': 0, 'connectivity': 0}
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.unsqueeze(1).to(DEVICE, non_blocking=True)
                
                # Forward pass without autocast (for stability)
                logits = model(images)
                loss, components = criterion(logits, masks)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                epoch_loss += loss.item()
                for k in epoch_components:
                    epoch_components[k] += components[k]
                
                global_step += 1
                
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch+1}/{PHASE1_EPOCHS} [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
            
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        scheduler.step()
        
        # Validation
        val_loss, metrics, val_components = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\n📈 Phase 1 - Epoch {epoch+1}/{PHASE1_EPOCHS}")
        print(f"   Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val IoU: {metrics['iou']:.4f} (thresh={metrics['threshold']:.2f})")
        print(f"   Val F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
        print(f"   Loss breakdown: IoU={val_components['iou']:.4f}, Tversky={val_components['tversky']:.4f}, Boundary={val_components['boundary']:.4f}\n")
        
        # Save Phase 1 checkpoint after each epoch
        phase1_path = os.path.join(SAVE_DIR, f"phase1_epoch_{epoch+1}.pth")
        save_checkpoint({
            'phase': 1,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_iou': metrics['iou'],
            'val_f1': metrics['f1'],
            'best_threshold': metrics['threshold'],
            'temperature': model.temperature.temperature.item(),
        }, phase1_path)
        
        if metrics['iou'] > best_val_iou:
            best_val_iou = metrics['iou']
            best_threshold = metrics['threshold']
            
            # Save best model from Phase 1
            best_phase1_path = os.path.join(SAVE_DIR, "phase1_best.pth")
            save_checkpoint({
                'phase': 1,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': metrics['iou'],
                'val_f1': metrics['f1'],
                'best_threshold': metrics['threshold'],
                'temperature': model.temperature.temperature.item(),
            }, best_phase1_path, is_best=True)
    
    phase1_time = time.time() - phase1_start
    print(f"✅ Phase 1 complete in {format_time(phase1_time)}")
    print(f"   Best IoU: {best_val_iou:.4f}\n")
    
    # ==================== PHASE 2: Full Training ====================
    print("=" * 80)
    print("📍 PHASE 2: End-to-End Fine-Tuning (All Layers Unfrozen)")
    print("=" * 80 + "\n")
    
    model.unfreeze_backbone()
    
    # New optimizer for all parameters
    optimizer = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
    
    # FIXED: Removed 'verbose' parameter (not available in older PyTorch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7
    )
    
    patience_counter = 0
    patience_limit = 7
    
    phase2_start = time.time()
    
    for epoch in range(PHASE2_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_components = {'iou': 0, 'tversky': 0, 'boundary': 0, 'connectivity': 0}
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
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
                
                epoch_loss += loss.item()
                for k in epoch_components:
                    epoch_components[k] += components[k]
                
                global_step += 1
                
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch+1}/{PHASE2_EPOCHS} [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
            
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        
        # Validation
        val_loss, metrics, val_components = validate(model, val_loader, criterion, DEVICE)
        
        # Step scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['iou'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        print(f"\n{'='*80}")
        print(f"📈 Phase 2 - Epoch {epoch+1}/{PHASE2_EPOCHS} | Time: {format_time(epoch_time)}")
        print(f"{'='*80}")
        print(f"   Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val IoU: {metrics['iou']:.4f} ⭐ (thresh={metrics['threshold']:.2f})")
        print(f"   Val F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"   Loss: IoU={val_components['iou']:.4f}, Tversky={val_components['tversky']:.4f}, Bdry={val_components['boundary']:.4f}, Conn={val_components['connectivity']:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}\n")
        
        # Save checkpoint every epoch
        save_path = os.path.join(SAVE_DIR, f"phase2_epoch_{epoch+1}.pth")
        save_checkpoint({
            'phase': 2,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_iou': metrics['iou'],
            'val_f1': metrics['f1'],
            'best_threshold': metrics['threshold'],
            'temperature': model.temperature.temperature.item(),
        }, save_path)
        
        # Save best model
        if metrics['iou'] > best_val_iou:
            best_val_iou = metrics['iou']
            best_threshold = metrics['threshold']
            patience_counter = 0
            
            best_path = os.path.join(SAVE_DIR, "ecorevive_refined_BEST.pth")
            save_checkpoint({
                'phase': 2,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': metrics['iou'],
                'val_f1': metrics['f1'],
                'best_threshold': metrics['threshold'],
                'temperature': model.temperature.temperature.item(),
            }, best_path, is_best=True)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⏹️  Early stopping: No improvement for {patience_limit} epochs\n")
            break
    
    phase2_time = time.time() - phase2_start
    total_time = phase1_time + phase2_time
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"   Best Validation IoU: {best_val_iou:.4f}")
    print(f"   Optimal Threshold: {best_threshold:.2f}")
    print(f"   Total Time: {format_time(total_time)}")
    print(f"   Estimated Cost (A100): ~${(total_time/3600)*4:.2f}")
    print("=" * 80)
    print(f"\n📦 Best model saved to: {os.path.join(SAVE_DIR, 'ecorevive_refined_BEST.pth')}")
    print(f"   Use threshold={best_threshold:.2f} for inference")
    print(f"   Temperature scaling: {model.temperature.temperature.item():.3f}")
    print(f"\n📂 All checkpoints saved in: {SAVE_DIR}")


if __name__ == "__main__":
    try:
        train_refined()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()