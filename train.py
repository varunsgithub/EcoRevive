import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from dataset import EcoReviveDataset
from model import UNet, FocalLoss, DiceLoss

DIRS = ["/Users/varunsingh/Desktop/Projects/EcoRevive/Train Data/Nature_Restoration_California", "/Users/varunsingh/Desktop/Projects/EcoRevive/Train Data/Nature_Restoration_Louisiana"]
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def validate(model, val_loader, focal_criterion, dice_criterion):
    """Validation loop"""
    model.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float().unsqueeze(1)
            
            outputs = model(images)
            loss_focal = focal_criterion(outputs, masks)
            loss_dice = dice_criterion(outputs, masks)
            loss = 0.7 * loss_focal + 0.3 * loss_dice
            
            val_loss += loss.item()
            val_iou += calculate_iou(outputs, masks)
    
    return val_loss / len(val_loader), val_iou / len(val_loader)

def train():
    print(f"🚀 Starting Training on Device: {DEVICE}")
    
    # 1. Dataset
    full_dataset = EcoReviveDataset(DIRS)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"📊 Dataset: {len(full_dataset)} total, {train_size} train, {val_size} val")
    
    # 2. Loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Model
    model = UNet(n_channels=10, n_classes=1).to(DEVICE)
    print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 4. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 5. Loss
    focal_criterion = FocalLoss(alpha=0.8, gamma=2)
    dice_criterion = DiceLoss()
    
    # 6. Mixed Precision
    scaler = GradScaler() if DEVICE == "cuda" else None
    
    # 7. Training
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_iou = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float().unsqueeze(1)
            
            # Forward with mixed precision
            if scaler:
                with autocast():
                    outputs = model(images)
                    loss_focal = focal_criterion(outputs, masks)
                    loss_dice = dice_criterion(outputs, masks)
                    loss = 0.7 * loss_focal + 0.3 * loss_dice
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_focal = focal_criterion(outputs, masks)
                loss_dice = dice_criterion(outputs, masks)
                loss = 0.7 * loss_focal + 0.3 * loss_dice
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        val_loss, val_iou = validate(model, val_loader, focal_criterion, dice_criterion)
        scheduler.step(val_loss)
        
        # Logging
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f} | Train IoU: {train_iou/len(train_loader):.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
            }, 'ecorevive_best.pth')
            print(f"✅ Best model saved (Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'ecorevive_checkpoint_epoch{epoch+1}.pth')
    
    print("\n🎉 Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()