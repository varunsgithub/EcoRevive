"""
Loss Functions for California Fire Model
Simple, focused losses for burn severity regression.

Design philosophy:
1. Start simple (MSE + BCE)
2. Add complexity only if needed
3. Focus on regression (continuous 0-1 target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Simple Mean Squared Error for regression."""
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return F.mse_loss(probs, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error - more robust to outliers."""
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return F.l1_loss(probs, targets)


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with optional positive class weighting.
    Good for imbalanced datasets (more unburned than burned pixels).
    """
    
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Create position weight tensor
        if self.pos_weight != 1.0:
            weight = torch.ones_like(targets)
            weight[targets > 0.5] = self.pos_weight
            return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        else:
            return F.binary_cross_entropy_with_logits(logits, targets)


class SmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss - combines MSE and MAE benefits."""
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return F.smooth_l1_loss(probs, targets, beta=self.beta)


class DiceLoss(nn.Module):
    """
    Dice loss for better overlap optimization.
    Works well for segmentation-like tasks.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """
    Intersection over Union loss.
    Direct optimization of the IoU metric.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # IoU
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
    """
    Simple combined loss: BCE + Dice.
    This is the recommended starting point.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 2.0,  # Weight burned pixels more
    ):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            total_loss: Combined loss value
            components: Dict with individual loss values
        """
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        
        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        
        components = {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, components


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples.
    
    Only use if CombinedLoss isn't working well.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        focal_weight = (1 - pt) ** self.gamma
        
        loss = self.alpha * focal_weight * bce
        
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss with controllable FN/FP trade-off.
    alpha > beta: penalize false negatives more (better recall)
    beta > alpha: penalize false positives more (better precision)
    
    Only use if you need to tune precision/recall trade-off.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # FN weight
        beta: float = 0.3,   # FP weight
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false negatives, false positives
        tp = (probs_flat * targets_flat).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


# ============================================================
# LOSS FACTORY
# ============================================================
def get_loss_function(name: str = 'combined', **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        name: 'mse', 'mae', 'bce', 'dice', 'iou', 'combined', 'focal', 'tversky'
        **kwargs: Arguments for the loss function
        
    Returns:
        Loss function module
    """
    losses = {
        'mse': MSELoss,
        'mae': MAELoss,
        'bce': BCEWithLogitsLoss,
        'smooth_l1': SmoothL1Loss,
        'dice': DiceLoss,
        'iou': IoULoss,
        'combined': CombinedLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
    }
    
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name](**kwargs)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.rand(2, 1, 64, 64)
    
    losses_to_test = ['mse', 'mae', 'bce', 'dice', 'iou', 'combined', 'focal', 'tversky']
    
    print(f"\nInput shapes: logits={logits.shape}, targets={targets.shape}")
    print("\nLoss values:")
    print("-" * 40)
    
    for name in losses_to_test:
        loss_fn = get_loss_function(name)
        
        result = loss_fn(logits, targets)
        
        if isinstance(result, tuple):
            loss_val, components = result
            print(f"   {name:<12}: {loss_val.item():.4f}")
        else:
            print(f"   {name:<12}: {result.item():.4f}")
    
    print("\n[OK] All loss functions working!")
    print("=" * 60)
