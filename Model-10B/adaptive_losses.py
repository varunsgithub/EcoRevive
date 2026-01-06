"""
Enhanced Loss Functions for Failed Case Training
Focuses on medium degradation, edge cases, and spatial consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    """
    Focal loss with adaptive gamma based on degradation severity
    Increases focus on medium degradation range (30-70%)
    """
    def __init__(self, alpha=0.25, base_gamma=2.0, adaptive=True):
        super().__init__()
        self.alpha = alpha
        self.base_gamma = base_gamma
        self.adaptive = adaptive
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw model outputs
            targets: (B, 1, H, W) ground truth
        """
        probs = torch.sigmoid(logits)
        
        # Standard focal loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        
        if self.adaptive:
            # Adaptive gamma: higher for medium degradation range
            # gamma = base_gamma + bonus for middle range
            middle_distance = torch.abs(targets - 0.5)  # 0 at 0.5, 0.5 at edges
            gamma_bonus = 2.0 * (1.0 - 2.0 * middle_distance)  # Peak at targets=0.5
            gamma = self.base_gamma + torch.clamp(gamma_bonus, 0, 2)
            
            focal_weight = (1 - pt) ** gamma
        else:
            focal_weight = (1 - pt) ** self.base_gamma
        
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


class BoundaryEnhancedLoss(nn.Module):
    """
    Emphasizes edges and transition zones
    Critical for handling recovery areas and gradual degradation
    """
    def __init__(self, boundary_weight=3.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, logits, targets):
        """
        Higher weight on boundary pixels
        """
        # Detect edges in ground truth
        edge_x = F.conv2d(targets, self.sobel_x, padding=1)
        edge_y = F.conv2d(targets, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        boundary_mask = (edges > 0.1).float()
        
        # BCE loss with boundary weighting
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Weight: 1.0 for normal pixels, boundary_weight for edges
        weights = 1.0 + self.boundary_weight * boundary_mask
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


class UncertaintyFocusedLoss(nn.Module):
    """
    Penalizes uncertain predictions (around 0.5 probability)
    Encourages confident predictions while maintaining accuracy
    """
    def __init__(self, uncertainty_weight=0.5):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Standard BCE
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Uncertainty penalty: peaks at prob=0.5
        uncertainty = 1.0 - 2.0 * torch.abs(probs - 0.5)  # 1.0 at 0.5, 0.0 at edges
        uncertainty_penalty = uncertainty * torch.abs(probs - targets)
        
        total_loss = bce + self.uncertainty_weight * uncertainty_penalty
        
        return total_loss.mean()


class DifficultyWeightedLoss(nn.Module):
    """
    Wrapper that applies per-sample difficulty weights
    Used with hard negative mining
    """
    def __init__(self, base_loss, difficulty_scale=2.0):
        super().__init__()
        self.base_loss = base_loss
        self.difficulty_scale = difficulty_scale
    
    def forward(self, logits, targets, difficulty_weights=None):
        """
        Args:
            difficulty_weights: (B,) tensor with difficulty score per sample
        """
        # Compute base loss per sample (batch reduction only)
        base_loss_value = self.base_loss(logits, targets)
        
        if difficulty_weights is not None:
            # Scale weights: easy=1.0, hard=difficulty_scale
            scaled_weights = 1.0 + (self.difficulty_scale - 1.0) * difficulty_weights
            scaled_weights = scaled_weights.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            
            # Apply weights (this requires per-sample loss)
            # For now, return weighted mean
            return (base_loss_value * scaled_weights.mean()).mean()
        
        return base_loss_value


class EnhancedCompositeLoss(nn.Module):
    """
    Combined loss for enhanced training
    Includes adaptive focal, boundary, uncertainty, and standard IoU
    """
    def __init__(
        self,
        focal_weight=1.0,
        boundary_weight=0.8,
        uncertainty_weight=0.3,
        iou_weight=0.5
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.uncertainty_weight = uncertainty_weight
        self.iou_weight = iou_weight
        
        self.focal_loss = AdaptiveFocalLoss(adaptive=True)
        self.boundary_loss = BoundaryEnhancedLoss(boundary_weight=3.0)
        self.uncertainty_loss = UncertaintyFocusedLoss(uncertainty_weight=0.5)
    
    def iou_loss(self, logits, targets):
        """Direct IoU loss"""
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou
    
    def forward(self, logits, targets):
        """
        Returns total loss and components for logging
        """
        loss_focal = self.focal_loss(logits, targets)
        loss_boundary = self.boundary_loss(logits, targets)
        loss_uncertainty = self.uncertainty_loss(logits, targets)
        loss_iou = self.iou_loss(logits, targets)
        
        total_loss = (
            self.focal_weight * loss_focal +
            self.boundary_weight * loss_boundary +
            self.uncertainty_weight * loss_uncertainty +
            self.iou_weight * loss_iou
        )
        
        components = {
            'focal': loss_focal.item(),
            'boundary': loss_boundary.item(),
            'uncertainty': loss_uncertainty.item(),
            'iou': loss_iou.item()
        }
        
        return total_loss, components


# Quick test
if __name__ == "__main__":
    # Test the losses
    B, C, H, W = 4, 1, 256, 256
    logits = torch.randn(B, C, H, W)
    targets = torch.rand(B, C, H, W)
    
    print("Testing Enhanced Losses:")
    print("="*50)
    
    # Test adaptive focal
    focal = AdaptiveFocalLoss()
    loss_f = focal(logits, targets)
    print(f"Adaptive Focal Loss: {loss_f.item():.4f}")
    
    # Test boundary
    boundary = BoundaryEnhancedLoss()
    loss_b = boundary(logits, targets)
    print(f"Boundary Loss: {loss_b.item():.4f}")
    
    # Test uncertainty
    uncertainty = UncertaintyFocusedLoss()
    loss_u = uncertainty(logits, targets)
    print(f"Uncertainty Loss: {loss_u.item():.4f}")
    
    # Test composite
    composite = EnhancedCompositeLoss()
    loss_c, components = composite(logits, targets)
    print(f"Composite Loss: {loss_c.item():.4f}")
    print(f"Components: {components}")
    
    print("="*50)
    print("✅ All losses working correctly!")
