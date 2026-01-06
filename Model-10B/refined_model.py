import torch
import torch.nn as nn
import torch.nn.functional as F
from model import UNet  # Import your existing model


class SpatialRefinementModule(nn.Module):
    """Lightweight spatial refinement - adds connectivity without heavy computation"""
    def __init__(self, channels=64):
        super().__init__()
        # Dilated convolutions for multi-scale context
        self.conv1 = nn.Conv2d(channels, channels//2, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(channels, channels//2, 3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(channels, channels//2, 3, padding=8, dilation=8)
        
        self.fusion = nn.Conv2d(channels + channels//2*3, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        d1 = F.relu(self.conv1(x))
        d2 = F.relu(self.conv2(x))
        d3 = F.relu(self.conv3(x))
        fused = torch.cat([x, d1, d2, d3], dim=1)
        out = self.relu(self.bn(self.fusion(fused)))
        return out + x  # Residual connection


class TemperatureScaling(nn.Module):
    """Learnable temperature for probability calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature


class RefinedEcoReviveModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.device = device

        # backbone + refinement (same as training)
        self.backbone = UNet(n_channels=10, n_classes=1, base_channels=64)
        self.spatial_refine = SpatialRefinementModule(channels=64)
        self.temperature = TemperatureScaling()
        self.final_conv = torch.nn.Conv2d(64, 1, kernel_size=1)

        self.to(device)

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(
        checkpoint_path,
        map_location=self.device,
        weights_only=False  # REQUIRED in PyTorch 2.6+
    )

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        print("✅ Weights loaded successfully")
        if missing:
            print("⚠️ Missing keys:", missing)
        if unexpected:
            print("⚠️ Unexpected keys:", unexpected)


    def forward(self, x):
        # SAME forward you trained with
        x1 = self.backbone.inc(x)
        x2 = self.backbone.down1(self.backbone.pool(x1))
        x3 = self.backbone.down2(self.backbone.pool(x2))
        x4 = self.backbone.down3(self.backbone.pool(x3))
        x5 = self.backbone.down4(self.backbone.pool(x4))

        d4 = self.backbone.up1(x5)
        d4 = self.backbone.conv1(torch.cat([self.backbone.att4(d4, x4), d4], 1))

        d3 = self.backbone.up2(d4)
        d3 = self.backbone.conv2(torch.cat([self.backbone.att3(d3, x3), d3], 1))

        d2 = self.backbone.up3(d3)
        d2 = self.backbone.conv3(torch.cat([self.backbone.att2(d2, x2), d2], 1))

        d1 = self.backbone.up4(d2)
        d1 = self.backbone.conv4(torch.cat([self.backbone.att1(d1, x1), d1], 1))

        d1 = self.spatial_refine(d1)
        logits = self.temperature(self.final_conv(d1))
        return logits

    
    def freeze_backbone(self):
        """Freeze backbone for initial fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze for full training"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ==================== ADVANCED LOSS FUNCTIONS ====================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes edges.
    FIXED: Properly handles device and dtype for Sobel kernels.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw logits from model (B, 1, H, W) - NOT sigmoidized
            targets: Ground truth masks (B, 1, H, W)
        """
        # CRITICAL FIX: Move Sobel kernels to same device AND dtype as targets
        sobel_x = self.sobel_x.to(device=targets.device, dtype=targets.dtype)
        sobel_y = self.sobel_y.to(device=targets.device, dtype=targets.dtype)
        
        # Compute target boundaries
        edge_x = F.conv2d(targets, sobel_x, padding=1)
        edge_y = F.conv2d(targets, sobel_y, padding=1)
        target_edges = torch.sqrt(edge_x**2 + edge_y**2)
        target_edges = (target_edges > 0.1).float()  # Binary edges
        
        # Use BCE with logits (autocast-safe)
        # This combines sigmoid + BCE in one operation
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Weight boundary pixels more
        weights = 1.0 + 2.0 * target_edges
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


class ConnectivityLoss(nn.Module):
    """Encourages spatially connected predictions"""
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        """
        Penalize isolated predictions by encouraging local agreement
        Uses 8-connected neighborhood
        """
        probs = torch.sigmoid(logits)
        
        # Average pooling to get neighborhood consensus
        kernel_size = 3
        neighborhood = F.avg_pool2d(probs, kernel_size, stride=1, padding=1)
        
        # Penalize disagreement between pixel and its neighborhood
        consistency = (probs - neighborhood).abs().mean()
        
        return consistency


class IoULoss(nn.Module):
    """Direct IoU optimization - backpropagates through IoU metric"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # IoU computation
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - IoU as loss (minimize)
        return 1 - iou


class FocalTverskyLoss(nn.Module):
    """
    Tversky loss with focal weighting - better for handling FN bias
    alpha controls FN penalty, beta controls FP penalty
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Higher = penalize FN more
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Tversky index
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        # Focal weighting
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple objectives.
    All components now work with autocast and mixed precision.
    """
    def __init__(self, iou_weight=1.0, tversky_weight=1.0, boundary_weight=0.5, connectivity_weight=0.3):
        super().__init__()
        self.iou_weight = iou_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.connectivity_weight = connectivity_weight
        
        self.boundary_loss = BoundaryLoss()
    
    def iou_loss(self, logits, targets):
        """IoU loss using logits"""
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou
    
    def tversky_loss(self, logits, targets, alpha=0.7, beta=0.3):
        """Tversky loss (focuses on false negatives)"""
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return 1 - tversky
    
    def connectivity_loss(self, logits, targets):
        """
        Penalize disconnected predictions.
        Uses morphological operations to check connectivity.
        """
        probs = torch.sigmoid(logits)
        
        # Max pooling to find connected regions
        pooled = F.max_pool2d(probs, kernel_size=3, stride=1, padding=1)
        
        # Connectivity penalty
        connectivity = torch.abs(probs - pooled).mean()
        return connectivity
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (B, 1, H, W) - NOT sigmoidized
            targets: Ground truth (B, 1, H, W)
        """
        loss_iou = self.iou_loss(logits, targets)
        loss_tversky = self.tversky_loss(logits, targets)
        loss_boundary = self.boundary_loss(logits, targets)
        loss_connectivity = self.connectivity_loss(logits, targets)
        
        total_loss = (
            self.iou_weight * loss_iou +
            self.tversky_weight * loss_tversky +
            self.boundary_weight * loss_boundary +
            self.connectivity_weight * loss_connectivity
        )
        
        # Return loss and components for logging
        components = {
            'iou': loss_iou.item(),
            'tversky': loss_tversky.item(),
            'boundary': loss_boundary.item(),
            'connectivity': loss_connectivity.item()
        }
        
        return total_loss, components