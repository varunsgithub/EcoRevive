"""
Model Architecture for California Fire Detection
Clean U-Net with optional attention gates.

Design principles:
1. Simple, well-tested architecture
2. Single output channel (regression: burn severity 0-1)
3. Optional attention gates (can be disabled)
4. Proper weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Standard double convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Helps the model focus on relevant features.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Features from decoder path (upsampled)
            skip: Features from encoder path (skip connection)
            
        Returns:
            Attention-weighted skip features
        """
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        
        attention = self.relu(g + s)
        attention = self.psi(attention)
        
        return skip * attention


class UNetEncoder(nn.Module):
    """U-Net encoder (downsampling path)."""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        c = base_channels
        
        self.inc = ConvBlock(in_channels, c)
        self.down1 = ConvBlock(c, c * 2)
        self.down2 = ConvBlock(c * 2, c * 4)
        self.down3 = ConvBlock(c * 4, c * 8, dropout=dropout)
        self.down4 = ConvBlock(c * 8, c * 8, dropout=dropout)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x1 = self.inc(x)                    # (B, c, 256, 256)
        x2 = self.down1(self.pool(x1))      # (B, c*2, 128, 128)
        x3 = self.down2(self.pool(x2))      # (B, c*4, 64, 64)
        x4 = self.down3(self.pool(x3))      # (B, c*8, 32, 32)
        x5 = self.down4(self.pool(x4))      # (B, c*8, 16, 16) - bottleneck
        
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    """U-Net decoder (upsampling path) with optional attention."""
    
    def __init__(
        self,
        base_channels: int = 64,
        use_attention: bool = True,
    ):
        super().__init__()
        
        c = base_channels
        self.use_attention = use_attention
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(c, c, 2, stride=2)
        
        # Convolution after concatenation
        self.conv1 = ConvBlock(c * 8 + c * 4, c * 4)  # skip(c*8) + up(c*4)
        self.conv2 = ConvBlock(c * 4 + c * 2, c * 2)  # skip(c*4) + up(c*2)
        self.conv3 = ConvBlock(c * 2 + c, c)          # skip(c*2) + up(c)
        self.conv4 = ConvBlock(c + c, c)              # skip(c) + up(c)
        
        # Attention gates (optional)
        if use_attention:
            self.att1 = AttentionGate(c * 4, c * 8, c * 4)
            self.att2 = AttentionGate(c * 2, c * 4, c * 2)
            self.att3 = AttentionGate(c, c * 2, c)
            self.att4 = AttentionGate(c, c, c // 2)
    
    def forward(self, x1, x2, x3, x4, x5):
        """
        Args:
            x1-x4: Skip connections from encoder
            x5: Bottleneck features
        """
        # Decoder stage 1
        d4 = self.up1(x5)
        if self.use_attention:
            x4 = self.att1(d4, x4)
        d4 = self.conv1(torch.cat([x4, d4], dim=1))
        
        # Decoder stage 2
        d3 = self.up2(d4)
        if self.use_attention:
            x3 = self.att2(d3, x3)
        d3 = self.conv2(torch.cat([x3, d3], dim=1))
        
        # Decoder stage 3
        d2 = self.up3(d3)
        if self.use_attention:
            x2 = self.att3(d2, x2)
        d2 = self.conv3(torch.cat([x2, d2], dim=1))
        
        # Decoder stage 4
        d1 = self.up4(d2)
        if self.use_attention:
            x1 = self.att4(d1, x1)
        d1 = self.conv4(torch.cat([x1, d1], dim=1))
        
        return d1


class CaliforniaFireModel(nn.Module):
    """
    U-Net model for California fire severity prediction.
    
    Predicts continuous burn severity (0-1) from Sentinel-2 imagery.
    """
    
    def __init__(
        self,
        input_channels: int = 10,
        output_channels: int = 1,
        base_channels: int = 64,
        use_attention: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.encoder = UNetEncoder(
            in_channels=input_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
        
        self.decoder = UNetDecoder(
            base_channels=base_channels,
            use_attention=use_attention,
        )
        
        # Final output layer
        self.output = nn.Conv2d(base_channels, output_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) with C=10 Sentinel-2 bands
            
        Returns:
            Logits tensor (B, 1, H, W) for burn severity
            Apply sigmoid for probability output
        """
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder
        features = self.decoder(x1, x2, x3, x4, x5)
        
        # Output
        logits = self.output(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction with sigmoid activation.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Probability tensor (B, 1, H, W) in range [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# ============================================================
# MODEL LOADING/SAVING
# ============================================================
def load_model(
    checkpoint_path: str,
    device: str = 'cpu',
    **model_kwargs,
) -> CaliforniaFireModel:
    """Load model from checkpoint."""
    model = CaliforniaFireModel(**model_kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def save_model(
    model: CaliforniaFireModel,
    path: str,
    epoch: int = 0,
    optimizer=None,
    scheduler=None,
    metrics: dict = None,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CaliforniaFireModel")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    model = CaliforniaFireModel(
        input_channels=10,
        output_channels=1,
        base_channels=64,
        use_attention=True,
    ).to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # Test forward pass
    x = torch.randn(2, 10, 256, 256).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    print("\n[OK] Model test passed!")
    print("=" * 60)
