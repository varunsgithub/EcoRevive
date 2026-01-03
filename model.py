import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True), # Sharp boundaries for segmentation

            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True) #ReLU is better for image segmentation !
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        
        # We use slight dropout in deeper layers to prevent overfitting in the deep layers.
        self.down3 = DoubleConv(256, 512, dropout_p=0.2)
        self.down4 = DoubleConv(512, 512, dropout_p=0.3) 
        
        self.pool = nn.MaxPool2d(2)

        # --- DECODER (Upsampling) ---
        # Up 1: Input 512. Output 256. 
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Skip connection from down3 is 512. So 256 + 512 = 768 inputs.
        self.conv1 = DoubleConv(768, 256) 
        
        # Up 2: Input 256. Output 128.
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Skip from down2 is 256. So 128 + 256 = 384 inputs.
        self.conv2 = DoubleConv(384, 128)
        
        # Up 3: Input 128. Output 64.
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Skip from down1 is 128. So 64 + 128 = 192 inputs.
        self.conv3 = DoubleConv(192, 64)
        
        # Up 4: Input 64. Output 64.
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # Skip from inc is 64. So 64 + 64 = 128 inputs.
        self.conv4 = DoubleConv(128, 64)

        # Output Layer (1x1 Conv)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))

        # Decoder
        x = self.up1(x5)
        # We use strict concatenation. 
        x = torch.cat([x4, x], dim=1) 
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits


##Loss Functions

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: (Batch, 1, H, W) -> Raw Logits
        # targets: (Batch, 1, H, W) -> Binary 0/1
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        # Apply Sigmoid to turn logits into 0-1 probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice