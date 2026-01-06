import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Enhanced U-Net with Attention Gates and Deeper Architecture"""
    def __init__(self, n_channels, n_classes, base_channels=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = DoubleConv(base_channels, base_channels * 2)
        self.down2 = DoubleConv(base_channels * 2, base_channels * 4)
        self.down3 = DoubleConv(base_channels * 4, base_channels * 8, dropout_p=0.2)
        self.down4 = DoubleConv(base_channels * 8, base_channels * 8, dropout_p=0.3)
        
        self.pool = nn.MaxPool2d(2)

        # Attention gates - FIXED: Match the upsampled channels, not bottleneck channels
        self.att4 = AttentionBlock(F_g=base_channels * 4, F_l=base_channels * 8, F_int=base_channels * 4)
        self.att3 = AttentionBlock(F_g=base_channels * 2, F_l=base_channels * 4, F_int=base_channels * 2)
        self.att2 = AttentionBlock(F_g=base_channels, F_l=base_channels * 2, F_int=base_channels)
        self.att1 = AttentionBlock(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 12, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 6, base_channels * 2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 3, base_channels)
        
        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 2, base_channels)

        # Output with deep supervision
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
        # Deep supervision outputs (optional, can improve training)
        self.out_ds1 = nn.Conv2d(base_channels * 4, n_classes, kernel_size=1)
        self.out_ds2 = nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)

    def forward(self, x, return_deep_supervision=False):
        # Encoder
        x1 = self.inc(x)          # 64 channels
        x2 = self.down1(self.pool(x1))  # 128 channels
        x3 = self.down2(self.pool(x2))  # 256 channels
        x4 = self.down3(self.pool(x3))  # 512 channels
        x5 = self.down4(self.pool(x4))  # 512 channels (bottleneck)

        # Decoder with attention
        d4 = self.up1(x5)  # 512 -> 256 channels after upsampling
        x4_att = self.att4(g=d4, x=x4)  # d4 is 256ch (F_g), x4 is 512ch (F_l)
        d4 = torch.cat([x4_att, d4], dim=1)  # 512 + 256 = 768
        d4 = self.conv1(d4)  # 768 -> 256
        
        d3 = self.up2(d4)  # 256 -> 128 channels after upsampling
        x3_att = self.att3(g=d3, x=x3)  # d3 is 128ch (F_g), x3 is 256ch (F_l)
        d3 = torch.cat([x3_att, d3], dim=1)  # 256 + 128 = 384
        d3 = self.conv2(d3)  # 384 -> 128
        
        d2 = self.up3(d3)  # 128 -> 64 channels after upsampling
        x2_att = self.att2(g=d2, x=x2)  # d2 is 64ch (F_g), x2 is 128ch (F_l)
        d2 = torch.cat([x2_att, d2], dim=1)  # 128 + 64 = 192
        d2 = self.conv3(d2)  # 192 -> 64
        
        d1 = self.up4(d2)  # 64 -> 64 channels after upsampling
        x1_att = self.att1(g=d1, x=x1)  # d1 is 64ch (F_g), x1 is 64ch (F_l)
        d1 = torch.cat([x1_att, d1], dim=1)  # 64 + 64 = 128
        d1 = self.conv4(d1)  # 128 -> 64

        logits = self.outc(d1)
        
        if return_deep_supervision and self.training:
            # Deep supervision outputs (upsampled to match main output)
            ds1 = F.interpolate(self.out_ds1(d4), size=logits.shape[2:], mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.out_ds2(d3), size=logits.shape[2:], mode='bilinear', align_corners=False)
            return logits, ds1, ds2
        
        return logits


## Loss Functions

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(pos_weight, torch.Tensor):
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        pos_weight = self.pos_weight
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Positive class Dice
        inter_pos = (inputs * targets).sum()
        union_pos = inputs.sum() + targets.sum()
        dice_pos = (2 * inter_pos + self.smooth) / (union_pos + self.smooth)

        # Negative class Dice
        inputs_neg = 1 - inputs
        targets_neg = 1 - targets
        inter_neg = (inputs_neg * targets_neg).sum()
        union_neg = inputs_neg.sum() + targets_neg.sum()
        dice_neg = (2 * inter_neg + self.smooth) / (union_neg + self.smooth)

        return 1 - 0.5 * (dice_pos + dice_neg)


class CombinedLoss(nn.Module):
    """Combined loss with optimal weighting"""
    def __init__(self, focal_weight=0.4, dice_weight=0.6, pos_weight=None):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        return self.focal_weight * loss_focal + self.dice_weight * loss_dice