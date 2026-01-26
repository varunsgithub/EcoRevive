"""
Local Model Testing Script
Uses the actual band statistics from training
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import BANDS
from model.architecture import CaliforniaFireModel, load_model

# =============================================================================
# BAND STATISTICS FROM TRAINING (1000 tiles, ~63M pixels)
# =============================================================================
BAND_MEANS = np.array([
    472.8,   # B2 - Blue
    673.8,   # B3 - Green
    770.8,   # B4 - Red
    1087.8,  # B5 - Red Edge 1
    1747.6,  # B6 - Red Edge 2
    1997.1,  # B7 - Red Edge 3
    2106.4,  # B8 - NIR
    2188.9,  # B8A - NIR narrow
    1976.1,  # B11 - SWIR1
    1404.5,  # B12 - SWIR2
], dtype=np.float32)

BAND_STDS = np.array([
    223.7,   # B2
    255.4,   # B3
    345.6,   # B4
    313.5,   # B5
    366.7,   # B6
    417.6,   # B7
    476.7,   # B8
    437.1,   # B8A
    472.7,   # B11
    438.4,   # B12
], dtype=np.float32)

NUM_BANDS = 10
MODEL_PATH = Path('/Users/varunsingh/Desktop/Projects/EcoRevive/models/California/best_model.pth')

# =============================================================================
# MODEL CONFIG
# =============================================================================
MODEL_CONFIG = {
    'input_channels': 10,
    'output_channels': 1,
    'base_channels': 64,
    'use_attention': True,
    'dropout': 0.2,
}

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def normalize_image(image):
    """Normalize using training statistics."""
    image = np.clip(image, 0, 10000).astype(np.float32)
    for i in range(min(NUM_BANDS, image.shape[0])):
        image[i] = (image[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    image = np.clip(image, -3, 3)
    image = (image + 3) / 6
    return image

def predict_burn_severity(image, model, device, use_tta=True):
    """Predict burn severity with TTA."""
    image = image[:NUM_BANDS]
    image = np.nan_to_num(image, nan=0.0, posinf=10000.0, neginf=0.0)
    image = normalize_image(image)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            preds = []
            preds.append(torch.sigmoid(model(image_tensor)))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(image_tensor, [3]))), [3]))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(image_tensor, [2]))), [2]))
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(image_tensor, 1, [2, 3]))), -1, [2, 3]))
            severity = torch.stack(preds).mean(0)
        else:
            severity = torch.sigmoid(model(image_tensor))
        severity = severity.cpu().numpy()[0, 0]
    
    confidence = np.abs(severity - 0.5).mean() * 2
    return severity, confidence

def create_synthetic_healthy_forest(size=256):
    """Create synthetic healthy vegetation."""
    np.random.seed(42)
    base = {'B2': 400, 'B3': 600, 'B4': 350, 'B5': 1500, 'B6': 2800, 
            'B7': 3200, 'B8': 3500, 'B8A': 3600, 'B11': 1200, 'B12': 700}
    image = np.zeros((10, size, size), dtype=np.float32)
    for i, (band, val) in enumerate(base.items()):
        noise = np.random.normal(0, val * 0.1, (size, size))
        image[i] = np.clip(val + noise, 0, 10000)
    return image

def create_synthetic_burned(size=256, severity=0.7):
    """Create synthetic burned area."""
    np.random.seed(123)
    healthy = {'B2': 400, 'B3': 600, 'B4': 350, 'B5': 1500, 'B6': 2800, 
               'B7': 3200, 'B8': 3500, 'B8A': 3600, 'B11': 1200, 'B12': 700}
    burned = {'B2': 800, 'B3': 700, 'B4': 650, 'B5': 700, 'B6': 800, 
              'B7': 900, 'B8': 1000, 'B8A': 1100, 'B11': 2000, 'B12': 1500}
    
    burn_mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(20):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(20, 60)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        burn_mask += np.exp(-dist**2 / (2 * r**2)) * np.random.uniform(0.5, 1.0)
    burn_mask = np.clip(burn_mask, 0, 1) * severity
    
    image = np.zeros((10, size, size), dtype=np.float32)
    for i, band in enumerate(BANDS):
        base = healthy[band] * (1 - burn_mask) + burned[band] * burn_mask
        noise = np.random.normal(0, base.mean() * 0.05, (size, size))
        image[i] = np.clip(base + noise, 0, 10000)
    return image, burn_mask

def get_cmap():
    colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027', '#a50026']
    return LinearSegmentedColormap.from_list('severity', colors)

def main():
    print("=" * 60)
    print("🔥 LOCAL MODEL TESTING")
    print("=" * 60)
    
    device = get_device()
    print(f"\n🖥️ Device: {device}")
    print(f"📁 Model: {MODEL_PATH}")
    
    # Load model
    model = load_model(str(MODEL_PATH), device=device, **MODEL_CONFIG)
    model.eval()
    print("✅ Model loaded!")
    
    # Test synthetic healthy
    print("\n🌲 Testing SYNTHETIC HEALTHY FOREST...")
    healthy = create_synthetic_healthy_forest()
    sev_h, conf_h = predict_burn_severity(healthy, model, device)
    print(f"   Severity: {sev_h.mean():.1%} (expected: <20%)")
    print(f"   {'✅ PASS' if sev_h.mean() < 0.3 else '⚠️ HIGH'}")
    
    # Test synthetic burned
    print("\n🔥 Testing SYNTHETIC BURNED AREA...")
    burned, gt = create_synthetic_burned(severity=0.7)
    sev_b, conf_b = predict_burn_severity(burned, model, device)
    print(f"   Severity: {sev_b.mean():.1%} (expected: >50%)")
    print(f"   {'✅ PASS' if sev_b.mean() > 0.5 else '⚠️ LOW'}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = get_cmap()
    
    axes[0,0].imshow(np.clip(np.stack([healthy[2], healthy[1], healthy[0]], axis=-1)/3000*2.5, 0, 1))
    axes[0,0].set_title('Healthy Forest RGB')
    axes[0,0].axis('off')
    
    im1 = axes[0,1].imshow(sev_h, cmap=cmap, vmin=0, vmax=1)
    axes[0,1].set_title(f'Severity: {sev_h.mean():.1%}')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1])
    
    axes[1,0].imshow(np.clip(np.stack([burned[2], burned[1], burned[0]], axis=-1)/3000*2.5, 0, 1))
    axes[1,0].set_title('Burned Area RGB')
    axes[1,0].axis('off')
    
    im2 = axes[1,1].imshow(sev_b, cmap=cmap, vmin=0, vmax=1)
    axes[1,1].set_title(f'Severity: {sev_b.mean():.1%}')
    axes[1,1].axis('off')
    plt.colorbar(im2, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'test_outputs' / 'local_test_results.png', dpi=150)
    print(f"\n💾 Saved: {PROJECT_ROOT / 'test_outputs' / 'local_test_results.png'}")
    plt.show()

if __name__ == "__main__":
    main()
