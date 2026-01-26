#!/usr/bin/env python3
"""
EcoRevive Model Test - EXACT Training Config
=============================================
Uses the EXACT same normalization and visualization as training:
- Normalization: (image - mean) / std, clip [-3, 3], then (image + 3) / 6
- Colormap: 'hot' (black -> red -> yellow -> white)
- RGB: False color B5, B4, B3 (bands [3, 2, 1])

Usage:
    python test_fires_exact.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FIRE_MODEL_ROOT = PROJECT_ROOT / "California-Fire-Model"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FIRE_MODEL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import requests

# =============================================================================
# EXACT TRAINING CONFIG
# =============================================================================
TILE_SIZE = 256
TILE_SCALE = 10
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Band statistics from training (1000 tiles, ~63M pixels)
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

# =============================================================================
# TEST TILES - Fire locations
# =============================================================================
TEST_TILES = {
    'dixie': {
        'name': 'Dixie Fire (2021)',
        'center': (40.05, -121.20),
        'dates': ('2021-10-01', '2021-11-30'),
    },
    'caldor': {
        'name': 'Caldor Fire (2021)', 
        'center': (38.75, -120.20),
        'dates': ('2021-09-15', '2021-11-15'),
    },
    'camp': {
        'name': 'Camp Fire (2018)',
        'center': (39.76, -121.62),
        'dates': ('2018-12-01', '2019-02-28'),
    },
}

# =============================================================================
# EARTH ENGINE
# =============================================================================
def initialize_ee():
    try:
        import ee
        try:
            ee.Initialize(project='hale-life-482914-r0')
            print("✅ Earth Engine initialized")
            return True
        except:
            ee.Authenticate()
            ee.Initialize(project='hale-life-482914-r0')
            return True
    except Exception as e:
        print(f"❌ Earth Engine failed: {e}")
        return False

def download_tile(center, start_date, end_date):
    """Download 256x256 tile at 10m resolution."""
    import ee
    
    center_lat, center_lon = center
    half_deg_lat = (TILE_SIZE * TILE_SCALE / 2) / 111000
    half_deg_lon = (TILE_SIZE * TILE_SCALE / 2) / (111000 * np.cos(np.radians(center_lat)))
    
    roi = ee.Geometry.Rectangle([
        center_lon - half_deg_lon, center_lat - half_deg_lat,
        center_lon + half_deg_lon, center_lat + half_deg_lat
    ])
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(BANDS))
    
    count = collection.size().getInfo()
    print(f"   Found {count} images")
    
    composite = collection.median().clip(roi)
    
    url = composite.getDownloadURL({
        'bands': BANDS,
        'region': roi,
        'dimensions': f'{TILE_SIZE}x{TILE_SIZE}',
        'format': 'NPY'
    })
    
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    data = np.load(io.BytesIO(response.content), allow_pickle=True)
    
    if isinstance(data, np.ndarray) and data.dtype.names:
        arrays = [data[band].astype(np.float32) for band in BANDS]
        image = np.stack(arrays, axis=0)
    else:
        image = data.astype(np.float32)
    
    print(f"   Downloaded: {image.shape}, range: {image.min():.0f} - {image.max():.0f}")
    return image

# =============================================================================
# EXACT TRAINING NORMALIZATION
# =============================================================================
def normalize_exact(image):
    """
    EXACT normalization from training:
    1. Replace NaN with 0
    2. (image - mean) / std
    3. Clip to [-3, 3]
    4. Scale to [0, 1] via (x + 3) / 6
    """
    image = np.nan_to_num(image, nan=0.0)
    
    normalized = np.zeros_like(image)
    for i in range(10):
        normalized[i] = (image[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    
    # Clip to [-3, 3]
    normalized = np.clip(normalized, -3, 3)
    
    # Scale to [0, 1]
    normalized = (normalized + 3) / 6
    
    return normalized

# =============================================================================
# MODEL
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def load_model(checkpoint_path, device):
    from model.architecture import CaliforniaFireModel
    
    model = CaliforniaFireModel(
        input_channels=10, output_channels=1,
        base_channels=64, use_attention=True,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def predict(image_normalized, model, device):
    """Predict using normalized image."""
    x = torch.from_numpy(image_normalized).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return probs

# =============================================================================
# VISUALIZATION - EXACT TRAINING CONFIG
# =============================================================================
def create_rgb_exact(image_normalized):
    """
    EXACT RGB from training: bands [3, 2, 1] = B5, B4, B3 (false color)
    """
    rgb = np.stack([
        image_normalized[3],  # B5 - Red Edge 1 -> R
        image_normalized[2],  # B4 - Red -> G
        image_normalized[1],  # B3 - Green -> B
    ], axis=-1)
    return np.clip(rgb, 0, 1)

def visualize_results(results, output_path):
    """
    EXACT visualization from training with 'hot' colormap.
    """
    n_fires = len(results)
    fig, axes = plt.subplots(n_fires, 4, figsize=(16, 4 * n_fires))
    
    if n_fires == 1:
        axes = [axes]
    
    for i, (fire_key, data) in enumerate(results.items()):
        probs = data['prediction']
        
        # RGB - exact training config
        axes[i][0].imshow(data['rgb'])
        axes[i][0].set_title(f"{data['name']}\nRGB", fontsize=11)
        axes[i][0].axis('off')
        
        # Ground Truth placeholder (we don't have GT from EE)
        # Show raw prediction instead
        fire_pct = (probs > 0.3).mean() * 100
        im1 = axes[i][1].imshow(probs, cmap='hot', vmin=0, vmax=1)
        axes[i][1].set_title(f"Prediction\nmax={probs.max():.2f}", fontsize=11)
        axes[i][1].axis('off')
        plt.colorbar(im1, ax=axes[i][1], fraction=0.046)
        
        # Same prediction with different colorbar for comparison
        im2 = axes[i][2].imshow(probs, cmap='hot', vmin=0, vmax=1)
        axes[i][2].set_title(f"Severity Map\nmean={probs.mean():.2f}", fontsize=11)
        axes[i][2].axis('off')
        
        # Binary at 0.3 threshold - exact training config
        binary = probs > 0.3
        axes[i][3].imshow(binary, cmap='gray')
        axes[i][3].set_title(f"Binary @ 0.3\n{fire_pct:.1f}% fire", fontsize=11)
        axes[i][3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {output_path}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("🔥 EcoRevive Test - EXACT Training Config")
    print("=" * 60)
    print("Using:")
    print("  - Normalization: (x-mean)/std, clip[-3,3], (x+3)/6")
    print("  - Colormap: 'hot' (black→red→yellow→white)")
    print("  - RGB: False color B5,B4,B3")
    print("=" * 60)
    
    if not initialize_ee():
        return
    
    device = get_device()
    print(f"\n🖥️ Device: {device}")
    
    checkpoint = FIRE_MODEL_ROOT / "checkpoints" / "model.pth"
    if not checkpoint.exists():
        print(f"❌ Model not found: {checkpoint}")
        return
    
    model = load_model(checkpoint, device)
    print("✅ Model loaded!")
    
    results = {}
    
    for fire_key, info in TEST_TILES.items():
        print(f"\n{'='*50}")
        print(f"📍 {info['name']}")
        print(f"   Center: {info['center']}")
        
        try:
            # Download raw image
            raw_image = download_tile(
                center=info['center'],
                start_date=info['dates'][0],
                end_date=info['dates'][1]
            )
            
            # Normalize EXACTLY like training
            normalized = normalize_exact(raw_image)
            
            # Predict
            prediction = predict(normalized, model, device)
            
            fire_pct = (prediction > 0.3).mean() * 100
            print(f"   ✅ Result: mean={prediction.mean():.2f}, max={prediction.max():.2f}, fire={fire_pct:.1f}%")
            
            results[fire_key] = {
                'name': info['name'],
                'raw_image': raw_image,
                'normalized': normalized,
                'rgb': create_rgb_exact(normalized),
                'prediction': prediction,
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        output_path = PROJECT_ROOT / "test_fires_exact_results.png"
        visualize_results(results, output_path)
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        for key, data in results.items():
            pct = (data['prediction'] > 0.3).mean() * 100
            high = (data['prediction'] > 0.7).mean() * 100
            print(f"   {data['name']}: {pct:.1f}% fire, {high:.1f}% high severity")

if __name__ == "__main__":
    main()
