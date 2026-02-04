#!/usr/bin/env python3
"""
EcoRevive Model Test Script - Final Version
=============================================
Downloads Sentinel-2 imagery from Earth Engine using the CORRECT method
(getDownloadURL with fixed dimensions) and tests the model.

Usage:
    python test_fires_final.py
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
FIRE_MODEL_ROOT = PROJECT_ROOT / "California-Fire-Model"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FIRE_MODEL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import io
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================
TILE_SIZE = 256  # pixels
TILE_SCALE = 10  # meters per pixel
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# =============================================================================
# TEST TILES - Specific locations within fire perimeters
# =============================================================================
TEST_TILES = {
    'dixie_center': {
        'name': 'Dixie Fire - Center',
        'center': (40.05, -121.20),  # lat, lon
        'post_fire_dates': ('2021-10-01', '2021-11-30'),
        'description': 'Core burn area of Dixie Fire (2021)',
    },
    'caldor_center': {
        'name': 'Caldor Fire - Center', 
        'center': (38.75, -120.20),
        'post_fire_dates': ('2021-09-15', '2021-11-15'),
        'description': 'Core burn area of Caldor Fire (2021)',
    },
    'camp_paradise': {
        'name': 'Camp Fire - Paradise',
        'center': (39.76, -121.62),  # Town of Paradise
        'post_fire_dates': ('2018-12-01', '2019-02-28'),
        'description': 'Town of Paradise - Camp Fire (2018)',
    },
    'healthy_tahoe': {
        'name': 'Lake Tahoe - Healthy',
        'center': (39.05, -120.05),
        'post_fire_dates': ('2023-07-01', '2023-08-31'),
        'description': 'Healthy forest reference near Lake Tahoe',
    },
}

# =============================================================================
# BAND STATISTICS (from training)
# =============================================================================
BAND_MEANS = np.array([
    472.8, 673.8, 770.8, 1087.8, 1747.6, 
    1997.1, 2106.4, 2188.9, 1976.1, 1404.5
], dtype=np.float32)

BAND_STDS = np.array([
    223.7, 255.4, 345.6, 313.5, 366.7,
    417.6, 476.7, 437.1, 472.7, 438.4
], dtype=np.float32)

# =============================================================================
# EARTH ENGINE DOWNLOAD
# =============================================================================
def initialize_ee():
    """Initialize Earth Engine."""
    try:
        import ee
        try:
            ee.Initialize(project='hale-life-482914-r0')
            print("[OK] Earth Engine initialized")
            return True
        except:
            ee.Authenticate()
            ee.Initialize(project='hale-life-482914-r0')
            return True
    except Exception as e:
        print(f"[ERROR] Earth Engine failed: {e}")
        return False

def download_tile(center, start_date, end_date):
    """
    Download a 256x256 tile at 10m resolution using getDownloadURL.
    
    Args:
        center: (lat, lon) tuple
        start_date, end_date: date strings
        
    Returns:
        numpy array (10, 256, 256)
    """
    import ee
    
    center_lat, center_lon = center
    
    # Create fixed-size region: 256 * 10m = 2560m = 2.56km
    # Convert to degrees (approximate)
    half_size_deg_lat = (TILE_SIZE * TILE_SCALE / 2) / 111000
    half_size_deg_lon = (TILE_SIZE * TILE_SCALE / 2) / (111000 * np.cos(np.radians(center_lat)))
    
    roi = ee.Geometry.Rectangle([
        center_lon - half_size_deg_lon,
        center_lat - half_size_deg_lat,
        center_lon + half_size_deg_lon,
        center_lat + half_size_deg_lat
    ])
    
    # Get Sentinel-2 composite
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(BANDS))
    
    count = collection.size().getInfo()
    print(f"   Found {count} images")
    
    if count == 0:
        raise ValueError("No Sentinel-2 images found")
    
    composite = collection.median().clip(roi)
    
    # Download with FIXED dimensions - this is the key!
    url = composite.getDownloadURL({
        'bands': BANDS,
        'region': roi,
        'dimensions': f'{TILE_SIZE}x{TILE_SIZE}',
        'format': 'NPY'
    })
    
    print(f"   Downloading 256x256 @ 10m...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    # Parse numpy array
    data = np.load(io.BytesIO(response.content), allow_pickle=True)
    
    # Handle structured array format from EE
    if isinstance(data, np.ndarray) and data.dtype.names:
        arrays = [data[band].astype(np.float32) for band in BANDS]
        image_array = np.stack(arrays, axis=0)
    else:
        image_array = data.astype(np.float32)
    
    print(f"   Shape: {image_array.shape}, range: {image_array.min():.0f} - {image_array.max():.0f}")
    
    return image_array

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
        input_channels=10,
        output_channels=1,
        base_channels=64,
        use_attention=True,
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

def normalize_image(image):
    """Normalize using training statistics."""
    normalized = np.zeros_like(image)
    for i in range(10):
        normalized[i] = (image[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    return normalized

def predict_severity(image, model, device, use_tta=True):
    """Run inference with TTA."""
    image = normalize_image(image)
    tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            preds = []
            preds.append(torch.sigmoid(model(tensor)))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(tensor, [3]))), [3]))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(tensor, [2]))), [2]))
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(tensor, 1, [2, 3]))), -1, [2, 3]))
            severity = torch.stack(preds).mean(0)
        else:
            severity = torch.sigmoid(model(tensor))
        
        return severity.cpu().numpy()[0, 0]

# =============================================================================
# VISUALIZATION
# =============================================================================
def get_fire_colormap():
    """Black-to-golden colormap."""
    colors = [
        (0.0, 0.0, 0.0),      # Black
        (0.4, 0.1, 0.0),      # Dark red
        (0.7, 0.3, 0.0),      # Orange-red
        (0.9, 0.6, 0.0),      # Orange
        (1.0, 0.85, 0.0),     # Golden-yellow
    ]
    return LinearSegmentedColormap.from_list('fire', colors)

def create_rgb(bands):
    """RGB from B4, B3, B2."""
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    return np.clip(rgb / 2500 * 2.0, 0, 1)

def visualize_results(results, output_path):
    """Create visualization."""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    
    cmap = get_fire_colormap()
    
    for i, (key, data) in enumerate(results.items()):
        # RGB
        axes[i][0].imshow(data['rgb'])
        axes[i][0].set_title(f"{data['name']}\nRGB", fontsize=10)
        axes[i][0].axis('off')
        
        # Prediction
        im = axes[i][1].imshow(data['prediction'], cmap=cmap, vmin=0, vmax=1)
        pct = (data['prediction'] > 0.3).mean() * 100
        axes[i][1].set_title(f"Prediction\nmax={data['prediction'].max():.2f}", fontsize=10)
        axes[i][1].axis('off')
        plt.colorbar(im, ax=axes[i][1], fraction=0.046)
        
        # Binary
        axes[i][2].imshow(data['prediction'] > 0.3, cmap='gray')
        axes[i][2].set_title(f"Binary @ 0.3\n{pct:.1f}% fire", fontsize=10)
        axes[i][2].axis('off')
        
        # Stats
        ax = axes[i][3]
        ax.axis('off')
        ax.text(0.1, 0.5, f"""
{data['name']}
{data['description']}
Center: {data['center']}

Mean: {data['prediction'].mean():.1%}
Max: {data['prediction'].max():.1%}
Fire (>30%): {pct:.1f}%
High (>70%): {(data['prediction'] > 0.7).mean()*100:.1f}%
""", fontsize=10, transform=ax.transAxes, va='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVE] Saved: {output_path}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("[FIRE] EcoRevive Model Test - Using getDownloadURL")
    print("=" * 60)
    
    if not initialize_ee():
        return
    
    device = get_device()
    print(f"\n[DEVICE] Device: {device}")
    
    checkpoint = FIRE_MODEL_ROOT / "checkpoints" / "model.pth"
    if not checkpoint.exists():
        print(f"[ERROR] Model not found: {checkpoint}")
        return
    
    model = load_model(checkpoint, device)
    print("[OK] Model loaded!")
    
    results = {}
    
    for key, info in TEST_TILES.items():
        print(f"\n{'='*50}")
        print(f"[LOCATION] {info['name']}")
        print(f"   Center: {info['center']}")
        print(f"   {info['description']}")
        
        try:
            image = download_tile(
                center=info['center'],
                start_date=info['post_fire_dates'][0],
                end_date=info['post_fire_dates'][1]
            )
            
            prediction = predict_severity(image, model, device)
            pct = (prediction > 0.3).mean() * 100
            print(f"   [OK] Prediction: mean={prediction.mean():.1%}, max={prediction.max():.1%}, fire={pct:.1f}%")
            
            results[key] = {
                'name': info['name'],
                'center': info['center'],
                'description': info['description'],
                'image': image,
                'rgb': create_rgb(image),
                'prediction': prediction,
            }
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        output_path = PROJECT_ROOT / "test_fires_final_results.png"
        visualize_results(results, output_path)
        
        print("\n" + "=" * 60)
        print("[SUMMARY] SUMMARY")
        print("=" * 60)
        for key, data in results.items():
            pct = (data['prediction'] > 0.3).mean() * 100
            high_pct = (data['prediction'] > 0.7).mean() * 100
            status = "[FIRE]" if pct > 20 else "[TREE]"
            print(f"   {status} {data['name']}: {pct:.1f}% fire, {high_pct:.1f}% high severity")

if __name__ == "__main__":
    main()
