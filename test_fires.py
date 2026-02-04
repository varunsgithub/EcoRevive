#!/usr/bin/env python3
"""
EcoRevive Model Test Script
============================
Downloads Sentinel-2 imagery from Earth Engine for the exact fire locations
shown in the training visualization (Dixie, Caldor, Camp fires) and runs
model inference with the black-to-golden colormap.

Usage:
    python test_fires.py
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
FIRE_MODEL_ROOT = PROJECT_ROOT / "California-Fire-Model"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FIRE_MODEL_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

# =============================================================================
# FIRE LOCATIONS (from config.py - exact coordinates used in training)
# =============================================================================
TEST_FIRES = {
    'dixie': {
        'name': 'Dixie Fire (2021)',
        'center': (40.05, -121.20),  # lat, lon
        'bbox': [-121.60, 39.75, -120.50, 40.55],  # west, south, east, north
        'post_fire_dates': ('2021-10-01', '2021-11-30'),
        'expected_fire_pct': 0.303,  # 30.3% from image
    },
    'caldor': {
        'name': 'Caldor Fire (2021)',
        'center': (38.78, -120.15),
        'bbox': [-120.50, 38.55, -119.80, 39.00],
        'post_fire_dates': ('2021-09-15', '2021-11-15'),
        'expected_fire_pct': 0.233,  # 23.3% from image
    },
    'camp': {
        'name': 'Camp Fire (2018)',
        'center': (39.76, -121.62),
        'bbox': [-121.90, 39.60, -121.30, 39.95],
        'post_fire_dates': ('2018-12-01', '2019-02-28'),
        'expected_fire_pct': 0.160,  # 16.0% from image
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
# EARTH ENGINE SETUP
# =============================================================================
def initialize_ee():
    """Initialize Earth Engine."""
    try:
        import ee
        try:
            ee.Initialize(project='hale-life-482914-r0')
            print("[OK] Earth Engine initialized")
            return True
        except Exception as e:
            print(f"Trying authentication: {e}")
            ee.Authenticate()
            ee.Initialize(project='hale-life-482914-r0')
            return True
    except Exception as e:
        print(f"[ERROR] Earth Engine failed: {e}")
        return False

def download_sentinel2_tile(bbox, start_date, end_date, tile_size=256):
    """
    Download a single Sentinel-2 tile from Earth Engine.
    
    Args:
        bbox: [west, south, east, north]
        start_date, end_date: date range strings
        tile_size: output image size
        
    Returns:
        numpy array (10, tile_size, tile_size)
    """
    import ee
    
    # Define region
    region = ee.Geometry.Rectangle(bbox)
    
    # Get Sentinel-2 Surface Reflectance
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']))
    
    # Get median composite
    image = collection.median()
    
    # Calculate scale to get desired tile size
    # bbox size in degrees -> meters -> scale
    lon_diff = bbox[2] - bbox[0]
    lat_diff = bbox[3] - bbox[1]
    avg_lat = (bbox[1] + bbox[3]) / 2
    
    # Approximate meters per degree
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 110540
    
    width_m = lon_diff * meters_per_deg_lon
    height_m = lat_diff * meters_per_deg_lat
    
    # Use scale that gives us ~256x256
    scale = max(width_m, height_m) / tile_size
    scale = max(scale, 10)  # Minimum 10m resolution
    
    print(f"   Region: {width_m/1000:.1f}km x {height_m/1000:.1f}km, scale: {scale:.1f}m/px")
    
    # Sample the image
    bands_data = []
    band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
    # Get data as numpy array
    try:
        # Use getRegion for faster download
        sample = image.sampleRectangle(region=region, defaultValue=0)
        
        for band in band_names:
            arr = np.array(sample.get(band).getInfo())
            bands_data.append(arr)
        
        # Stack and resize
        image_array = np.stack(bands_data, axis=0).astype(np.float32)
        
        # Resize to target size if needed
        if image_array.shape[1] != tile_size or image_array.shape[2] != tile_size:
            from PIL import Image
            resized = []
            for i in range(10):
                img = Image.fromarray(image_array[i])
                img = img.resize((tile_size, tile_size), Image.BILINEAR)
                resized.append(np.array(img))
            image_array = np.stack(resized, axis=0)
            
        return image_array
        
    except Exception as e:
        print(f"   [WARNING] sampleRectangle failed: {e}")
        print("   Trying reduceRegion method...")
        
        # Fallback: create grid and sample
        return download_sentinel2_grid(image, region, tile_size, scale)

def download_sentinel2_grid(image, region, tile_size, scale):
    """Fallback grid-based download."""
    import ee
    
    # Generate sample points in a grid
    coords = region.bounds().getInfo()['coordinates'][0]
    west, south = coords[0]
    east, north = coords[2]
    
    lons = np.linspace(west, east, tile_size)
    lats = np.linspace(north, south, tile_size)  # Note: north to south
    
    # Sample at grid points (this is slower but more reliable)
    points = []
    for lat in lats:
        for lon in lons:
            points.append([lon, lat])
    
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p)) for p in points])
    
    sampled = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=scale
    ).getInfo()
    
    # Parse results
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    image_array = np.zeros((10, tile_size, tile_size), dtype=np.float32)
    
    for idx, feat in enumerate(sampled['features']):
        i = idx // tile_size
        j = idx % tile_size
        props = feat['properties']
        for b, band in enumerate(bands):
            val = props.get(band, 0)
            image_array[b, i, j] = val if val is not None else 0
    
    return image_array

# =============================================================================
# MODEL LOADING & INFERENCE
# =============================================================================
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def load_model(checkpoint_path, device):
    """Load the California Fire Model."""
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
    image = np.clip(image, 0, 10000).astype(np.float32)
    normalized = np.zeros_like(image)
    for i in range(10):
        normalized[i] = (image[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    return normalized

def predict_severity(image, model, device, use_tta=True):
    """Run inference with optional test-time augmentation."""
    # Normalize
    image = normalize_image(image)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            preds = []
            # Original
            preds.append(torch.sigmoid(model(image_tensor)))
            # Horizontal flip
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(image_tensor, [3]))), [3]))
            # Vertical flip
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(image_tensor, [2]))), [2]))
            # 90 degree rotation
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(image_tensor, 1, [2, 3]))), -1, [2, 3]))
            severity = torch.stack(preds).mean(0)
        else:
            severity = torch.sigmoid(model(image_tensor))
        
        severity = severity.cpu().numpy()[0, 0]
    
    return severity

# =============================================================================
# VISUALIZATION (Black-to-Golden colormap)
# =============================================================================
def get_fire_colormap():
    """Create black-to-golden colormap matching training visualization."""
    colors = [
        (0.0, 0.0, 0.0),      # Black (0%)
        (0.4, 0.1, 0.0),      # Dark brown-red (25%)
        (0.7, 0.3, 0.0),      # Orange-red (50%)
        (0.9, 0.6, 0.0),      # Orange (75%)
        (1.0, 0.85, 0.0),     # Golden-yellow (100%)
    ]
    return LinearSegmentedColormap.from_list('fire_severity', colors)

def create_rgb(bands):
    """Create RGB composite from bands (B4, B3, B2 = R, G, B)."""
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)  # B4, B3, B2
    rgb = np.clip(rgb / 3000 * 2.5, 0, 1)
    return rgb

def visualize_results(results, output_path):
    """Create visualization matching the training image style."""
    n_fires = len(results)
    fig, axes = plt.subplots(n_fires, 4, figsize=(16, 4 * n_fires))
    
    if n_fires == 1:
        axes = [axes]
    
    cmap = get_fire_colormap()
    
    for i, (fire_key, data) in enumerate(results.items()):
        # RGB
        axes[i][0].imshow(data['rgb'])
        axes[i][0].set_title(f"{data['name']}\nRGB", fontsize=10)
        axes[i][0].axis('off')
        
        # Prediction (black-to-golden)
        im1 = axes[i][1].imshow(data['prediction'], cmap=cmap, vmin=0, vmax=1)
        fire_pct = (data['prediction'] > 0.3).mean() * 100
        axes[i][1].set_title(f"Prediction\n{fire_pct:.1f}% fire (max={data['prediction'].max():.2f})", fontsize=10)
        axes[i][1].axis('off')
        plt.colorbar(im1, ax=axes[i][1], fraction=0.046)
        
        # Binary at 0.3 threshold
        binary = (data['prediction'] > 0.3).astype(float)
        axes[i][2].imshow(binary, cmap='gray', vmin=0, vmax=1)
        axes[i][2].set_title(f"Binary @ 0.3\nExpected: {data['expected_pct']*100:.1f}%", fontsize=10)
        axes[i][2].axis('off')
        
        # Stats
        ax = axes[i][3]
        ax.axis('off')
        stats_text = f"""
Fire: {data['name']}
─────────────────
Mean Severity: {data['prediction'].mean():.1%}
Max Severity: {data['prediction'].max():.1%}
Fire (>0.3): {fire_pct:.1f}%
Expected: {data['expected_pct']*100:.1f}%
─────────────────
Match: {'[OK] Good' if abs(fire_pct/100 - data['expected_pct']) < 0.15 else '[WARNING] Off'}
"""
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                transform=ax.transAxes, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVE] Saved visualization: {output_path}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("[FIRE] EcoRevive Model Test - California Fires")
    print("=" * 60)
    
    # Initialize Earth Engine
    if not initialize_ee():
        print("[ERROR] Cannot proceed without Earth Engine")
        return
    
    # Load model
    device = get_device()
    print(f"\n[DEVICE] Device: {device}")
    
    checkpoint_path = FIRE_MODEL_ROOT / "checkpoints" / "model.pth"
    if not checkpoint_path.exists():
        print(f"[ERROR] Model not found at {checkpoint_path}")
        return
    
    print(f"[FILE] Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    print("[OK] Model loaded!")
    
    # Test each fire
    results = {}
    
    for fire_key, fire_info in TEST_FIRES.items():
        print(f"\n{'='*40}")
        print(f"[FIRE] Testing: {fire_info['name']}")
        print(f"   Center: {fire_info['center']}")
        print(f"   Dates: {fire_info['post_fire_dates']}")
        
        # Download from Earth Engine
        print("   [DOWNLOAD] Downloading Sentinel-2 imagery...")
        try:
            image = download_sentinel2_tile(
                bbox=fire_info['bbox'],
                start_date=fire_info['post_fire_dates'][0],
                end_date=fire_info['post_fire_dates'][1],
                tile_size=256
            )
            print(f"   [OK] Downloaded: {image.shape}")
            
            # Run inference
            print("   [FIRE] Running inference...")
            prediction = predict_severity(image, model, device)
            print(f"   [OK] Prediction complete: mean={prediction.mean():.2%}, max={prediction.max():.2%}")
            
            # Store results
            results[fire_key] = {
                'name': fire_info['name'],
                'image': image,
                'rgb': create_rgb(image),
                'prediction': prediction,
                'expected_pct': fire_info['expected_fire_pct'],
            }
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize
    if results:
        output_path = PROJECT_ROOT / "test_fires_results.png"
        visualize_results(results, output_path)
        
        # Summary
        print("\n" + "=" * 60)
        print("[SUMMARY] SUMMARY")
        print("=" * 60)
        for fire_key, data in results.items():
            fire_pct = (data['prediction'] > 0.3).mean() * 100
            expected = data['expected_pct'] * 100
            match = "[OK]" if abs(fire_pct - expected) < 15 else "[WARNING]"
            print(f"   {data['name']}: {fire_pct:.1f}% predicted vs {expected:.1f}% expected {match}")
    else:
        print("\n[ERROR] No results to display")

if __name__ == "__main__":
    main()
