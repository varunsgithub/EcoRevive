#!/usr/bin/env python3
"""
EcoRevive Model Test Script - Corrected Version
=================================================
Uses properly-sized tiles (~2.5km at 10m resolution) centered on fire areas.
This matches the tile size used during model training.

Usage:
    python test_fires_v2.py
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
# TILE SIZE - Match training configuration
# =============================================================================
TILE_SIZE = 256  # pixels
TILE_SCALE = 10  # meters per pixel (Sentinel-2 native resolution)
# This gives us 2.56km x 2.56km tiles

# =============================================================================
# FIRE LOCATIONS - Using specific tile centers within fire perimeters
# These are approximate centers of heavily burned areas
# =============================================================================
TEST_TILES = {
    'dixie_center': {
        'name': 'Dixie Fire - Center',
        'center': (40.05, -121.20),  # lat, lon - center of fire
        'post_fire_dates': ('2021-10-01', '2021-11-30'),
        'description': 'Core burn area of Dixie Fire',
    },
    'caldor_center': {
        'name': 'Caldor Fire - Center',
        'center': (38.75, -120.20),
        'post_fire_dates': ('2021-09-15', '2021-11-15'),
        'description': 'Core burn area of Caldor Fire',
    },
    'camp_paradise': {
        'name': 'Camp Fire - Paradise',
        'center': (39.76, -121.62),  # Town of Paradise
        'post_fire_dates': ('2018-12-01', '2019-02-28'),
        'description': 'Paradise area devastated by Camp Fire',
    },
    # Additional test tiles for validation
    'dixie_edge': {
        'name': 'Dixie Fire - Edge',
        'center': (40.20, -121.40),
        'post_fire_dates': ('2021-10-01', '2021-11-30'),
        'description': 'Edge of Dixie Fire perimeter',
    },
    'healthy_tahoe': {
        'name': 'Lake Tahoe - Healthy',
        'center': (39.05, -120.05),  # Healthy forest near Tahoe
        'post_fire_dates': ('2023-07-01', '2023-08-31'),
        'description': 'Healthy reference area near Lake Tahoe',
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
            print("✅ Earth Engine initialized")
            return True
        except Exception as e:
            print(f"Trying authentication: {e}")
            ee.Authenticate()
            ee.Initialize(project='hale-life-482914-r0')
            return True
    except Exception as e:
        print(f"❌ Earth Engine failed: {e}")
        return False

def center_to_bbox(center_lat, center_lon, size_m=2560):
    """
    Convert center point to bounding box.
    
    Args:
        center_lat, center_lon: Center coordinates
        size_m: Size in meters (default 2560m = 256px * 10m)
        
    Returns:
        [west, south, east, north]
    """
    # Approximate degrees per meter at this latitude
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
    
    half_size_deg_lat = (size_m / 2) / meters_per_deg_lat
    half_size_deg_lon = (size_m / 2) / meters_per_deg_lon
    
    return [
        center_lon - half_size_deg_lon,  # west
        center_lat - half_size_deg_lat,  # south
        center_lon + half_size_deg_lon,  # east
        center_lat + half_size_deg_lat,  # north
    ]

def download_sentinel2_tile(center, start_date, end_date):
    """
    Download a 256x256 Sentinel-2 tile at 10m resolution.
    
    Args:
        center: (lat, lon) tuple
        start_date, end_date: date range strings
        
    Returns:
        numpy array (10, 256, 256)
    """
    import ee
    
    # Calculate bounding box for 2.56km tile
    bbox = center_to_bbox(center[0], center[1], size_m=TILE_SIZE * TILE_SCALE)
    
    # Define region
    region = ee.Geometry.Rectangle(bbox)
    
    print(f"   Tile: {TILE_SIZE}x{TILE_SIZE} @ {TILE_SCALE}m = {TILE_SIZE*TILE_SCALE/1000:.2f}km")
    print(f"   BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    
    # Get Sentinel-2 Surface Reflectance
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']))
    
    count = collection.size().getInfo()
    print(f"   Found {count} images")
    
    if count == 0:
        raise ValueError("No images found for this location/date range")
    
    # Get median composite
    image = collection.median()
    
    # Sample the image at 10m resolution
    try:
        sample = image.sampleRectangle(region=region, defaultValue=0)
        
        bands_data = []
        band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        
        for band in band_names:
            arr = np.array(sample.get(band).getInfo())
            bands_data.append(arr)
        
        # Stack bands
        image_array = np.stack(bands_data, axis=0).astype(np.float32)
        
        # Resize to exactly 256x256 if needed
        if image_array.shape[1] != TILE_SIZE or image_array.shape[2] != TILE_SIZE:
            from PIL import Image
            resized = []
            for i in range(10):
                img = Image.fromarray(image_array[i])
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
                resized.append(np.array(img))
            image_array = np.stack(resized, axis=0)
            print(f"   Resized from {bands_data[0].shape} to {TILE_SIZE}x{TILE_SIZE}")
        
        return image_array
        
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        raise

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
    # Enhance for visualization
    rgb = np.clip(rgb / 2500 * 2.0, 0, 1)
    return rgb

def visualize_results(results, output_path):
    """Create visualization matching the training image style."""
    n_tiles = len(results)
    fig, axes = plt.subplots(n_tiles, 4, figsize=(16, 4 * n_tiles))
    
    if n_tiles == 1:
        axes = [axes]
    
    cmap = get_fire_colormap()
    
    for i, (tile_key, data) in enumerate(results.items()):
        # RGB
        axes[i][0].imshow(data['rgb'])
        axes[i][0].set_title(f"{data['name']}\nRGB (2.56km tile)", fontsize=10)
        axes[i][0].axis('off')
        
        # Prediction (black-to-golden)
        im1 = axes[i][1].imshow(data['prediction'], cmap=cmap, vmin=0, vmax=1)
        fire_pct = (data['prediction'] > 0.3).mean() * 100
        axes[i][1].set_title(f"Prediction\n{fire_pct:.1f}% fire @ 0.3 threshold", fontsize=10)
        axes[i][1].axis('off')
        plt.colorbar(im1, ax=axes[i][1], fraction=0.046)
        
        # Binary at 0.3 threshold
        binary = (data['prediction'] > 0.3).astype(float)
        axes[i][2].imshow(binary, cmap='gray', vmin=0, vmax=1)
        axes[i][2].set_title(f"Binary @ 0.3", fontsize=10)
        axes[i][2].axis('off')
        
        # Stats
        ax = axes[i][3]
        ax.axis('off')
        stats_text = f"""
{data['name']}
----------------------
Center: {data['center']}
Description: {data['description']}

Mean Severity: {data['prediction'].mean():.1%}
Max Severity: {data['prediction'].max():.1%}
Fire (>30%): {fire_pct:.1f}%
High (>70%): {(data['prediction'] > 0.7).mean()*100:.1f}%
"""
        ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                transform=ax.transAxes, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_path}")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("EcoRevive Model Test - 2.56km Tiles")
    print("=" * 60)
    
    # Initialize Earth Engine
    if not initialize_ee():
        print("Cannot proceed without Earth Engine")
        return
    
    # Load model
    device = get_device()
    print(f"\nDevice: {device}")
    
    checkpoint_path = FIRE_MODEL_ROOT / "checkpoints" / "model.pth"
    if not checkpoint_path.exists():
        print(f"Model not found at {checkpoint_path}")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    print("Model loaded!")
    
    # Test each tile
    results = {}
    
    for tile_key, tile_info in TEST_TILES.items():
        print(f"\n{'='*50}")
        print(f"Testing: {tile_info['name']}")
        print(f"   Center: {tile_info['center']}")
        print(f"   Dates: {tile_info['post_fire_dates']}")
        print(f"   {tile_info['description']}")
        
        # Download from Earth Engine
        print("   Downloading...")
        try:
            image = download_sentinel2_tile(
                center=tile_info['center'],
                start_date=tile_info['post_fire_dates'][0],
                end_date=tile_info['post_fire_dates'][1],
            )
            print(f"   Downloaded: {image.shape}")
            
            # Check for valid data
            if image.max() < 100:
                print(f"   WARNING: Very low values, may be nodata")
            
            # Run inference
            print("   Running inference...")
            prediction = predict_severity(image, model, device)
            fire_pct = (prediction > 0.3).mean() * 100
            print(f"   Result: mean={prediction.mean():.1%}, max={prediction.max():.1%}, fire%={fire_pct:.1f}%")
            
            # Store results
            results[tile_key] = {
                'name': tile_info['name'],
                'center': tile_info['center'],
                'description': tile_info['description'],
                'image': image,
                'rgb': create_rgb(image),
                'prediction': prediction,
            }
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize
    if results:
        output_path = PROJECT_ROOT / "test_fires_v2_results.png"
        print(f"\n{'='*50}")
        print("Generating visualization...")
        visualize_results(results, output_path)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for tile_key, data in results.items():
            fire_pct = (data['prediction'] > 0.3).mean() * 100
            high_pct = (data['prediction'] > 0.7).mean() * 100
            print(f"   {data['name']}: {fire_pct:.1f}% fire, {high_pct:.1f}% high severity")
    else:
        print("\nNo results to display")

if __name__ == "__main__":
    main()
