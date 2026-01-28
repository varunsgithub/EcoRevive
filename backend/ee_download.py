"""
Earth Engine Download Module
Downloads Sentinel-2 imagery matching the California Fire Model specifications EXACTLY.

Matches training config:
- TILE_SIZE = 256 pixels
- TILE_SCALE = 10 meters per pixel  
- BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
"""

import ee
import numpy as np
from typing import Tuple, Dict, Any
import requests
from PIL import Image
import io
import base64

# Model specifications - MUST MATCH config.py
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
TILE_SIZE = 256  # Pixels per tile
TILE_SCALE = 10  # Meters per pixel (Sentinel-2 native)
CLOUD_THRESHOLD = 20

# Earth Engine project
EE_PROJECT_ID = 'hale-life-482914-r0'


def initialize_ee(project_id: str = EE_PROJECT_ID):
    """Initialize Google Earth Engine."""
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"✅ Earth Engine initialized with project: {project_id}")
        return True
    except Exception as e:
        print(f"⚠️ EE initialization failed: {e}")
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized (default)")
            return True
        except Exception as e2:
            print(f"❌ EE initialization failed: {e2}")
            return False


def get_sentinel2_composite(bbox: Dict[str, float], start_date: str, end_date: str):
    """
    Get a cloud-free Sentinel-2 composite for given date range.
    Matches training script: COPERNICUS/S2_SR_HARMONIZED with cloud filtering.
    """
    roi = ee.Geometry.Rectangle([
        bbox['west'], bbox['south'],
        bbox['east'], bbox['north']
    ])
    
    # Sentinel-2 Level-2A (Surface Reflectance) - same as training
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
    )
    
    # Use median composite to reduce clouds and noise - same as training
    composite = collection.median().select(BANDS)
    
    # Mask invalid pixels (0 or > 10000) - same as training
    valid_mask = composite.select('B4').gt(0).And(composite.select('B4').lt(10000))
    composite = composite.updateMask(valid_mask)
    
    return composite.clip(roi), roi


def download_for_inference(
    bbox: Dict[str, float],
    start_date: str = '2023-06-01',
    end_date: str = '2023-09-30',
    max_tiles: int = 250  # Limit to prevent excessive downloads
) -> Tuple[list, Dict[str, Any]]:
    """
    Download Sentinel-2 imagery ready for model inference.
    Supports TILED downloads for large areas - breaks into 2.56km tiles.
    
    Returns:
        Tuple of (list of tile dicts, metadata dict)
        Each tile dict: {'image': array, 'row': int, 'col': int, 'center': (lat, lon)}
    """
    # Calculate tile size in degrees (approximate)
    center_lat = (bbox['south'] + bbox['north']) / 2
    tile_size_deg_lat = (TILE_SIZE * TILE_SCALE) / 111000  # ~0.023 degrees
    tile_size_deg_lon = (TILE_SIZE * TILE_SCALE) / (111000 * np.cos(np.radians(center_lat)))
    
    # Calculate selection dimensions
    width_deg = bbox['east'] - bbox['west']
    height_deg = bbox['north'] - bbox['south']
    
    # Calculate number of tiles needed
    n_cols = max(1, int(np.ceil(width_deg / tile_size_deg_lon)))
    n_rows = max(1, int(np.ceil(height_deg / tile_size_deg_lat)))
    total_tiles = n_rows * n_cols
    
    print(f"   Selection: {width_deg*111:.1f}km x {height_deg*111:.1f}km")
    print(f"   Tile grid: {n_cols}x{n_rows} = {total_tiles} tiles")
    
    # Limit tiles to prevent excessive downloads
    if total_tiles > max_tiles:
        print(f"   ⚠️ Too many tiles ({total_tiles}), using center {max_tiles} tiles")
        # Reduce grid size
        scale = np.sqrt(max_tiles / total_tiles)
        n_cols = max(1, int(n_cols * scale))
        n_rows = max(1, int(n_rows * scale))
        total_tiles = n_rows * n_cols
        print(f"   Reduced to: {n_cols}x{n_rows} = {total_tiles} tiles")
    
    # Get base composite for whole region (for cloud-free median)
    composite, _ = get_sentinel2_composite(bbox, start_date, end_date)
    
    # Calculate tile centers
    half_tile_lat = tile_size_deg_lat / 2
    half_tile_lon = tile_size_deg_lon / 2
    
    # Start from bottom-left
    start_lon = bbox['west'] + half_tile_lon
    start_lat = bbox['south'] + half_tile_lat
    
    # If we have fewer tiles than needed, center them
    if n_cols * tile_size_deg_lon < width_deg:
        start_lon = bbox['west'] + (width_deg - n_cols * tile_size_deg_lon) / 2 + half_tile_lon
    if n_rows * tile_size_deg_lat < height_deg:
        start_lat = bbox['south'] + (height_deg - n_rows * tile_size_deg_lat) / 2 + half_tile_lat
    
    # Download each tile
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            tile_center_lon = start_lon + col * tile_size_deg_lon
            tile_center_lat = start_lat + row * tile_size_deg_lat
            
            # Create tile ROI
            tile_roi = ee.Geometry.Rectangle([
                tile_center_lon - half_tile_lon,
                tile_center_lat - half_tile_lat,
                tile_center_lon + half_tile_lon,
                tile_center_lat + half_tile_lat
            ])
            
            try:
                print(f"   Tile ({row},{col}): ({tile_center_lat:.4f}, {tile_center_lon:.4f})")
                
                # Download tile
                url = composite.clip(tile_roi).getDownloadURL({
                    'bands': BANDS,
                    'region': tile_roi,
                    'dimensions': f'{TILE_SIZE}x{TILE_SIZE}',
                    'format': 'NPY'
                })
                
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                data = np.load(io.BytesIO(response.content), allow_pickle=True)
                
                if isinstance(data, np.ndarray) and data.dtype.names:
                    arrays = [data[band].astype(np.float32) for band in BANDS]
                    image_array = np.stack(arrays, axis=0)
                else:
                    image_array = data.astype(np.float32)
                
                if image_array.shape != (10, TILE_SIZE, TILE_SIZE):
                    image_array = resize_array(image_array, TILE_SIZE)
                
                tiles.append({
                    'image': image_array,
                    'row': row,
                    'col': col,
                    'center': (tile_center_lat, tile_center_lon)
                })
                
            except Exception as e:
                print(f"   ⚠️ Tile ({row},{col}) failed: {e}")
    
    # Metadata
    metadata = {
        'bands': BANDS,
        'tile_size': TILE_SIZE,
        'scale': TILE_SCALE,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_tiles': len(tiles),
        'bbox': bbox,
        'date_range': {'start': start_date, 'end': end_date},
    }
    
    print(f"   ✅ Downloaded {len(tiles)}/{total_tiles} tiles")
    
    return tiles, metadata


def resize_array(array: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a (C, H, W) array to (C, target_size, target_size)."""
    c, h, w = array.shape
    resized = np.zeros((c, target_size, target_size), dtype=array.dtype)
    
    for i in range(c):
        img = Image.fromarray(array[i])
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        resized[i] = np.array(img_resized)
    
    return resized


def create_rgb_from_bands(image_array: np.ndarray) -> str:
    """
    Create an RGB visualization from Sentinel-2 bands.
    Uses FALSE COLOR: B5 (Red Edge), B4 (Red), B3 (Green) - EXACT training config.
    
    Args:
        image_array: (10, 256, 256) - bands in order B2,B3,B4,B5,...
        
    Returns:
        Base64 encoded PNG string
    """
    # Band statistics from training
    BAND_MEANS = [472.8, 673.8, 770.8, 1087.8, 1747.6, 1997.1, 2106.4, 2188.9, 1976.1, 1404.5]
    BAND_STDS = [223.7, 255.4, 345.6, 313.5, 366.7, 417.6, 476.7, 437.1, 472.7, 438.4]
    
    # EXACT training normalization
    normalized = np.zeros_like(image_array, dtype=np.float32)
    for i in range(10):
        normalized[i] = (image_array[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    normalized = np.clip(normalized, -3, 3)
    normalized = (normalized + 3) / 6  # Scale to [0, 1]
    
    # FALSE COLOR: B5, B4, B3 (bands [3, 2, 1]) - EXACT training config
    rgb = np.stack([
        normalized[3],  # B5 - Red Edge 1 -> R
        normalized[2],  # B4 - Red -> G
        normalized[1],  # B3 - Green -> B
    ], axis=-1)
    
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    # Create image
    img = Image.fromarray(rgb, mode='RGB')
    
    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"


# Alias for compatibility
download_sentinel2_for_model = download_for_inference


# Test function
if __name__ == "__main__":
    # Test with a small area in California (Dixie Fire region)
    test_bbox = {
        'west': -121.25,
        'south': 40.00,
        'east': -121.15,
        'north': 40.10
    }
    
    if initialize_ee():
        print("Downloading test imagery...")
        try:
            image, meta = download_for_inference(test_bbox)
            print(f"✅ Downloaded: {image.shape}")
            print(f"   Center: {meta['center']}")
            print(f"   Value range: {meta['min_val']:.0f} - {meta['max_val']:.0f}")
            
            # Test RGB creation
            rgb = create_rgb_from_bands(image)
            print(f"✅ Created RGB visualization ({len(rgb)} chars)")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
