"""
Compute Band Statistics from Training Data
Calculates actual mean and standard deviation for each Sentinel-2 band.

This is CRITICAL - using incorrect normalization stats causes:
- Feature misalignment between training and inference
- Poor generalization
- Reduced model performance

Run this after downloading data, before training.
"""

import os
import sys
import json
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# Add parent directory for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_DATA_DIR, BANDS, NUM_BANDS

# ============================================================
# CONFIGURATION
# ============================================================
# Where to find downloaded tiles
DATA_PATHS = [
    RAW_DATA_DIR / "fires",    # Fire imagery
    RAW_DATA_DIR / "healthy",  # Healthy reference imagery
]

# Output file
STATS_OUTPUT = DATA_DIR / "band_statistics.json"

# Sample size (for very large datasets, sample a subset)
MAX_TILES_TO_SAMPLE = 5000  # Set to None to use all tiles
PIXELS_PER_TILE = 256 * 256


# ============================================================
# STATISTICS COMPUTATION
# ============================================================
def collect_all_tiles(data_paths):
    """Collect all .tif files from data directories."""
    all_tiles = []
    
    for data_path in data_paths:
        if not data_path.exists():
            print(f"⚠️  Path not found: {data_path}")
            continue
            
        # Walk through all subdirectories
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.tif'):
                    all_tiles.append(Path(root) / file)
    
    return all_tiles


def compute_running_stats(tiles, max_tiles=None):
    """
    Compute mean and std using Welford's online algorithm.
    Memory efficient - doesn't load all data at once.
    """
    if max_tiles and len(tiles) > max_tiles:
        # Random sample
        np.random.seed(42)
        tiles = np.random.choice(tiles, max_tiles, replace=False)
        print(f"   Sampling {max_tiles} tiles from {len(tiles)} total")
    
    # Initialize running statistics
    count = 0
    mean = np.zeros(NUM_BANDS, dtype=np.float64)
    M2 = np.zeros(NUM_BANDS, dtype=np.float64)  # Sum of squared differences
    
    # Also track min/max for range checking
    global_min = np.full(NUM_BANDS, np.inf)
    global_max = np.full(NUM_BANDS, -np.inf)
    
    valid_tiles = 0
    
    for tile_path in tqdm(tiles, desc="Computing statistics"):
        try:
            with rasterio.open(tile_path) as src:
                data = src.read()
                
                # Expect 11 bands (10 Sentinel-2 + 1 label)
                if data.shape[0] < NUM_BANDS:
                    continue
                
                # Extract only the spectral bands (not the label)
                spectral = data[:NUM_BANDS]
                
                # Flatten to (bands, pixels)
                spectral = spectral.reshape(NUM_BANDS, -1)
                
                # Mask invalid pixels (NaN, Inf, or out of range)
                valid_mask = np.isfinite(spectral).all(axis=0)
                valid_mask &= (spectral[0] > 0)  # At least B2 > 0
                valid_mask &= (spectral[0] < 10000)  # Within S2 range
                
                spectral = spectral[:, valid_mask]
                
                if spectral.shape[1] == 0:
                    continue
                
                valid_tiles += 1
                
                # Update running statistics (Welford's algorithm)
                for pixel_idx in range(spectral.shape[1]):
                    count += 1
                    pixel = spectral[:, pixel_idx]
                    
                    delta = pixel - mean
                    mean += delta / count
                    delta2 = pixel - mean
                    M2 += delta * delta2
                
                # Update min/max
                global_min = np.minimum(global_min, spectral.min(axis=1))
                global_max = np.maximum(global_max, spectral.max(axis=1))
                
        except Exception as e:
            # Skip problematic tiles
            continue
    
    # Compute standard deviation
    if count > 1:
        variance = M2 / (count - 1)
        std = np.sqrt(variance)
    else:
        std = np.ones(NUM_BANDS)
    
    return {
        'means': mean.tolist(),
        'stds': std.tolist(),
        'mins': global_min.tolist(),
        'maxs': global_max.tolist(),
        'pixel_count': count,
        'tile_count': valid_tiles,
    }


def compute_stats_batch(tiles, max_tiles=None, batch_size=100):
    """
    Alternative: Batch computation (faster but uses more memory).
    Good for smaller datasets.
    """
    if max_tiles and len(tiles) > max_tiles:
        np.random.seed(42)
        tiles = list(np.random.choice(tiles, max_tiles, replace=False))
    
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    
    for i in tqdm(range(0, len(tiles), batch_size), desc="Processing batches"):
        batch_tiles = tiles[i:i+batch_size]
        batch_data = []
        
        for tile_path in batch_tiles:
            try:
                with rasterio.open(tile_path) as src:
                    data = src.read()
                    
                    if data.shape[0] < NUM_BANDS:
                        continue
                    
                    spectral = data[:NUM_BANDS].astype(np.float32)
                    
                    # Mask invalid
                    valid_mask = np.isfinite(spectral).all(axis=0)
                    valid_mask &= (spectral[0] > 0) & (spectral[0] < 10000)
                    
                    if valid_mask.sum() > 100:  # At least 100 valid pixels
                        # Compute per-tile stats
                        valid_spectral = spectral[:, valid_mask.flatten()]
                        all_means.append(valid_spectral.mean(axis=1))
                        all_stds.append(valid_spectral.std(axis=1))
                        all_mins.append(valid_spectral.min(axis=1))
                        all_maxs.append(valid_spectral.max(axis=1))
                        
            except Exception:
                continue
    
    if len(all_means) == 0:
        raise ValueError("No valid tiles found!")
    
    # Aggregate across tiles
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    all_mins = np.array(all_mins)
    all_maxs = np.array(all_maxs)
    
    return {
        'means': all_means.mean(axis=0).tolist(),
        'stds': all_stds.mean(axis=0).tolist(),  # Average of per-tile stds
        'mins': all_mins.min(axis=0).tolist(),
        'maxs': all_maxs.max(axis=0).tolist(),
        'tile_count': len(all_means),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("📊 COMPUTING BAND STATISTICS")
    print("=" * 70)
    
    # Collect all tiles
    print("\n📂 Collecting tiles...")
    tiles = collect_all_tiles(DATA_PATHS)
    
    if len(tiles) == 0:
        print("\n❌ No tiles found!")
        print(f"   Expected data in: {DATA_PATHS}")
        print("\n   Please download data first:")
        print("   python download_fire_data.py")
        return
    
    print(f"   Found {len(tiles)} tiles")
    
    # Compute statistics
    print("\n📈 Computing statistics...")
    
    if len(tiles) < 1000:
        # Small dataset: batch method
        stats = compute_stats_batch(tiles)
    else:
        # Large dataset: streaming method
        stats = compute_running_stats(tiles, max_tiles=MAX_TILES_TO_SAMPLE)
    
    # Save results
    print(f"\n💾 Saving to: {STATS_OUTPUT}")
    STATS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BAND STATISTICS SUMMARY")
    print("=" * 70)
    
    print(f"\n   Tiles processed: {stats['tile_count']}")
    
    print("\n   Band Statistics:")
    print(f"   {'Band':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("   " + "-" * 50)
    
    for i, band in enumerate(BANDS):
        mean = stats['means'][i]
        std = stats['stds'][i]
        min_val = stats.get('mins', [0]*NUM_BANDS)[i]
        max_val = stats.get('maxs', [10000]*NUM_BANDS)[i]
        print(f"   {band:<6} {mean:>10.1f} {std:>10.1f} {min_val:>10.1f} {max_val:>10.1f}")
    
    # Generate Python code for config.py
    print("\n" + "=" * 70)
    print("📝 UPDATE config.py WITH THESE VALUES:")
    print("=" * 70)
    
    means_str = ", ".join([f"{v:.1f}" for v in stats['means']])
    stds_str = ", ".join([f"{v:.1f}" for v in stats['stds']])
    
    print(f"""
# Computed from training data - {stats['tile_count']} tiles
BAND_MEANS = [{means_str}]
BAND_STDS = [{stds_str}]
""")
    
    print("=" * 70)
    print("✅ Done! Band statistics saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
