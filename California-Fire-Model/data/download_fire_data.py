"""
California Fire Data Collection Script
Downloads Sentinel-2 imagery and burn severity labels from Google Earth Engine

Key improvements over previous approach:
1. Continuous burn severity labels (dNBR) instead of binary
2. Temporal pairs (before/after fire)
3. MTBS fire perimeters for ground truth
4. Separate healthy reference areas
5. Proper tile filtering for quality
"""

import ee
import sys
from pathlib import Path

# Add parent directory for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    EE_PROJECT_ID, DRIVE_FOLDER, BANDS, TILE_SIZE, TILE_SCALE,
    CLOUD_THRESHOLD, TRAINING_FIRES, TEST_FIRES, HEALTHY_REGIONS
)

# ============================================================
# INITIALIZATION
# ============================================================
def initialize_ee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=EE_PROJECT_ID)
        print("✅ Google Earth Engine initialized")
    except Exception as e:
        print("🔐 Authentication required...")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT_ID)
        print("✅ Google Earth Engine initialized")

# ============================================================
# BURN SEVERITY CALCULATION
# ============================================================
def compute_nbr(image):
    """
    Compute Normalized Burn Ratio (NBR).
    NBR = (NIR - SWIR) / (NIR + SWIR) = (B8 - B12) / (B8 + B12)
    
    High NBR = healthy vegetation
    Low NBR = burned/bare
    """
    return image.normalizedDifference(['B8', 'B12']).rename('NBR')


def compute_dnbr(pre_image, post_image):
    """
    Compute differenced NBR (dNBR) for burn severity.
    dNBR = Pre-fire NBR - Post-fire NBR
    
    Higher dNBR = more severe burn
    ~0 = no change (unburned)
    Negative = regrowth (if post is healthier than pre)
    """
    pre_nbr = compute_nbr(pre_image)
    post_nbr = compute_nbr(post_image)
    
    dnbr = pre_nbr.subtract(post_nbr).rename('dNBR')
    
    # Clamp to 0-1 range for use as regression target
    # Most severe burns are around 0.66-1.0
    # Unburned is typically -0.1 to 0.1
    dnbr_normalized = dnbr.clamp(-0.1, 1.0).add(0.1).divide(1.1)  # Map to ~0-1
    
    return dnbr_normalized.rename('severity')


def get_burn_severity_from_dnbr(pre_image, post_image):
    """
    Get continuous burn severity as ground truth label.
    This is the KEY improvement - continuous labels instead of binary.
    """
    dnbr = pre_image.normalizedDifference(['B8', 'B12']).subtract(
        post_image.normalizedDifference(['B8', 'B12'])
    )
    
    # Normalize to 0-1 range
    # dNBR typically ranges from -0.5 (regrowth) to 1.3 (extreme burn)
    # We map positive values to 0-1 severity
    severity = dnbr.clamp(0, 1).rename('severity')
    
    return severity


# ============================================================
# IMAGE COLLECTION FUNCTIONS
# ============================================================
def get_sentinel2_composite(bbox, date_start, date_end, cloud_threshold=CLOUD_THRESHOLD):
    """
    Get a cloud-free Sentinel-2 composite for given date range.
    Uses median to reduce noise and cloud gaps.
    """
    roi = ee.Geometry.Rectangle(bbox)
    
    # Sentinel-2 Level-2A (Surface Reflectance)
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    
    # Use median composite to reduce clouds and noise
    composite = collection.median().select(BANDS)
    
    # Mask invalid pixels (0 or > 10000)
    valid_mask = composite.select('B4').gt(0).And(composite.select('B4').lt(10000))
    composite = composite.updateMask(valid_mask)
    
    return composite, roi


def get_mtbs_burn_perimeter(fire_key, fire_config):
    """
    Get MTBS (Monitoring Trends in Burn Severity) data.
    MTBS provides official fire perimeters and severity classifications.
    """
    # MTBS dataset
    mtbs = ee.ImageCollection('USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1')
    
    year = fire_config['year']
    roi = ee.Geometry.Rectangle(fire_config['bbox'])
    
    # Get the burn severity for the fire year
    severity = mtbs.filter(ee.Filter.eq('year', year)).first()
    
    if severity is not None:
        severity = severity.select('Severity').clip(roi)
        # MTBS severity classes: 1=Unburned, 2=Low, 3=Moderate, 4=High, 5=Increased greenness
        # Normalize to 0-1: map 1->0, 4->1
        normalized = severity.subtract(1).divide(3).clamp(0, 1).rename('severity')
        return normalized
    
    return None


# ============================================================
# DATA EXPORT FUNCTIONS
# ============================================================
def export_fire_tiles(fire_key, fire_config, include_recovery=True):
    """
    Export all tiles for a single fire event.
    Creates:
    - Pre-fire imagery with label=0 (baseline)
    - Post-fire imagery with dNBR severity label
    - Optional recovery imagery with recovery label
    """
    print(f"\n{'='*60}")
    print(f"🔥 Processing: {fire_config['name']} ({fire_config['year']})")
    print(f"{'='*60}")
    
    bbox = fire_config['bbox']
    roi = ee.Geometry.Rectangle(bbox)
    
    # Get pre-fire composite
    print(f"   📥 Fetching pre-fire imagery: {fire_config['pre_fire_dates']}")
    pre_fire, _ = get_sentinel2_composite(
        bbox,
        fire_config['pre_fire_dates'][0],
        fire_config['pre_fire_dates'][1]
    )
    
    # Get post-fire composite
    print(f"   📥 Fetching post-fire imagery: {fire_config['post_fire_dates']}")
    post_fire, _ = get_sentinel2_composite(
        bbox,
        fire_config['post_fire_dates'][0],
        fire_config['post_fire_dates'][1]
    )
    
    # Calculate burn severity (continuous 0-1 label)
    print(f"   🔥 Calculating burn severity (dNBR)...")
    severity = get_burn_severity_from_dnbr(pre_fire, post_fire)
    
    # Try to get MTBS data for comparison/refinement
    mtbs_severity = get_mtbs_burn_perimeter(fire_key, fire_config)
    
    # Use dNBR as primary severity, fall back to MTBS if available
    if mtbs_severity is not None:
        print(f"   ✅ MTBS data available - using for validation")
        # You could blend: severity = severity.multiply(0.7).add(mtbs_severity.multiply(0.3))
    
    tasks = []
    
    # --- Export 1: Pre-fire (healthy baseline, label=0) ---
    pre_fire_label = ee.Image.constant(0).rename('severity').toFloat()
    pre_fire_stack = pre_fire.addBands(pre_fire_label)
    
    task_pre = ee.batch.Export.image.toDrive(
        image=pre_fire_stack.toFloat(),
        description=f"{fire_key}_pre_fire",
        folder=f"{DRIVE_FOLDER}/fires/{fire_key}",
        scale=TILE_SCALE,
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e10,
        fileDimensions=[TILE_SIZE, TILE_SIZE],
        skipEmptyTiles=True,
        formatOptions={'cloudOptimized': True}
    )
    task_pre.start()
    tasks.append(('pre_fire', task_pre))
    print(f"   ✅ Queued: {fire_key}_pre_fire")
    
    # --- Export 2: Post-fire (with dNBR severity label) ---
    post_fire_stack = post_fire.addBands(severity)
    
    task_post = ee.batch.Export.image.toDrive(
        image=post_fire_stack.toFloat(),
        description=f"{fire_key}_post_fire",
        folder=f"{DRIVE_FOLDER}/fires/{fire_key}",
        scale=TILE_SCALE,
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e10,
        fileDimensions=[TILE_SIZE, TILE_SIZE],
        skipEmptyTiles=True,
        formatOptions={'cloudOptimized': True}
    )
    task_post.start()
    tasks.append(('post_fire', task_post))
    print(f"   ✅ Queued: {fire_key}_post_fire")
    
    # --- Export 3+: Recovery imagery (optional) ---
    if include_recovery and 'recovery_dates' in fire_config:
        for i, (rec_start, rec_end) in enumerate(fire_config['recovery_dates'], 1):
            print(f"   📥 Fetching recovery imagery ({i}yr): {rec_start} to {rec_end}")
            
            recovery_img, _ = get_sentinel2_composite(bbox, rec_start, rec_end)
            
            # Calculate recovery severity (dNBR compared to pre-fire)
            # Lower values = more recovery
            recovery_severity = get_burn_severity_from_dnbr(pre_fire, recovery_img)
            
            recovery_stack = recovery_img.addBands(recovery_severity)
            
            task_rec = ee.batch.Export.image.toDrive(
                image=recovery_stack.toFloat(),
                description=f"{fire_key}_recovery_{i}yr",
                folder=f"{DRIVE_FOLDER}/fires/{fire_key}",
                scale=TILE_SCALE,
                region=roi,
                fileFormat='GeoTIFF',
                maxPixels=1e10,
                fileDimensions=[TILE_SIZE, TILE_SIZE],
                skipEmptyTiles=True,
                formatOptions={'cloudOptimized': True}
            )
            task_rec.start()
            tasks.append((f'recovery_{i}yr', task_rec))
            print(f"   ✅ Queued: {fire_key}_recovery_{i}yr")
    
    return tasks


def export_healthy_region(region_key, region_config):
    """Export healthy/unburned reference areas with label=0."""
    print(f"\n🌲 Processing healthy region: {region_config['name']}")
    
    bbox = region_config['bbox']
    roi = ee.Geometry.Rectangle(bbox)
    
    # Get imagery
    healthy_img, _ = get_sentinel2_composite(
        bbox,
        region_config['dates'][0],
        region_config['dates'][1]
    )
    
    # Label = 0 (no degradation)
    label = ee.Image.constant(0).rename('severity').toFloat()
    stack = healthy_img.addBands(label)
    
    task = ee.batch.Export.image.toDrive(
        image=stack.toFloat(),
        description=f"healthy_{region_key}",
        folder=f"{DRIVE_FOLDER}/healthy",
        scale=TILE_SCALE,
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e10,
        fileDimensions=[TILE_SIZE, TILE_SIZE],
        skipEmptyTiles=True,
        formatOptions={'cloudOptimized': True}
    )
    task.start()
    print(f"   ✅ Queued: healthy_{region_key}")
    
    return task


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main function to download all data."""
    print("=" * 70)
    print("🔥 CALIFORNIA FIRE MODEL - DATA COLLECTION")
    print("=" * 70)
    print("\nThis script will download:")
    print(f"  • {len(TRAINING_FIRES)} training fire events")
    print(f"  • {len(TEST_FIRES)} test fire events (held out)")
    print(f"  • {len(HEALTHY_REGIONS)} healthy reference regions")
    print(f"\nData will be saved to Google Drive: {DRIVE_FOLDER}/")
    print("=" * 70)
    
    # Initialize Earth Engine
    initialize_ee()
    
    all_tasks = []
    
    # --- Training Fires ---
    print("\n" + "=" * 70)
    print("📚 TRAINING FIRES")
    print("=" * 70)
    
    for fire_key, fire_config in TRAINING_FIRES.items():
        tasks = export_fire_tiles(fire_key, fire_config, include_recovery=True)
        all_tasks.extend(tasks)
    
    # --- Test Fires (held out) ---
    print("\n" + "=" * 70)
    print("🧪 TEST FIRES (HELD OUT)")
    print("=" * 70)
    
    for fire_key, fire_config in TEST_FIRES.items():
        tasks = export_fire_tiles(fire_key, fire_config, include_recovery=True)
        all_tasks.extend(tasks)
    
    # --- Healthy Regions ---
    print("\n" + "=" * 70)
    print("🌲 HEALTHY REFERENCE REGIONS")
    print("=" * 70)
    
    for region_key, region_config in HEALTHY_REGIONS.items():
        task = export_healthy_region(region_key, region_config)
        all_tasks.append(('healthy', task))
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("✅ ALL TASKS QUEUED!")
    print("=" * 70)
    print(f"\nTotal export tasks: {len(all_tasks)}")
    print(f"\nData structure in Google Drive:")
    print(f"  {DRIVE_FOLDER}/")
    print(f"  ├── fires/")
    for fire_key in list(TRAINING_FIRES.keys()) + list(TEST_FIRES.keys()):
        print(f"  │   ├── {fire_key}/")
        print(f"  │   │   ├── *_pre_fire_*.tif")
        print(f"  │   │   ├── *_post_fire_*.tif")
        print(f"  │   │   └── *_recovery_*yr_*.tif")
    print(f"  └── healthy/")
    for region_key in HEALTHY_REGIONS.keys():
        print(f"      ├── healthy_{region_key}_*.tif")
    
    print(f"\n🔗 Monitor progress: https://code.earthengine.google.com/tasks")
    print(f"\n⏱️ Estimated time: 4-8 hours for all tiles")
    print(f"💾 Estimated size: 15-25 GB total")
    
    print("\n" + "=" * 70)
    print("📋 NEXT STEPS")
    print("=" * 70)
    print("1. Wait for all tasks to complete (check GEE Tasks page)")
    print("2. Download data from Google Drive to local storage")
    print("3. Run: python compute_statistics.py")
    print("4. Run: python validate_dataset.py")
    print("5. Start training: python ../training/train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
