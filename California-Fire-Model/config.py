"""
California Fire Model Configuration
All settings in one place - no magic numbers scattered across files
"""

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# EARTH ENGINE
# ============================================================
EE_PROJECT_ID = 'hale-life-482914-r0'  # Your Earth Engine project ID
DRIVE_FOLDER = 'California_Fire_Model'  # Google Drive folder for exports

# ============================================================
# SENTINEL-2 BANDS
# ============================================================
# All 10 bands we use (excluding B1, B9, B10 which are low-res or atmospheric)
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
NUM_BANDS = len(BANDS)

# Band descriptions for reference
BAND_INFO = {
    'B2': 'Blue (490nm)',
    'B3': 'Green (560nm)',
    'B4': 'Red (665nm)',
    'B5': 'Red Edge 1 (705nm)',
    'B6': 'Red Edge 2 (740nm)',
    'B7': 'Red Edge 3 (783nm)',
    'B8': 'NIR (842nm)',
    'B8A': 'NIR narrow (865nm)',
    'B11': 'SWIR 1 (1610nm)',
    'B12': 'SWIR 2 (2190nm)',
}

# ============================================================
# NORMALIZATION STATISTICS
# ============================================================
# IMPORTANT: These will be COMPUTED from your training data
# Run: python data/compute_statistics.py
# Then update these values!

# Placeholder values (will be replaced after running compute_statistics.py)
BAND_MEANS = None  # Will be set after computing
BAND_STDS = None   # Will be set after computing

# Default fallback values (rough estimates, use only if computation fails)
DEFAULT_BAND_MEANS = [1339, 1167, 1002, 1296, 1835, 2149, 2290, 2410, 2004, 1075]
DEFAULT_BAND_STDS = [545, 476, 571, 532, 614, 731, 811, 872, 856, 611]

def get_band_stats():
    """Get band statistics, loading from computed file if available."""
    stats_file = DATA_DIR / "band_statistics.json"
    
    if stats_file.exists():
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        return stats['means'], stats['stds']
    else:
        print("⚠️  Using default band statistics. Run compute_statistics.py first!")
        return DEFAULT_BAND_MEANS, DEFAULT_BAND_STDS

# ============================================================
# CALIFORNIA FIRES - TRAINING SET
# ============================================================
TRAINING_FIRES = {
    'dixie_2021': {
        'name': 'Dixie Fire',
        'year': 2021,
        'center': (40.05, -121.20),  # lat, lon
        'bbox': [-121.60, 39.75, -120.50, 40.55],  # west, south, east, north
        'pre_fire_dates': ('2021-05-01', '2021-06-30'),
        'post_fire_dates': ('2021-10-01', '2021-11-30'),
        'recovery_dates': [
            ('2022-06-01', '2022-08-31'),  # 1 year
            ('2023-06-01', '2023-08-31'),  # 2 years
        ],
        'acres': 963309,
        'counties': ['Butte', 'Plumas', 'Lassen', 'Shasta', 'Tehama'],
    },
    'caldor_2021': {
        'name': 'Caldor Fire',
        'year': 2021,
        'center': (38.78, -120.15),
        'bbox': [-120.50, 38.55, -119.80, 39.00],
        'pre_fire_dates': ('2021-06-01', '2021-07-31'),
        'post_fire_dates': ('2021-09-15', '2021-11-15'),
        'recovery_dates': [
            ('2022-06-01', '2022-08-31'),
            ('2023-06-01', '2023-08-31'),
        ],
        'acres': 221835,
        'counties': ['El Dorado', 'Amador', 'Alpine'],
    },
    'creek_2020': {
        'name': 'Creek Fire',
        'year': 2020,
        'center': (37.15, -119.25),
        'bbox': [-119.60, 36.90, -118.90, 37.50],
        'pre_fire_dates': ('2020-06-01', '2020-08-01'),
        'post_fire_dates': ('2020-11-01', '2020-12-31'),
        'recovery_dates': [
            ('2021-06-01', '2021-08-31'),
            ('2022-06-01', '2022-08-31'),
            ('2023-06-01', '2023-08-31'),
        ],
        'acres': 379895,
        'counties': ['Fresno', 'Madera'],
    },
    'camp_2018': {
        'name': 'Camp Fire',
        'year': 2018,
        'center': (39.76, -121.62),
        'bbox': [-121.90, 39.60, -121.30, 39.95],
        'pre_fire_dates': ('2018-06-01', '2018-09-30'),
        'post_fire_dates': ('2018-12-01', '2019-02-28'),
        'recovery_dates': [
            ('2019-06-01', '2019-08-31'),
            ('2020-06-01', '2020-08-31'),
            ('2021-06-01', '2021-08-31'),
            ('2022-06-01', '2022-08-31'),
            ('2023-06-01', '2023-08-31'),
        ],
        'acres': 153336,
        'counties': ['Butte'],
    },
    'mendocino_complex_2018': {
        'name': 'Mendocino Complex Fire',
        'year': 2018,
        'center': (39.25, -122.80),
        'bbox': [-123.20, 38.90, -122.40, 39.60],
        'pre_fire_dates': ('2018-05-01', '2018-06-30'),
        'post_fire_dates': ('2018-09-15', '2018-11-15'),
        'recovery_dates': [
            ('2019-06-01', '2019-08-31'),
            ('2020-06-01', '2020-08-31'),
        ],
        'acres': 459123,
        'counties': ['Mendocino', 'Lake', 'Colusa', 'Glenn'],
    },
    'thomas_2017': {
        'name': 'Thomas Fire',
        'year': 2017,
        'center': (34.45, -119.35),
        'bbox': [-119.80, 34.20, -118.90, 34.70],
        'pre_fire_dates': ('2017-09-01', '2017-11-30'),
        'post_fire_dates': ('2018-02-01', '2018-04-30'),
        'recovery_dates': [
            ('2018-06-01', '2018-08-31'),
            ('2019-06-01', '2019-08-31'),
            ('2020-06-01', '2020-08-31'),
        ],
        'acres': 281893,
        'counties': ['Ventura', 'Santa Barbara'],
    },
    'rim_2013': {
        'name': 'Rim Fire',
        'year': 2013,
        'center': (37.90, -119.95),
        'bbox': [-120.30, 37.65, -119.60, 38.15],
        'pre_fire_dates': ('2013-06-01', '2013-07-31'),
        'post_fire_dates': ('2013-10-01', '2013-11-30'),
        'recovery_dates': [
            ('2014-06-01', '2014-08-31'),
            ('2016-06-01', '2016-08-31'),
            ('2018-06-01', '2018-08-31'),
            ('2020-06-01', '2020-08-31'),
            ('2023-06-01', '2023-08-31'),  # 10 year recovery
        ],
        'acres': 257314,
        'counties': ['Tuolumne'],
    },
}

# ============================================================
# CALIFORNIA FIRES - TEST SET (HELD OUT)
# ============================================================
TEST_FIRES = {
    'kincade_2019': {
        'name': 'Kincade Fire',
        'year': 2019,
        'center': (38.75, -122.75),
        'bbox': [-123.00, 38.55, -122.50, 38.95],
        'pre_fire_dates': ('2019-08-01', '2019-10-15'),
        'post_fire_dates': ('2019-11-15', '2020-01-31'),
        'recovery_dates': [
            ('2020-06-01', '2020-08-31'),
            ('2021-06-01', '2021-08-31'),
        ],
        'acres': 77758,
        'counties': ['Sonoma'],
    },
    'woolsey_2018': {
        'name': 'Woolsey Fire',
        'year': 2018,
        'center': (34.12, -118.82),
        'bbox': [-119.10, 33.95, -118.55, 34.30],
        'pre_fire_dates': ('2018-08-01', '2018-10-31'),
        'post_fire_dates': ('2018-12-01', '2019-02-28'),
        'recovery_dates': [
            ('2019-06-01', '2019-08-31'),
            ('2020-06-01', '2020-08-31'),
        ],
        'acres': 96949,
        'counties': ['Los Angeles', 'Ventura'],
    },
}

# ============================================================
# HEALTHY/UNBURNED REFERENCE AREAS
# ============================================================
HEALTHY_REGIONS = {
    'sierra_tahoe': {
        'name': 'Lake Tahoe Basin (Healthy)',
        'bbox': [-120.25, 38.90, -119.85, 39.20],
        'dates': ('2024-06-01', '2024-08-31'),
        'vegetation': 'Conifer Forest',
    },
    'plumas_healthy': {
        'name': 'Plumas National Forest (Healthy)',
        'bbox': [-121.40, 39.90, -121.00, 40.20],
        'dates': ('2024-06-01', '2024-08-31'),
        'vegetation': 'Mixed Conifer',
    },
    'redwood_healthy': {
        'name': 'Redwood National Park',
        'bbox': [-124.10, 41.15, -123.70, 41.45],
        'dates': ('2024-06-01', '2024-08-31'),
        'vegetation': 'Coastal Redwood',
    },
    'oak_foothills': {
        'name': 'Sierra Foothills (Healthy)',
        'bbox': [-121.00, 38.50, -120.60, 38.80],
        'dates': ('2024-04-01', '2024-06-30'),
        'vegetation': 'Oak Woodland',
    },
}

# ============================================================
# DATA COLLECTION SETTINGS
# ============================================================
TILE_SIZE = 256  # Pixels per tile
TILE_SCALE = 10  # Meters per pixel (Sentinel-2 native)
CLOUD_THRESHOLD = 10  # Max cloud percentage
MIN_VALID_PIXELS = 0.8  # Minimum fraction of valid pixels in tile

# ============================================================
# BURN SEVERITY (dNBR) THRESHOLDS
# ============================================================
SEVERITY_THRESHOLDS = {
    'unburned': (float('-inf'), 0.10),
    'low': (0.10, 0.27),
    'moderate_low': (0.27, 0.44),
    'moderate_high': (0.44, 0.66),
    'high': (0.66, float('inf')),
}

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
MODEL_CONFIG = {
    'input_channels': NUM_BANDS,
    'output_channels': 1,  # Single channel: burn severity 0-1
    'base_channels': 64,
    'use_attention': True,
    'dropout': 0.2,
}

# ============================================================
# TRAINING SETTINGS
# ============================================================
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'early_stopping_patience': 10,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'gradient_clip': 1.0,
}

# Validation split
VAL_SPLIT = 0.15  # 15% of training fires for validation

# ============================================================
# AUGMENTATION
# ============================================================
AUGMENTATION_CONFIG = {
    'random_rotate90': True,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_contrast': {
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
        'p': 0.3,
    },
    'gaussian_noise': {
        'std_limit': (10, 30),
        'p': 0.2,
    },
}

# ============================================================
# INFERENCE SETTINGS
# ============================================================
INFERENCE_CONFIG = {
    'batch_size': 32,
    'use_tta': True,  # Test-time augmentation
    'tta_transforms': ['none', 'hflip', 'vflip', 'rotate90'],
}

# ============================================================
# LOGGING
# ============================================================
WANDB_PROJECT = 'california-fire-model'  # Weights & Biases project name
USE_WANDB = False  # Set to True if you want experiment tracking

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_all_fire_configs():
    """Return combined training and test fire configurations."""
    return {**TRAINING_FIRES, **TEST_FIRES}

def get_fire_by_name(fire_key):
    """Get fire configuration by key."""
    all_fires = get_all_fire_configs()
    return all_fires.get(fire_key)

def is_test_fire(fire_key):
    """Check if a fire is in the held-out test set."""
    return fire_key in TEST_FIRES

def print_config_summary():
    """Print a summary of the configuration."""
    print("=" * 60)
    print("🔥 California Fire Model Configuration")
    print("=" * 60)
    print(f"\n📂 Directories:")
    print(f"   Data: {DATA_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print(f"   Logs: {LOG_DIR}")
    print(f"\n🔥 Training Fires: {len(TRAINING_FIRES)}")
    for key, fire in TRAINING_FIRES.items():
        print(f"   - {fire['name']} ({fire['year']}): {fire['acres']:,} acres")
    print(f"\n🧪 Test Fires (held out): {len(TEST_FIRES)}")
    for key, fire in TEST_FIRES.items():
        print(f"   - {fire['name']} ({fire['year']}): {fire['acres']:,} acres")
    print(f"\n🌲 Healthy Reference Regions: {len(HEALTHY_REGIONS)}")
    print(f"\n⚙️ Model: {NUM_BANDS} input bands, {MODEL_CONFIG['base_channels']} base channels")
    print(f"\n🏋️ Training: batch={TRAINING_CONFIG['batch_size']}, "
          f"lr={TRAINING_CONFIG['learning_rate']}, epochs={TRAINING_CONFIG['epochs']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
