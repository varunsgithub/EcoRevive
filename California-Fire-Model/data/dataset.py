"""
PyTorch Dataset for California Fire Model
Clean implementation with proper normalization and filtering.

Key improvements:
1. Uses computed band statistics (not hardcoded)
2. Filters out invalid tiles automatically
3. Returns metadata (fire name, stage) for per-category validation
4. Proper handling of continuous labels (regression)
"""

import os
import sys
import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Add parent directory for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, RAW_DATA_DIR, NUM_BANDS, 
    get_band_stats, AUGMENTATION_CONFIG, TRAINING_FIRES, TEST_FIRES
)


class CaliforniaFireDataset(Dataset):
    """
    PyTorch Dataset for California Fire data.
    
    Features:
    - Automatic normalization using computed statistics
    - Filtering of invalid tiles
    - Optional augmentation
    - Metadata for per-fire validation
    - Support for both training and inference
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        mode: str = 'train',  # 'train', 'val', or 'test'
        augment: bool = True,
        min_valid_ratio: float = 0.7,
        exclude_fires: Optional[List[str]] = None,
        include_only_fires: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dirs: List of directories containing .tif files
            mode: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            min_valid_ratio: Minimum ratio of valid pixels required
            exclude_fires: Fire keys to exclude (for held-out testing)
            include_only_fires: Only include these fires (overrides exclude)
        """
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.min_valid_ratio = min_valid_ratio
        
        # Load normalization statistics
        self.band_means, self.band_stds = get_band_stats()
        self.band_means = np.array(self.band_means, dtype=np.float32)
        self.band_stds = np.array(self.band_stds, dtype=np.float32)
        
        # Collect all tile paths
        self.samples = self._collect_samples(
            data_dirs, exclude_fires, include_only_fires
        )
        
        print(f"📂 {mode.upper()} Dataset: {len(self.samples)} tiles from {len(data_dirs)} directories")
        
        # Build augmentation pipeline
        self.transform = self._build_transforms() if self.augment else None
    
    def _collect_samples(
        self,
        data_dirs: List[str],
        exclude_fires: Optional[List[str]],
        include_only_fires: Optional[List[str]],
    ) -> List[Dict]:
        """Collect all valid sample paths with metadata."""
        samples = []
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                print(f"   ⚠️ Path not found: {data_path}")
                continue
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if not file.endswith('.tif'):
                        continue
                    
                    tile_path = Path(root) / file
                    
                    # Extract metadata from path
                    rel_path = tile_path.relative_to(data_path)
                    parts = rel_path.parts
                    
                    # Determine category and stage
                    if 'fires' in str(tile_path):
                        if len(parts) >= 2:
                            fire_key = parts[0]
                            stage = self._extract_stage(file)
                        else:
                            fire_key = 'unknown'
                            stage = 'unknown'
                    else:
                        fire_key = 'healthy'
                        stage = 'reference'
                    
                    # Apply filters
                    if include_only_fires is not None:
                        if fire_key not in include_only_fires:
                            continue
                    elif exclude_fires is not None:
                        if fire_key in exclude_fires:
                            continue
                    
                    samples.append({
                        'path': str(tile_path),
                        'fire_key': fire_key,
                        'stage': stage,
                    })
        
        return samples
    
    def _extract_stage(self, filename: str) -> str:
        """Extract stage (pre_fire, post_fire, recovery_Xyr) from filename."""
        filename_lower = filename.lower()
        
        if 'pre_fire' in filename_lower:
            return 'pre_fire'
        elif 'post_fire' in filename_lower:
            return 'post_fire'
        elif 'recovery' in filename_lower:
            # Extract year number if present
            import re
            match = re.search(r'recovery_(\d+)yr', filename_lower)
            if match:
                return f'recovery_{match.group(1)}yr'
            return 'recovery'
        else:
            return 'unknown'
    
    def _build_transforms(self):
        """Build augmentation pipeline for training."""
        transforms = []
        
        cfg = AUGMENTATION_CONFIG
        
        if cfg.get('random_rotate90', True):
            transforms.append(A.RandomRotate90(p=0.5))
        
        if cfg.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if cfg.get('vertical_flip', True):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if 'brightness_contrast' in cfg:
            bc = cfg['brightness_contrast']
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=bc.get('brightness_limit', 0.1),
                contrast_limit=bc.get('contrast_limit', 0.1),
                p=bc.get('p', 0.3)
            ))
        
        if 'gaussian_noise' in cfg:
            gn = cfg['gaussian_noise']
            transforms.append(A.GaussNoise(
                std_limit=gn.get('std_limit', (10, 30)),
                per_channel=True,
                p=gn.get('p', 0.2)
            ))
        
        return A.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using computed band statistics.
        
        Args:
            image: (C, H, W) array with raw reflectance values
            
        Returns:
            Normalized image in [0, 1] range
        """
        # Clip extreme values
        image = np.clip(image, 0, 10000).astype(np.float32)
        
        # Standardize per band
        for i in range(NUM_BANDS):
            image[i] = (image[i] - self.band_means[i]) / (self.band_stds[i] + 1e-6)
        
        # Clip to reasonable range and scale to [0, 1]
        image = np.clip(image, -3, 3)
        image = (image + 3) / 6
        
        return image
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Returns:
            image: (10, H, W) normalized tensor
            label: (1, H, W) severity tensor [0-1]
            metadata: dict with fire_key, stage, path
        """
        sample = self.samples[index]
        
        # Load tile
        with rasterio.open(sample['path']) as src:
            data = src.read()
        
        # Handle size issues (pad or crop to 256x256)
        _, h, w = data.shape
        if h != 256 or w != 256:
            padded = np.zeros((data.shape[0], 256, 256), dtype=data.dtype)
            actual_h = min(h, 256)
            actual_w = min(w, 256)
            padded[:, :actual_h, :actual_w] = data[:, :actual_h, :actual_w]
            data = padded
        
        # Split spectral bands and label
        image = data[:NUM_BANDS]  # (10, 256, 256)
        label = data[NUM_BANDS]   # (256, 256)
        
        # Handle invalid values
        image = np.nan_to_num(image, nan=0.0, posinf=10000.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clip label to valid range
        label = np.clip(label, 0.0, 1.0).astype(np.float32)
        
        # Normalize image
        image = self.normalize(image)
        
        # Apply augmentation
        if self.transform is not None:
            # Transpose for albumentations: (C, H, W) -> (H, W, C)
            image_hwc = image.transpose(1, 2, 0)
            
            augmented = self.transform(image=image_hwc, mask=label)
            
            image = augmented['image'].transpose(2, 0, 1)  # Back to (C, H, W)
            label = augmented['mask']
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)  # (1, H, W)
        
        # Metadata for tracking
        metadata = {
            'fire_key': sample['fire_key'],
            'stage': sample['stage'],
            'path': sample['path'],
        }
        
        return image_tensor, label_tensor, metadata


class CaliforniaFireDatasetSimple(Dataset):
    """
    Simplified dataset that returns only (image, label) without metadata.
    Compatible with standard PyTorch training loops.
    """
    
    def __init__(self, *args, **kwargs):
        self.base_dataset = CaliforniaFireDataset(*args, **kwargs)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        image, label, _ = self.base_dataset[index]
        return image, label


# ============================================================
# DATA LOADING UTILITIES
# ============================================================
def create_train_val_datasets(
    data_dirs: List[str],
    val_split: float = 0.15,
    test_fires: Optional[List[str]] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create training, validation, and test datasets.
    
    Args:
        data_dirs: Base data directories
        val_split: Fraction of training fires for validation
        test_fires: Fire keys to hold out for testing
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if test_fires is None:
        test_fires = list(TEST_FIRES.keys())
    
    training_fire_keys = list(TRAINING_FIRES.keys())
    
    # Split training fires into train/val
    np.random.seed(42)
    np.random.shuffle(training_fire_keys)
    
    n_val = max(1, int(len(training_fire_keys) * val_split))
    val_fires = training_fire_keys[:n_val]
    train_fires = training_fire_keys[n_val:]
    
    print(f"📊 Data Split:")
    print(f"   Train fires: {train_fires}")
    print(f"   Val fires: {val_fires}")
    print(f"   Test fires: {test_fires}")
    
    # Create datasets
    train_dataset = CaliforniaFireDatasetSimple(
        data_dirs,
        mode='train',
        augment=True,
        include_only_fires=train_fires + ['healthy'],
    )
    
    val_dataset = CaliforniaFireDatasetSimple(
        data_dirs,
        mode='val',
        augment=False,
        include_only_fires=val_fires,
    )
    
    test_dataset = CaliforniaFireDataset(  # Full version with metadata
        data_dirs,
        mode='test',
        augment=False,
        include_only_fires=test_fires,
    )
    
    return train_dataset, val_dataset, test_dataset


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing CaliforniaFireDataset")
    print("=" * 60)
    
    # Test with sample data
    data_dirs = [
        str(RAW_DATA_DIR / "fires"),
        str(RAW_DATA_DIR / "healthy"),
    ]
    
    # Check if data exists
    has_data = any(Path(d).exists() for d in data_dirs)
    
    if not has_data:
        print("\n⚠️ No data found. Download data first:")
        print("   python download_fire_data.py")
    else:
        # Create dataset
        dataset = CaliforniaFireDataset(
            data_dirs,
            mode='train',
            augment=True,
        )
        
        print(f"\n✅ Dataset created: {len(dataset)} samples")
        
        # Test loading
        if len(dataset) > 0:
            image, label, meta = dataset[0]
            print(f"\n   Sample shape:")
            print(f"      Image: {image.shape}")
            print(f"      Label: {label.shape}")
            print(f"      Metadata: {meta}")
            
            print(f"\n   Value ranges:")
            print(f"      Image: [{image.min():.3f}, {image.max():.3f}]")
            print(f"      Label: [{label.min():.3f}, {label.max():.3f}]")
    
    print("\n" + "=" * 60)
