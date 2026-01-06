import os
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class EcoReviveDataset(Dataset):
    def __init__(self, image_dirs, augment=True, mode='train'):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        
        # Collect all .tif files
        self.images = []
        for path in image_dirs:
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
            self.images.extend(files)
        
        print(f"📂 Loaded {len(self.images)} images from {len(image_dirs)} directories")
        
        self.mode = mode
        self.augment = augment
        
        # Sentinel-2 L2A typical value ranges (approximate percentiles)
        # These are better than simple /10000 normalization
        self.band_means = np.array([1500, 1400, 1300, 1200, 1500, 2000, 2200, 2300, 2500, 800], dtype=np.float32)
        self.band_stds = np.array([1000, 1100, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 900], dtype=np.float32)
        
        # Training augmentation (OPTIMIZED - less heavy for faster loading)
        if augment and mode == 'train':
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-30, 30), p=0.5),
                
                # Optical augmentations (lighter)
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
                A.GaussNoise(std_limit=(10.0, 40.0), per_channel=True, p=0.25),
                
                # Spatial augmentations (lighter)
                A.CoarseDropout(
                    num_holes_range=(3, 6),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    p=0.3
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            ])
        # Validation augmentation (lighter for TTA)
        elif augment and mode == 'val':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.images)
    
    def normalize_image(self, image):
        """Improved normalization using band statistics"""
        # Clip extreme values first (remove outliers)
        image = np.clip(image, 0, 10000)
        
        # Standardize per band
        image = image.astype(np.float32)
        for i in range(10):
            image[i] = (image[i] - self.band_means[i]) / (self.band_stds[i] + 1e-6)
        
        # Clip to reasonable range
        image = np.clip(image, -3, 3)
        
        # Scale to [0, 1] for stability
        image = (image + 3) / 6
        
        return image
    
    def __getitem__(self, index):
        img_path = self.images[index]
        
        with rasterio.open(img_path) as src:
            data = src.read()  # Shape: (11, H, W)
            
            # Pad to 256×256 if needed
            _, h, w = data.shape
            if h != 256 or w != 256:
                padded_data = np.zeros((11, 256, 256), dtype=data.dtype)
                actual_h = min(h, 256)
                actual_w = min(w, 256)
                padded_data[:, :actual_h, :actual_w] = data[:, :actual_h, :actual_w]
                data = padded_data
        
        # Extract bands 1-10 (Sentinel-2) and band 11 (label)
        image = data[0:10, :, :]  # (10, 256, 256)
        mask = data[10, :, :]     # (256, 256)
        
        # Handle NaN/Inf values
        image = np.nan_to_num(image, nan=0.0, posinf=10000.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0)
        
        # Apply improved normalization
        image = self.normalize_image(image)
        
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply augmentation
        if self.transform:
            image_hwc = image.transpose(1, 2, 0)  # (10, 256, 256) → (256, 256, 10)
            augmented = self.transform(image=image_hwc, mask=mask)
            image = augmented['image'].transpose(2, 0, 1)  # → (10, 256, 256)
            mask = augmented['mask']
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        return image_tensor, mask_tensor