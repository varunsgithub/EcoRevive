"""This script will load the files from the directory, and make a list fo all images in the directory.
It then fetches the image (Band 1 - 10) and the target (Band 11) and returns the Normalized version of the same"""

import os
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset

class EcoReviveDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        if isinstance (image_dirs, str):
            image_dirs = [image_dirs]
        
        """For each file in the path, add the file paths to .tif files in the folder."""
        self.images = []
        for pa in image_dirs:
            files = [os.path.join(pa, f) for f in os.listdir(pa) if f.endswith('.tif')]
            self.images.extend(files)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]

        with rasterio.open(img_path) as src:
            data = src.read()

            _, h, w = data.shape
            if h != 256 or w != 256:
                # Create a blank black canvas of the correct size
                padded_data = np.zeros((11, 256, 256), dtype=data.dtype)
                # Copy the small image into the top-left corner
                actual_h = min(h, 256)
                actual_w = min(w, 256)
                padded_data[:, :actual_h, :actual_w] = data[:, :actual_h, :actual_w]
                data = padded_data

            #Extracting band 1-10 from the image.
            image = data[0:10, :, :]
            
            #Band 11 is the label (Target)
            mask = data[10, :, :]

            # PREPROCESSING (Normalization)
            # PS: Normalizing all the images (by 3000, to have a standardized mean and variance)
            image = image.astype(np.float32) / 3000.0
            image = np.clip(image, 0, 1) # Force any stray pixels >1 back to 1
            
            # TENSOR CONVERSION
            # PyTorch expects Float32 for images and Long (Integers) for masks
            image_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask).long()

            return image_tensor, mask_tensor