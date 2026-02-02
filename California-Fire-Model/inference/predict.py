"""
Inference Script for California Fire Model
Run predictions on new Sentinel-2 imagery.
"""

import sys
import argparse
import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Tuple

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import NUM_BANDS, get_band_stats, MODEL_CONFIG
from model.architecture import CaliforniaFireModel, load_model


class FirePredictor:
    """Handles inference for burn severity prediction."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        use_tta: bool = True,
        temperature: float = 0.5,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            use_tta: Whether to use test-time augmentation
            temperature: Prediction sharpening factor (0.1-1.0). Lower = more confident.
                         Default 0.5 for sharper predictions. Use 1.0 for original output.
        """
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
                
        # Load model
        self.model = load_model(checkpoint_path, device=str(self.device), **MODEL_CONFIG)
        self.model.eval()
                
        # Normalization stats
        self.band_means, self.band_stds = get_band_stats()
        self.band_means = np.array(self.band_means, dtype=np.float32)
        self.band_stds = np.array(self.band_stds, dtype=np.float32)
        
        # TTA
        self.use_tta = use_tta
        
        # Temperature scaling for confidence adjustment
        self.temperature = max(0.1, min(temperature, 2.0))  # Clamp to reasonable range
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using computed band statistics."""
        image = np.clip(image, 0, 10000).astype(np.float32)
        
        for i in range(min(NUM_BANDS, image.shape[0])):
            image[i] = (image[i] - self.band_means[i]) / (self.band_stds[i] + 1e-6)
        
        image = np.clip(image, -3, 3)
        image = (image + 3) / 6
        
        return image
    
    def predict_tile(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict burn severity for a single tile.
        
        Args:
            image: (C, H, W) raw Sentinel-2 reflectance, C >= 10
            
        Returns:
            severity: (H, W) burn severity map [0, 1]
            confidence: Average prediction confidence
        """
        # Ensure correct shape
        if image.shape[0] < NUM_BANDS:
            raise ValueError(f"Expected at least {NUM_BANDS} bands, got {image.shape[0]}")
        
        image = image[:NUM_BANDS]  # Take only spectral bands
        
        # Handle NaN/Inf
        image = np.nan_to_num(image, nan=0.0, posinf=10000.0, neginf=0.0)
        
        # Normalize
        image = self.normalize(image)
        
        # To tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_tta:
                # Test-time augmentation: average LOGITS from multiple transforms
                # Temperature scaling is applied after averaging for better results
                logits = []
                
                # Original
                logit = self.model(image_tensor)
                logits.append(logit)
                
                # Horizontal flip
                logit_hflip = self.model(torch.flip(image_tensor, [3]))
                logit_hflip = torch.flip(logit_hflip, [3])
                logits.append(logit_hflip)
                
                # Vertical flip
                logit_vflip = self.model(torch.flip(image_tensor, [2]))
                logit_vflip = torch.flip(logit_vflip, [2])
                logits.append(logit_vflip)
                
                # Rotate 90
                logit_rot = self.model(torch.rot90(image_tensor, 1, [2, 3]))
                logit_rot = torch.rot90(logit_rot, -1, [2, 3])
                logits.append(logit_rot)
                
                # Average logits, then apply temperature scaling
                avg_logit = torch.stack(logits).mean(0)
                severity = torch.sigmoid(avg_logit / self.temperature)
            else:
                logit = self.model(image_tensor)
                # Temperature scaling: divide logits to sharpen predictions
                severity = torch.sigmoid(logit / self.temperature)
            
            severity = severity.cpu().numpy()[0, 0]
        
        # Confidence: distance from 0.5 (uncertain predictions)
        confidence = np.abs(severity - 0.5).mean() * 2
        
        return severity, confidence
    
    def predict_file(self, input_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Predict burn severity for a GeoTIFF file.
        
        Args:
            input_path: Path to input GeoTIFF
            output_path: Optional path to save output GeoTIFF
            
        Returns:
            severity: (H, W) burn severity map
            metadata: Dict with prediction stats
        """
        
        with rasterio.open(input_path) as src:
            image = src.read()
            profile = src.profile.copy()
            
            original_shape = image.shape[1:]  # (H, W)
        
        # Handle different sizes
        h, w = original_shape
        
        if h == 256 and w == 256:
            # Standard tile
            severity, confidence = self.predict_tile(image)
        else:
            # Sliding window for larger images
            severity = self._predict_sliding_window(image)
            confidence = np.abs(severity - 0.5).mean() * 2
        
        # Stats
        metadata = {
            'mean_severity': float(severity.mean()),
            'max_severity': float(severity.max()),
            'burned_ratio': float((severity > 0.5).mean()),
            'confidence': float(confidence),
            'shape': list(severity.shape),
        }
        
        print(f"   Mean severity: {metadata['mean_severity']:.1%}")
        print(f"   Burned pixels: {metadata['burned_ratio']:.1%}")
        print(f"   Confidence: {metadata['confidence']:.1%}")
        
        # Save output
        if output_path:
            profile.update(
                count=1,
                dtype='float32',
                compress='lzw',
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(severity.astype(np.float32), 1)
            
            print(f" Saved: {output_path}")
        
        return severity, metadata
    
    def _predict_sliding_window(
        self,
        image: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32,
    ) -> np.ndarray:
        """Predict large images using sliding window."""
        c, h, w = image.shape
        
        # Output accumulator
        output = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        step = tile_size - overlap
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = image[:, y:y_end, x:x_end]
                
                # Pad if necessary
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = np.zeros((c, tile_size, tile_size), dtype=tile.dtype)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                
                # Predict
                pred, _ = self.predict_tile(tile)
                
                # Crop back
                pred = pred[:y_end-y, :x_end-x]
                
                # Accumulate
                output[y:y_end, x:x_end] += pred
                counts[y:y_end, x:x_end] += 1
        
        # Average overlapping regions
        output /= np.maximum(counts, 1)
        
        return output


def main(args):
    """Main inference function."""
    
    # Check input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f" Input not found: {input_path}")
        return
    
    # Create predictor
    predictor = FirePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_tta=not args.no_tta,
        temperature=args.temperature,
    )
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"{input_path.stem}_severity.tif")
    
    # Predict
    if input_path.is_file():
        severity, metadata = predictor.predict_file(str(input_path), output_path)
        
        print("\nResults:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    
    elif input_path.is_dir():
        # Process all .tif files in directory
        tif_files = list(input_path.glob("*.tif"))
        print(f"\n Processing {len(tif_files)} files...")
        
        output_dir = Path(args.output) if args.output else input_path / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        for tif_path in tif_files:
            out_path = output_dir / f"{tif_path.stem}_severity.tif"
            try:
                predictor.predict_file(str(tif_path), str(out_path))
            except Exception as e:
                print(f"  Error processing {tif_path.name}: {e}")
    
    print("\n Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run burn severity prediction')
    
    parser.add_argument('input', type=str,
                        help='Input GeoTIFF file or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (file or directory)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Prediction temperature (0.1-1.0). Lower = more confident. Default: 0.5')
    
    args = parser.parse_args()
    
    main(args)
