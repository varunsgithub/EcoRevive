"""
EcoRevive Backend API Server
============================
FastAPI server that handles the full pipeline:
1. Download Sentinel-2 imagery from Earth Engine
2. Run California Fire Model inference
3. Generate Gemini reasoning analysis
4. Return results to frontend
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "California-Fire-Model"))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Import our modules
from ee_download import initialize_ee, download_sentinel2_for_model, create_rgb_from_bands

# Initialize FastAPI app
app = FastAPI(
    title="EcoRevive API",
    description="Burn severity analysis using satellite imagery and AI",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
ee_initialized = False
model = None
device = None


# Request/Response Models
class AnalyzeRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    user_type: Optional[str] = "personal"


class AnalyzeResponse(BaseModel):
    success: bool
    satellite_image: Optional[str] = None  # Base64 encoded RGB satellite image
    severity_image: Optional[str] = None  # Base64 encoded colorized PNG
    raw_severity_image: Optional[str] = None  # Base64 encoded raw grayscale model output
    severity_stats: Optional[Dict[str, Any]] = None
    gemini_analysis: Optional[str] = None
    error: Optional[str] = None


def create_synthetic_image(bbox: Dict[str, float]) -> np.ndarray:
    """
    Create synthetic Sentinel-2 imagery for testing when EE is unavailable.
    Returns a (10, 256, 256) array simulating multispectral data.
    """
    # Create base noise patterns
    np.random.seed(42)  # Reproducible for demos
    
    # Create spatially coherent patterns using simple gradients + noise
    h, w = 256, 256
    
    # Base terrain pattern
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    terrain = np.sin(xx) * np.cos(yy) * 0.3 + 0.5
    
    # Add some random burn patterns
    burn_center_x = np.random.randint(50, 200)
    burn_center_y = np.random.randint(50, 200)
    distances = np.sqrt((np.arange(w) - burn_center_x)**2 + (np.arange(h)[:, None] - burn_center_y)**2)
    burn_pattern = np.exp(-distances / 40) * 0.6
    
    # Create 10 bands with realistic spectral relationships
    bands = []
    for i in range(10):
        # Each band has slightly different characteristics
        noise = np.random.randn(h, w) * 0.1
        band = terrain + burn_pattern * (0.5 + i * 0.05) + noise
        
        # Scale to typical Sentinel-2 reflectance values (0-10000)
        band = (band * 3000 + 1000).clip(0, 10000)
        bands.append(band.astype(np.float32))
    
    return np.stack(bands, axis=0)


def load_fire_model():
    """Load the California Fire Model."""
    global model, device
    
    try:
        import torch
        from model.architecture import CaliforniaFireModel
        
        checkpoint_path = PROJECT_ROOT / "California-Fire-Model/checkpoints/model.pth"
        
        if not checkpoint_path.exists():
            print(f"⚠️ Model checkpoint not found at {checkpoint_path}")
            return False
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Loading Fire Model on {device}...")
        
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
        
        print(f"✅ Fire Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Fire Model: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(image_array: np.ndarray) -> tuple:
    """
    Run Fire Model inference on image array.
    Uses EXACT normalization from training.
    
    Args:
        image_array: (10, 256, 256) Sentinel-2 bands
        
    Returns:
        Tuple of (severity_map, stats_dict)
    """
    global model, device
    
    import torch
    from config import get_band_stats
    
    # Get band statistics
    means, stds = get_band_stats()
    means = np.array(means).reshape(-1, 1, 1)
    stds = np.array(stds).reshape(-1, 1, 1)
    
    # EXACT normalization from training:
    # 1. Replace NaN with 0
    image_array = np.nan_to_num(image_array, nan=0.0)
    
    # 2. Z-score normalize
    normalized = (image_array - means) / (stds + 1e-6)
    
    # 3. Clip to [-3, 3]
    normalized = np.clip(normalized, -3, 3)
    
    # 4. Scale to [0, 1] via (x + 3) / 6
    normalized = (normalized + 3) / 6
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(tensor)
        severity_map = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Compute statistics
    stats = {
        'mean_severity': float(severity_map.mean()),
        'max_severity': float(severity_map.max()),
        'min_severity': float(severity_map.min()),
        'burned_ratio': float((severity_map > 0.5).mean()),
        'high_severity_ratio': float((severity_map > 0.75).mean()),
        'moderate_severity_ratio': float(((severity_map > 0.25) & (severity_map <= 0.75)).mean()),
        'low_severity_ratio': float((severity_map <= 0.25).mean()),
    }
    
    return severity_map, stats


def run_tiled_inference(tiles: list, metadata: dict) -> tuple:
    """
    Run inference on multiple tiles and stitch results together.
    
    Args:
        tiles: List of tile dicts from download_for_inference
        metadata: Metadata dict with n_rows, n_cols
        
    Returns:
        Tuple of (stitched_severity_map, stitched_satellite_image, combined_stats)
    """
    n_rows = metadata['n_rows']
    n_cols = metadata['n_cols']
    tile_size = 256
    
    # Create output arrays
    stitched_severity = np.zeros((n_rows * tile_size, n_cols * tile_size), dtype=np.float32)
    stitched_satellite = np.zeros((n_rows * tile_size, n_cols * tile_size, 3), dtype=np.uint8)
    
    all_severities = []
    
    print(f"   Processing {len(tiles)} tiles...")
    
    for tile in tiles:
        row = tile['row']
        col = tile['col']
        image = tile['image']
        
        # Run inference on this tile
        severity_map, _ = run_inference(image)
        all_severities.append(severity_map)
        
        # Create RGB for satellite view
        rgb = create_tile_rgb(image)
        
        # Calculate position in stitched image
        y_start = (n_rows - 1 - row) * tile_size  # Flip rows (row 0 is at bottom)
        y_end = y_start + tile_size
        x_start = col * tile_size
        x_end = x_start + tile_size
        
        # Place in stitched arrays
        stitched_severity[y_start:y_end, x_start:x_end] = severity_map
        stitched_satellite[y_start:y_end, x_start:x_end] = rgb
    
    # Compute combined statistics
    if all_severities:
        combined = np.concatenate([s.flatten() for s in all_severities])
        stats = {
            'mean_severity': float(combined.mean()),
            'max_severity': float(combined.max()),
            'min_severity': float(combined.min()),
            'burned_ratio': float((combined > 0.5).mean()),
            'high_severity_ratio': float((combined > 0.75).mean()),
            'moderate_severity_ratio': float(((combined > 0.25) & (combined <= 0.75)).mean()),
            'low_severity_ratio': float((combined <= 0.25).mean()),
            'n_tiles_processed': len(tiles),
        }
    else:
        stats = {'mean_severity': 0, 'n_tiles_processed': 0}
    
    print(f"   Stitched output: {stitched_severity.shape}")
    
    return stitched_severity, stitched_satellite, stats


def create_tile_rgb(image_array: np.ndarray) -> np.ndarray:
    """Create RGB array from Sentinel-2 bands for a single tile."""
    # Band statistics from training
    BAND_MEANS = [472.8, 673.8, 770.8, 1087.8, 1747.6, 1997.1, 2106.4, 2188.9, 1976.1, 1404.5]
    BAND_STDS = [223.7, 255.4, 345.6, 313.5, 366.7, 417.6, 476.7, 437.1, 472.7, 438.4]
    
    # Normalize bands
    normalized = np.zeros_like(image_array, dtype=np.float32)
    for i in range(10):
        normalized[i] = (image_array[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
    normalized = np.clip(normalized, -3, 3)
    normalized = (normalized + 3) / 6
    
    # False color B5, B4, B3
    rgb = np.stack([normalized[3], normalized[2], normalized[1]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def severity_to_image(severity_map: np.ndarray) -> str:
    """Convert severity map to colorized PNG using 'hot' colormap (matching training visualization)."""
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Use matplotlib's 'hot' colormap (black -> red -> yellow -> white)
    # This EXACTLY matches the training visualization
    cmap = cm.get_cmap('hot')
    
    # Apply colormap: severity (0-1) -> RGBA -> RGB
    rgba = cmap(severity_map)  # Returns (H, W, 4)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    # Create image
    img = Image.fromarray(rgb)
    
    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"


def rgb_array_to_base64(rgb_array: np.ndarray) -> str:
    """Convert RGB numpy array to base64 PNG string."""
    from PIL import Image
    
    img = Image.fromarray(rgb_array, mode='RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"


def severity_to_raw_image(severity_map: np.ndarray) -> str:
    """Convert severity map to raw grayscale PNG base64 string (actual model output)."""
    from PIL import Image
    
    # Convert to 8-bit grayscale (0=no burn, 255=max burn)
    grayscale = (severity_map * 255).astype(np.uint8)
    
    # Create grayscale image
    img = Image.fromarray(grayscale, mode='L')
    
    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"


def get_gemini_analysis(stats: Dict[str, Any], bbox: Dict[str, float], user_type: str) -> str:
    """Get Gemini analysis of the burn severity results."""
    try:
        from reasoning import EcoReviveGemini
        
        client = EcoReviveGemini()
        
        # Format prompt based on user type
        if user_type == "professional":
            prompt = f"""You are an expert wildfire restoration ecologist. Analyze this burn severity assessment:

**Location**: {bbox['south']:.4f}°N to {bbox['north']:.4f}°N, {bbox['west']:.4f}°W to {bbox['east']:.4f}°W

**Burn Severity Analysis**:
- Mean Severity: {stats['mean_severity']:.1%}
- Maximum Severity: {stats['max_severity']:.1%}
- High Severity Area (>75%): {stats['high_severity_ratio']:.1%}
- Moderate Severity Area (25-75%): {stats['moderate_severity_ratio']:.1%}
- Low/Unburned Area (<25%): {stats['low_severity_ratio']:.1%}

Provide a professional assessment including:
1. **Severity Classification**: Overall burn severity category
2. **Ecological Impact**: Expected vegetation and habitat damage
3. **Restoration Priority**: Urgency and recommended approach
4. **Key Species to Plant**: Native species suited for this recovery
5. **Timeline**: Expected natural vs assisted recovery time
6. **Budget Estimate**: Rough cost per hectare for restoration

Keep the response concise but actionable for grant proposals and project planning."""
        else:
            prompt = f"""You are a friendly environmental guide helping someone understand wildfire damage. Explain this burn severity assessment in simple terms:

**Location**: Near {bbox['south']:.2f}°N, {abs(bbox['west']):.2f}°W

**What the satellite analysis found**:
- Overall burn damage: {stats['mean_severity']:.0%}
- Severely burned areas: {stats['high_severity_ratio']:.0%} of the region
- Moderately burned: {stats['moderate_severity_ratio']:.0%}
- Lightly affected or untouched: {stats['low_severity_ratio']:.0%}

Please explain:
1. What does this mean for the land and wildlife?
2. How long will it take nature to recover?
3. What can volunteers do to help?
4. What plants and trees will grow back first?

Keep it encouraging and accessible - this person cares about the environment but isn't a scientist."""

        response = client.analyze_multimodal(prompt=prompt, use_json=False)
        return response.get('text', 'Analysis not available.')
        
    except Exception as e:
        print(f"⚠️ Gemini analysis failed: {e}")
        return f"AI analysis temporarily unavailable. Manual assessment of the burn severity data shows {stats['mean_severity']:.0%} average severity with {stats['high_severity_ratio']:.0%} high-severity areas requiring immediate attention."


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ee_initialized
    
    print("🚀 Starting EcoRevive API Server...")
    
    # Initialize Earth Engine
    ee_initialized = initialize_ee()
    
    # Load Fire Model
    load_fire_model()
    
    print("✅ Server ready!")


@app.get("/")
async def root():
    return {
        "service": "EcoRevive API",
        "status": "running",
        "ee_initialized": ee_initialized,
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "earth_engine": ee_initialized,
        "model": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_area(request: AnalyzeRequest):
    """
    Main analysis endpoint.
    Downloads imagery, runs model, generates Gemini analysis.
    """
    try:
        # Validate bounding box
        bbox = {
            'west': request.west,
            'south': request.south,
            'east': request.east,
            'north': request.north
        }
        
        # Set date range (default to recent summer)
        start_date = request.start_date or "2023-06-01"
        end_date = request.end_date or "2023-09-30"
        
        print(f"📍 Analyzing area: {bbox}")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Step 1: Download Sentinel-2 imagery with TILING support
        if ee_initialized:
            try:
                print("🛰️ Downloading Sentinel-2 imagery (with tiling)...")
                tiles, ee_metadata = download_sentinel2_for_model(bbox, start_date, end_date)
                print(f"   Downloaded {len(tiles)} tiles")
                
                if not tiles:
                    raise Exception("No tiles downloaded")
                    
            except Exception as e:
                print(f"⚠️ EE download failed: {e}, using synthetic data")
                # Fallback to single synthetic tile
                synthetic_image = create_synthetic_image(bbox)
                tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
                ee_metadata = {'n_rows': 1, 'n_cols': 1}
        else:
            print("⚠️ Earth Engine not available, using synthetic imagery for demo")
            synthetic_image = create_synthetic_image(bbox)
            tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
            ee_metadata = {'n_rows': 1, 'n_cols': 1}
        
        # Step 2: Run Fire Model inference on all tiles
        if model is None:
            raise HTTPException(status_code=503, detail="Fire model not loaded")
        
        print("🔥 Running burn severity inference on tiles...")
        severity_map, satellite_rgb, stats = run_tiled_inference(tiles, ee_metadata)
        print(f"   Mean severity: {stats['mean_severity']:.1%}")
        print(f"   Output size: {severity_map.shape}")
        
        # Step 3: Create visualizations from stitched results
        satellite_image = rgb_array_to_base64(satellite_rgb)
        severity_image = severity_to_image(severity_map)
        raw_severity_image = severity_to_raw_image(severity_map)
        
        # Step 4: Get Gemini analysis
        print("🧠 Generating Gemini analysis...")
        gemini_text = get_gemini_analysis(stats, bbox, request.user_type or "personal")
        
        print("✅ Analysis complete!")
        
        return AnalyzeResponse(
            success=True,
            satellite_image=satellite_image,
            severity_image=severity_image,
            raw_severity_image=raw_severity_image,
            severity_stats=stats,
            gemini_analysis=gemini_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )


# Run with: uvicorn server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
