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

# Enable CORS for frontend (local + production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://ecorevive.vercel.app",
        "https://ecorevive-original.vercel.app",
        "*",  # Allow all origins for hackathon demo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
ee_initialized = False
model = None
device = None
rag_system = None  # RAG system for knowledge-grounded responses


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
    gemini_analysis: Optional[str] = None  # Legacy verbose text for display
    layer2_output: Optional[Dict[str, Any]] = None  # Structured Layer 2 JSON
    layer3_context: Optional[Dict[str, Any]] = None  # Layer 3 contextual analysis (urban detection, cautions)
    carbon_analysis: Optional[Dict[str, Any]] = None  # Carbon sequestration calculator
    error: Optional[str] = None


class Layer2AnalyzeResponse(BaseModel):
    """Structured Layer 2 output for Layer 3 consumption."""
    success: bool
    layer2_output: Optional[Dict[str, Any]] = None  # Full structured Layer 2 data
    satellite_image: Optional[str] = None  # Base64 encoded RGB for reference
    severity_image: Optional[str] = None  # Base64 encoded colorized severity
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for the AI chat endpoint."""
    message: str  # User's message or quick action prompt
    action_type: Optional[str] = None  # Quick action ID (safety, hope, species, etc.)
    user_type: str = "personal"  # personal or professional
    context: Optional[Dict[str, Any]] = None  # Analysis context (layer2_output, severity_stats, bbox)


class ChatResponse(BaseModel):
    """Response from the AI chat endpoint."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


class PDFExportRequest(BaseModel):
    """Request for PDF export endpoint."""
    satellite_image: str
    severity_image: str
    severity_stats: Dict[str, Any]
    bbox: Dict[str, float]
    layer2_output: Optional[Dict[str, Any]] = None
    layer3_context: Optional[Dict[str, Any]] = None
    carbon_analysis: Optional[Dict[str, Any]] = None
    report_type: str = "personal"
    user_type: str = "personal"
    location_name: Optional[str] = None
    analysis_id: Optional[str] = None


class PDFExportResponse(BaseModel):
    """Response from PDF export endpoint."""
    success: bool
    pdf_base64: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None


class WordExportRequest(BaseModel):
    """Request for Word export endpoint."""
    satellite_image: str
    severity_image: str
    severity_stats: Dict[str, Any]
    bbox: Dict[str, float]
    layer2_output: Optional[Dict[str, Any]] = None
    layer3_context: Optional[Dict[str, Any]] = None
    carbon_analysis: Optional[Dict[str, Any]] = None
    report_type: str = "personal"
    user_type: str = "personal"
    location_name: Optional[str] = None


class WordExportResponse(BaseModel):
    """Response from Word export endpoint."""
    success: bool
    docx_base64: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None


class HopeVisualizationRequest(BaseModel):
    """Request for hope visualization endpoint."""
    ecosystem_type: str = "mixed_conifer"
    years_in_future: int = 15
    mean_severity: float = 0.5
    area_hectares: float = 100.0
    bbox: Optional[Dict[str, float]] = None


class HopeVisualizationResponse(BaseModel):
    """Response from hope visualization endpoint."""
    success: bool
    image_base64: Optional[str] = None
    forecast: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
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
            print(f"[WARNING] Model checkpoint not found at {checkpoint_path}")
            return False
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading Fire Model on {device}...")
        
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
        
        print(f"[OK] Fire Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load Fire Model: {e}")
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
    
    # Standardized Thresholds (matching reasoning engine)
    THRESH_HIGH = 0.66
    THRESH_MODERATE = 0.27
    THRESH_LOW = 0.1
    
    # Basic masks
    pixel_count = severity_map.size
    burned_mask = severity_map > THRESH_LOW
    burned_pixel_count = np.sum(burned_mask)
    
    # Compute statistics
    stats = {
        # Raw field-wide stats (legacy compatibility)
        'mean_severity': float(severity_map.mean()),
        'max_severity': float(severity_map.max()),
        'min_severity': float(severity_map.min()),
        'burned_ratio': float(np.mean(severity_map > 0.5)), # Keep legacy definition for now
        
        # Standardized Ratios (Field-wide)
        'high_severity_ratio': float(np.mean(severity_map > THRESH_HIGH)),
        'moderate_severity_ratio': float(np.mean((severity_map > THRESH_MODERATE) & (severity_map <= THRESH_HIGH))),
        'low_severity_ratio': float(np.mean((severity_map <= THRESH_MODERATE))),
        
        # Ecologically Valid Metrics (Within Burn Scar)
        'burned_area_ratio': float(burned_pixel_count / pixel_count) if pixel_count > 0 else 0,
        'mean_severity_in_burn_area': float(severity_map[burned_mask].mean()) if burned_pixel_count > 0 else 0.0,
        
        'high_severity_in_burn_area': float(np.sum(severity_map > THRESH_HIGH) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
        'moderate_severity_in_burn_area': float(np.sum((severity_map > THRESH_MODERATE) & (severity_map <= THRESH_HIGH)) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
        'low_severity_in_burn_area': float(np.sum((severity_map > THRESH_LOW) & (severity_map <= THRESH_MODERATE)) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
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
        
        # Standardized Thresholds
        THRESH_HIGH = 0.66
        THRESH_MODERATE = 0.27
        THRESH_LOW = 0.1
        
        pixel_count = combined.size
        burned_mask = combined > THRESH_LOW
        burned_pixel_count = np.sum(burned_mask)
        
        stats = {
            # Raw field-wide
            'mean_severity': float(combined.mean()),
            'max_severity': float(combined.max()),
            'min_severity': float(combined.min()),
            'burned_ratio': float((combined > 0.5).mean()),
            'n_tiles_processed': len(tiles),
            
             # Standardized Ratios (Field-wide)
            'high_severity_ratio': float(np.mean(combined > THRESH_HIGH)),
            'moderate_severity_ratio': float(np.mean((combined > THRESH_MODERATE) & (combined <= THRESH_HIGH))),
            'low_severity_ratio': float(np.mean((combined <= THRESH_MODERATE))),
            
            # Ecologically Valid Metrics (Within Burn Scar)
            'burned_area_ratio': float(burned_pixel_count / pixel_count) if pixel_count > 0 else 0,
            'mean_severity_in_burn_area': float(combined[burned_mask].mean()) if burned_pixel_count > 0 else 0.0,
            
            'high_severity_in_burn_area': float(np.sum(combined > THRESH_HIGH) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
            'moderate_severity_in_burn_area': float(np.sum((combined > THRESH_MODERATE) & (combined <= THRESH_HIGH)) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
            'low_severity_in_burn_area': float(np.sum((combined > THRESH_LOW) & (combined <= THRESH_MODERATE)) / burned_pixel_count) if burned_pixel_count > 0 else 0.0,
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


def get_gemini_analysis(
    stats: Dict[str, Any], 
    bbox: Dict[str, float], 
    user_type: str,
    satellite_rgb: np.ndarray = None,
    severity_map: np.ndarray = None
) -> Dict[str, Any]:
    """
    Get TRUE MULTIMODAL Gemini analysis of the burn severity results.
    
    This function sends BOTH the satellite image AND severity overlay to Gemini,
    enabling spatial reasoning about WHAT is burned, not just statistics.
    
    Args:
        stats: Burn severity statistics dict
        bbox: Bounding box dict with west, south, east, north
        user_type: "professional" or "personal"
        satellite_rgb: (H, W, 3) RGB satellite image array
        severity_map: (H, W) severity predictions from U-Net
        
    Returns:
        Dict with 'text' (formatted analysis) and 'analysis' (structured data)
    """
    try:
        # Check if we have images for true multimodal analysis
        if satellite_rgb is not None and severity_map is not None:
            return _get_multimodal_analysis(stats, bbox, user_type, satellite_rgb, severity_map)
        else:
            # Fallback to text-only analysis
            return _get_text_only_analysis(stats, bbox, user_type)
            
    except Exception as e:
        print(f"[WARNING] Gemini analysis failed: {e}")
        import traceback
        traceback.print_exc()
        fallback_text = (
            f"AI analysis temporarily unavailable. Manual assessment: "
            f"{stats['mean_severity']:.0%} average severity with "
            f"{stats['high_severity_ratio']:.0%} high-severity areas requiring attention."
        )
        return {'text': fallback_text, 'analysis': None}


def _get_multimodal_analysis(
    stats: Dict[str, Any],
    bbox: Dict[str, float],
    user_type: str,
    satellite_rgb: np.ndarray,
    severity_map: np.ndarray
) -> Dict[str, Any]:
    """
    TRUE MULTIMODAL analysis - sends images to Gemini for spatial reasoning.
    """
    from reasoning.gemini_multimodal import (
        MultimodalAnalyzer,
        create_image_pack,
        build_gemini_context
    )
    from reasoning import create_client
    
    print("   [INFO] Running TRUE MULTIMODAL analysis (sending images to Gemini)...")
    
    # Create Gemini client
    client = create_client()
    
    # Calculate center location
    center_lat = (bbox['north'] + bbox['south']) / 2
    center_lon = (bbox['west'] + bbox['east']) / 2
    location = (center_lat, center_lon)
    
    # Prepare RGB tile for multimodal - transpose from (H, W, 3) to (3, H, W)
    if satellite_rgb.ndim == 3 and satellite_rgb.shape[2] == 3:
        rgb_tile = np.moveaxis(satellite_rgb, -1, 0)
    else:
        rgb_tile = satellite_rgb
    
    # Build metadata for context
    metadata = {
        'bbox': bbox,
        'user_type': user_type,
        'fire_date': None,  # Could be passed from request
        'days_since_fire': None,
    }
    
    # Run multimodal analysis
    analyzer = MultimodalAnalyzer(client)
    result = analyzer.analyze(
        rgb_tile=rgb_tile,
        severity_map=severity_map,
        location=location,
        metadata=metadata,
        unet_confidence=0.85  # Assumed confidence
    )
    
    # Format the response for the frontend
    if result.get('status') == 'complete':
        analysis = result.get('analysis', {})
        
        # Build human-readable text from spatial analysis
        text_parts = []
        
        # Visual grounding section
        vg = analysis.get('visual_grounding', {})
        if vg:
            land_cover = ", ".join(vg.get('observed_land_cover', ['Unknown']))
            text_parts.append(f"## What Gemini Sees in the Satellite Image\n")
            text_parts.append(f"**Land Cover Types Detected:** {land_cover}\n")
            if vg.get('terrain_features'):
                text_parts.append(f"**Terrain Features:** {', '.join(vg.get('terrain_features', []))}\n")
            if vg.get('pre_fire_vegetation_description'):
                text_parts.append(f"**Vegetation Description:** {vg.get('pre_fire_vegetation_description')}\n")
        
        # Segmentation quality section
        sq = analysis.get('segmentation_quality', {})
        if sq:
            quality = sq.get('overall_quality', 'unknown')
            confidence = sq.get('confidence_in_prediction', 0)
            text_parts.append(f"\n## U-Net Prediction Quality Assessment\n")
            text_parts.append(f"**Overall Quality:** {quality.upper()}\n")
            text_parts.append(f"**Gemini's Confidence in Predictions:** {confidence:.0%}\n")
            if sq.get('artifact_flags'):
                text_parts.append(f"**[WARNING] Artifacts Detected:** {', '.join(sq.get('artifact_flags'))}\n")
            if sq.get('quality_notes'):
                text_parts.append(f"**Notes:** {sq.get('quality_notes')}\n")
        
        # Spatial patterns section
        sp = analysis.get('spatial_patterns', {})
        if sp:
            text_parts.append(f"\n## Spatial Pattern Analysis\n")
            
            frag = sp.get('fragmentation_assessment', {})
            if frag:
                text_parts.append(f"**Burn Patches Visible:** {frag.get('patch_count_visual', 'unknown')}\n")
                text_parts.append(f"**Connectivity:** {frag.get('connectivity', 'unknown')}\n")
                text_parts.append(f"**Shape Complexity:** {frag.get('shape_complexity', 'unknown')}\n")
            
            edge = sp.get('edge_characteristics', {})
            if edge:
                text_parts.append(f"**Edge Sharpness:** {edge.get('edge_sharpness', 'unknown')}\n")
                if edge.get('unburned_inclusions'):
                    text_parts.append(f"**Unburned Islands:** Yes - {edge.get('inclusion_significance', '')}\n")
            
            grad = sp.get('gradient_analysis', {})
            if grad:
                if grad.get('dominant_direction') and grad.get('dominant_direction') != 'none':
                    text_parts.append(f"**Fire Spread Direction:** {grad.get('dominant_direction')}\n")
                if grad.get('fire_behavior_inference'):
                    text_parts.append(f"**Fire Behavior Inference:** {grad.get('fire_behavior_inference')}\n")
        
        # Ecological interpretation
        ei = analysis.get('ecological_interpretation', {})
        if ei:
            text_parts.append(f"\n## Ecological Interpretation\n")
            text_parts.append(f"**Natural Regeneration Potential:** {ei.get('natural_regeneration_potential', 'unknown')}\n")
            text_parts.append(f"**Seed Source Availability:** {ei.get('seed_source_availability', 'unknown')}\n")
            if ei.get('regeneration_rationale'):
                text_parts.append(f"**Rationale:** {ei.get('regeneration_rationale')}\n")
            
            # Differential impacts
            impacts = ei.get('differential_impacts', [])
            if impacts:
                text_parts.append(f"\n**Differential Impacts:**\n")
                for impact in impacts[:3]:
                    text_parts.append(f"- {impact.get('vegetation_type', 'Unknown')}: "
                                     f"{impact.get('severity_level', '')} severity - "
                                     f"{impact.get('ecological_significance', '')}\n")
        
        # Priority zones section
        zones = analysis.get('priority_zones', [])
        if zones:
            text_parts.append(f"\n## Priority Restoration Zones\n")
            for zone in zones[:5]:
                urgency_label = {
                    'immediate': '[URGENT]',
                    '6_months': '[6 MONTHS]',
                    '1_year': '[1 YEAR]',
                    '2_3_years': '[2-3 YEARS]'
                }.get(zone.get('urgency', ''), '[--]')

                text_parts.append(f"\n### {urgency_label} Zone {zone.get('zone_id', '?')}: {zone.get('urgency', 'unknown').replace('_', ' ').title()}\n")
                text_parts.append(f"**Location:** {zone.get('location_description', 'See overlay')}\n")
                text_parts.append(f"**Severity:** {zone.get('severity', 'unknown')}\n")
                text_parts.append(f"**Why Priority:** {zone.get('priority_reason', 'N/A')}\n")
                text_parts.append(f"**Recommended Action:** {zone.get('recommended_intervention', 'N/A').replace('_', ' ').title()}\n")
        
        # Machine-readable signals section
        signals = analysis.get('signals_for_final_model', {})
        if signals:
            text_parts.append(f"\n## Restoration Scores (Machine-Readable)\n")
            text_parts.append(f"| Metric | Score |\n|--------|-------|\n")
            text_parts.append(f"| Restoration Potential | {signals.get('restoration_potential_score', 0):.2f} |\n")
            text_parts.append(f"| Intervention Urgency | {signals.get('intervention_urgency_score', 0):.2f} |\n")
            text_parts.append(f"| Ecological Complexity | {signals.get('ecological_complexity_score', 0):.2f} |\n")
            text_parts.append(f"| Risk Score | {signals.get('risk_score', 0):.2f} |\n")
        
        # Reasoning trace
        if analysis.get('reasoning_trace'):
            text_parts.append(f"\n---\n*Gemini's reasoning: {analysis.get('reasoning_trace')}*\n")
        
        formatted_text = "".join(text_parts)
        
        # Add human review warning if needed
        if result.get('human_review', {}).get('required'):
            triggers = result.get('human_review', {}).get('triggers', [])
            formatted_text = (
                f"**[WARNING] Human Review Recommended:** {', '.join(triggers)}\n\n"
                + formatted_text
            )
        
        print(f"   [OK] Multimodal analysis complete! Found {len(zones)} priority zones.")
        
        return {
            'text': formatted_text,
            'analysis': analysis,
            'signals': signals,
            'human_review_required': result.get('human_review', {}).get('required', False)
        }
    
    else:
        # Analysis failed - return error info
        error_text = f"[WARNING] Multimodal analysis failed: {result.get('error', result.get('status', 'unknown'))}"
        print(f"   {error_text}")
        return {'text': error_text, 'analysis': None}


def _get_text_only_analysis(stats: Dict[str, Any], bbox: Dict[str, float], user_type: str) -> Dict[str, Any]:
    """
    LEGACY text-only analysis (fallback when images aren't available).
    """
    from reasoning import EcoReviveGemini
    
    print("   [WARNING] Running LEGACY text-only analysis (no images sent to Gemini)")
    
    client = EcoReviveGemini()
    
    # Format prompt based on user type
    if user_type == "professional":
        prompt = f"""You are an expert wildfire restoration ecologist. Analyze this burn severity assessment:

**Location**: {abs(bbox['south']):.4f}°{'N' if bbox['south'] >= 0 else 'S'} to {abs(bbox['north']):.4f}°{'N' if bbox['north'] >= 0 else 'S'}, {abs(bbox['west']):.4f}°{'E' if bbox['west'] >= 0 else 'W'} to {abs(bbox['east']):.4f}°{'E' if bbox['east'] >= 0 else 'W'}

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

**Location**: Near {abs(bbox['south']):.2f}°{'N' if bbox['south'] >= 0 else 'S'}, {abs(bbox['west']):.2f}°{'E' if bbox['west'] >= 0 else 'W'}

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
    return {'text': response.get('text', 'Analysis not available.'), 'analysis': None}




@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ee_initialized, rag_system
    
    print("[INFO] Starting EcoRevive API Server...")
    
    # Initialize Earth Engine
    ee_initialized = initialize_ee()
    
    # Load Fire Model
    load_fire_model()
    
    # Initialize RAG system for knowledge-grounded responses
    try:
        from reasoning.rag.ecology_rag import CombinedRAG
        print("[INFO] Initializing RAG knowledge retrieval system...")
        rag_system = CombinedRAG()
        rag_system.initialize()
        print("[OK] RAG system ready!")
    except Exception as e:
        print(f"[WARNING] RAG initialization failed: {e}")
        print("   Chat will work but without knowledge grounding")
        rag_system = None
    
    print("[OK] Server ready!")


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
        
        print(f"[INFO] Analyzing area: {bbox}")
        print(f"[INFO] Date range: {start_date} to {end_date}")
        
        # Step 1: Download Sentinel-2 imagery with TILING support
        if ee_initialized:
            try:
                print("[INFO] Downloading Sentinel-2 imagery (with tiling)...")
                tiles, ee_metadata = download_sentinel2_for_model(bbox, start_date, end_date)
                print(f"   Downloaded {len(tiles)} tiles")
                
                if not tiles:
                    raise Exception("No tiles downloaded")
                    
            except Exception as e:
                print(f"[WARNING] EE download failed: {e}, using synthetic data")
                # Fallback to single synthetic tile
                synthetic_image = create_synthetic_image(bbox)
                tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
                ee_metadata = {'n_rows': 1, 'n_cols': 1}
        else:
            print("[WARNING] Earth Engine not available, using synthetic imagery for demo")
            synthetic_image = create_synthetic_image(bbox)
            tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
            ee_metadata = {'n_rows': 1, 'n_cols': 1}
        
        # Step 2: Run Fire Model inference on all tiles
        if model is None:
            raise HTTPException(status_code=503, detail="Fire model not loaded")
        
        print("[INFO] Running burn severity inference on tiles...")
        severity_map, satellite_rgb, stats = run_tiled_inference(tiles, ee_metadata)
        print(f"   Mean severity: {stats['mean_severity']:.1%}")
        print(f"   Output size: {severity_map.shape}")
        
        # Step 3: Create visualizations from stitched results
        satellite_image = rgb_array_to_base64(satellite_rgb)
        severity_image = severity_to_image(severity_map)
        raw_severity_image = severity_to_raw_image(severity_map)
        
        # Step 4: Generate Layer 2 structured output (JSON-only, single Gemini call)
        print("[INFO] Generating Layer 2 structured output...")
        layer2_output = None
        try:
            from reasoning.layer2_output import create_layer2_response
            from reasoning.gemini_multimodal import MultimodalAnalyzer
            from reasoning import create_client
            
            # Prepare RGB tile
            if satellite_rgb.ndim == 3 and satellite_rgb.shape[2] == 3:
                rgb_tile = np.moveaxis(satellite_rgb, -1, 0)
            else:
                rgb_tile = satellite_rgb
            
            center_lat = (bbox['north'] + bbox['south']) / 2
            center_lon = (bbox['west'] + bbox['east']) / 2
            location = (center_lat, center_lon)
            
            # Get structured Gemini analysis for Layer 2 (JSON-only, no verbose text)
            client = create_client()
            analyzer = MultimodalAnalyzer(client)
            l2_result = analyzer.analyze_for_layer2(
                rgb_tile=rgb_tile,
                severity_map=severity_map,
                location=location,
                unet_confidence=0.85
            )
            
            if l2_result.get('status') == 'complete':
                layer2_output = create_layer2_response(
                    severity_map=severity_map,
                    location=location,
                    bbox=bbox,
                    gemini_analysis=l2_result.get('layer2_data'),
                    model_confidence=0.85,
                    imagery_date=end_date
                )
                print("   [OK] Layer 2 JSON generated")
        except Exception as e:
            print(f"   [WARNING] Layer 2 generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Step 5: Run Layer 3 contextual analysis (urban detection & cautions)
        print("[INFO] Running Layer 3 contextual analysis...")
        layer3_context = None
        try:
            from reasoning.layer3_context import create_layer3_response

            layer3_context = create_layer3_response(
                client=client,
                rgb_image=satellite_rgb,
                severity_map=severity_map,
                location=location,
                use_gemini=True
            )

            # Log the land use classification
            land_use = layer3_context.get('land_use', {})
            caution_level = layer3_context.get('overall_caution_level', 'none')
            print(f"   Land use: {land_use.get('land_use_type', 'unknown')} "
                  f"(urban: {land_use.get('urban_percentage', 0):.0f}%)")
            print(f"   Caution level: {caution_level}")
            if caution_level in ['moderate', 'high']:
                print(f"   [WARNING] {land_use.get('caution_message', '')[:100]}...")
            print("   [OK] Layer 3 context generated")
        except Exception as e:
            print(f"   [WARNING] Layer 3 analysis failed: {e}")
            import traceback
            traceback.print_exc()

        # Step 6: Calculate carbon sequestration potential
        print("[INFO] Calculating carbon sequestration potential...")
        carbon_analysis = None
        try:
            from reasoning.carbon_calculator import create_carbon_response

            # Calculate area in hectares from bbox
            lat_mid = (bbox['north'] + bbox['south']) / 2
            km_per_deg_lon = 111.32 * np.cos(lat_mid * np.pi / 180)
            km_per_deg_lat = 110.574
            width_km = abs(bbox['east'] - bbox['west']) * km_per_deg_lon
            height_km = abs(bbox['north'] - bbox['south']) * km_per_deg_lat
            area_km2 = width_km * height_km
            area_hectares = area_km2 * 100  # 1 km² = 100 hectares

            # Get land use type from Layer 3 if available
            land_use_type = None
            if layer3_context:
                land_use_type = layer3_context.get('land_use', {}).get('land_use_type')

            carbon_analysis = create_carbon_response(
                area_hectares=area_hectares,
                severity_stats=stats,
                location=location,
                user_type=request.user_type,
                land_use_type=land_use_type
            )

            # Log the results
            summary = carbon_analysis.get('summary', {})
            print(f"   Carbon potential: {summary.get('headline', 'N/A')}")
            print("   [OK] Carbon analysis complete")
        except Exception as e:
            print(f"   [WARNING] Carbon calculation failed: {e}")
            import traceback
            traceback.print_exc()

        print("[OK] Analysis complete!")
        
        return AnalyzeResponse(
            success=True,
            satellite_image=satellite_image,
            severity_image=severity_image,
            raw_severity_image=raw_severity_image,
            severity_stats=stats,
            gemini_analysis=None,  # No longer sending verbose text
            layer2_output=layer2_output,
            layer3_context=layer3_context,
            carbon_analysis=carbon_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/layer2-analyze", response_model=Layer2AnalyzeResponse)
async def layer2_analyze_area(request: AnalyzeRequest):
    """
    Layer 2 structured analysis endpoint.
    
    Returns structured JSON output for Layer 3 consumption.
    This endpoint is optimized for programmatic access and includes:
    - Location context (lat/lon, state, country)
    - Site characteristics (soil, terrain, land-use history)
    - Ecosystem classification
    - Computed metrics (burn %, NDVI, healing time)
    - Spatial primitives (zones, hazards, risk grid)
    - Machine-readable signals for downstream workflows
    """
    try:
        from reasoning.layer2_output import create_layer2_response
        
        # Validate bounding box
        bbox = {
            'west': request.west,
            'south': request.south,
            'east': request.east,
            'north': request.north
        }
        
        # Set date range
        start_date = request.start_date or "2023-06-01"
        end_date = request.end_date or "2023-09-30"
        
        print(f"[INFO] Layer 2 Analysis: {bbox}")
        
        # Step 1: Download Sentinel-2 imagery
        if ee_initialized:
            try:
                print("[INFO] Downloading Sentinel-2 imagery...")
                tiles, ee_metadata = download_sentinel2_for_model(bbox, start_date, end_date)
                if not tiles:
                    raise Exception("No tiles downloaded")
            except Exception as e:
                print(f"[WARNING] EE download failed: {e}, using synthetic data")
                synthetic_image = create_synthetic_image(bbox)
                tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
                ee_metadata = {'n_rows': 1, 'n_cols': 1}
        else:
            print("[WARNING] Earth Engine not available, using synthetic imagery")
            synthetic_image = create_synthetic_image(bbox)
            tiles = [{'image': synthetic_image, 'row': 0, 'col': 0, 'center': (0, 0)}]
            ee_metadata = {'n_rows': 1, 'n_cols': 1}
        
        # Step 2: Run Fire Model inference
        if model is None:
            raise HTTPException(status_code=503, detail="Fire model not loaded")
        
        print("[INFO] Running burn severity inference...")
        severity_map, satellite_rgb, stats = run_tiled_inference(tiles, ee_metadata)
        
        # Step 3: Calculate center location
        center_lat = (bbox['north'] + bbox['south']) / 2
        center_lon = (bbox['west'] + bbox['east']) / 2
        location = (center_lat, center_lon)
        
        # Step 4: Run Gemini multimodal analysis for enhancement
        gemini_analysis = None
        try:
            # Prepare RGB tile for multimodal
            if satellite_rgb.ndim == 3 and satellite_rgb.shape[2] == 3:
                rgb_tile = np.moveaxis(satellite_rgb, -1, 0)
            else:
                rgb_tile = satellite_rgb
            
            from reasoning.gemini_multimodal import MultimodalAnalyzer
            from reasoning import create_client
            
            client = create_client()
            analyzer = MultimodalAnalyzer(client)
            
            # Use analyze_for_layer2 for structured JSON output
            result = analyzer.analyze_for_layer2(
                rgb_tile=rgb_tile,
                severity_map=severity_map,
                location=location,
                unet_confidence=0.85
            )
            
            if result.get('status') == 'complete':
                # Use layer2_data which is already transformed to Layer2Output schema
                gemini_analysis = result.get('layer2_data')
                
        except Exception as e:
            print(f"[WARNING] Gemini analysis failed: {e}, proceeding without enhancement")
        
        # Step 5: Generate Layer 2 structured output
        print("[INFO] Generating Layer 2 structured output...")
        layer2_output = create_layer2_response(
            severity_map=severity_map,
            location=location,
            bbox=bbox,
            gemini_analysis=gemini_analysis,
            model_confidence=0.85,
            imagery_date=end_date
        )
        
        # Step 6: Create images for reference
        satellite_image = rgb_array_to_base64(satellite_rgb)
        severity_image = severity_to_image(severity_map)
        
        print(f"[OK] Layer 2 analysis complete! Found {len(layer2_output.get('zones', []))} zones, "
              f"{len(layer2_output.get('hazards', []))} hazards")
        
        return Layer2AnalyzeResponse(
            success=True,
            layer2_output=layer2_output,
            satellite_image=satellite_image,
            severity_image=severity_image
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Layer 2 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return Layer2AnalyzeResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    RAG-Augmented AI Chat endpoint.

    Uses Gemini with Retrieval-Augmented Generation to produce
    knowledge-grounded, expert-level responses backed by:
    - Ecological knowledge base (species, ecoregions, fire ecology)
    - Legal knowledge base (permits, land ownership, regulations)
    
    The response is suitable for direct display in chat AND export to Word/PDF.
    """
    try:
        from reasoning import create_client

        print(f"[INFO] Chat request: action={request.action_type}, user={request.user_type}")

        # Get the Gemini client
        client = create_client()

        # Build context from the analysis results
        context = request.context or {}
        severity_stats = context.get('severity_stats', {})
        bbox = context.get('bbox', {})
        layer2 = context.get('layer2_output', {})

        # Calculate area if bbox provided
        area_km2 = 0
        center_lat, center_lon = 0, 0
        if bbox:
            lat_mid = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
            km_per_deg_lon = 111.32 * np.cos(lat_mid * np.pi / 180)
            km_per_deg_lat = 110.574
            width = abs(bbox.get('east', 0) - bbox.get('west', 0)) * km_per_deg_lon
            height = abs(bbox.get('north', 0) - bbox.get('south', 0)) * km_per_deg_lat
            area_km2 = width * height
            center_lat = lat_mid
            center_lon = (bbox.get('east', 0) + bbox.get('west', 0)) / 2

        # Build severity context (prioritize robust "in_burn_area" metrics)
        if 'mean_severity_in_burn_area' in severity_stats:
            # New robust metrics available
            mean_sev = severity_stats.get('mean_severity_in_burn_area', 0) * 100
            high_sev = severity_stats.get('high_severity_in_burn_area', 0) * 100
        else:
            # Fallback to legacy
            mean_sev = severity_stats.get('mean_severity', 0) * 100
            high_sev = severity_stats.get('high_severity_ratio', 0) * 100
        
        # Derive severity level for RAG queries
        if mean_sev > 60:
            severity_level = "high"
        elif mean_sev > 30:
            severity_level = "moderate"
        else:
            severity_level = "low"

        # Build location description for RAG queries
        # Format coordinates correctly: N/S for lat, E/W for lon
        lat_dir = "N" if center_lat >= 0 else "S"
        lon_dir = "E" if center_lon >= 0 else "W"
        location_description = f"Site at {abs(center_lat):.4f}°{lat_dir}, {abs(center_lon):.4f}°{lon_dir}, {area_km2:.1f} km²"

        # =====================================================
        # GEOGRAPHIC AWARENESS: Detect region and knowledge level
        # =====================================================
        geographic_context = ""
        biome_fire_ecology = ""
        try:
            from reasoning.rag.geographic_awareness import (
                get_region_info, 
                format_geographic_context,
                get_biome_fire_ecology_summary
            )
            
            region_info = get_region_info(center_lat, center_lon)
            geographic_context = format_geographic_context(region_info)
            biome_fire_ecology = get_biome_fire_ecology_summary(region_info.biome_type)
            
            print(f"   [GEO] Region: {region_info.region_name}, Knowledge Level: {region_info.knowledge_level}")
            print(f"   [GEO] Biome: {region_info.biome_type}, Fire Regime: {region_info.fire_regime}")
            if region_info.disclosure_required:
                print(f"   [GEO] ⚠️ Disclosure required: {region_info.knowledge_level} knowledge")
            
        except Exception as e:
            print(f"   [GEO] Geographic awareness failed: {e}, proceeding with California defaults")
            geographic_context = ""
            biome_fire_ecology = ""

        # =====================================================
        # REASONING GUARDRAILS: Apply cognitive constraints
        # =====================================================
        reasoning_constraints = ""
        try:
            from reasoning.rag.reasoning_framework import apply_reasoning_guardrails
            
            # Get land use type from Layer 3 if available
            layer3 = context.get('layer3_output', {})
            land_use = layer3.get('land_use', {})
            area_type = land_use.get('type', 'mixed_use')
            
            # Build Layer 2 JSON for evidence anchoring
            layer2_evidence = {
                "burn_severity_mean": mean_sev / 100.0, # Use robust metric
                "burn_severity_high_ratio": high_sev / 100.0, # Use robust metric
                "vegetation_cover_percent": layer2.get('vegetation_cover', 50),
                "urban_percentage": land_use.get('urban_percent', 0),
                "structures_detected": layer3.get('structures_detected', False),
                "water_body_present": layer3.get('water_bodies', False),
                "overall_confidence": layer2.get('confidence', 0.7),
            }
            
            # Apply reasoning guardrails
            reasoning_constraints = apply_reasoning_guardrails(
                area_type=area_type,
                layer2_json=layer2_evidence
            )
            print(f"   [GUARDRAILS] Applied area-type reasoning constraints: {area_type}")
            
        except Exception as e:
            print(f"   [GUARDRAILS] Constraint loading failed: {e}, using standard prompts")
            reasoning_constraints = ""

        # =====================================================
        # RAG RETRIEVAL: Get relevant knowledge base context
        # =====================================================
        rag_context = ""
        if rag_system is not None and request.action_type:
            try:
                print(f"   [RAG] Retrieving knowledge for action: {request.action_type}")
                
                # Map action types to appropriate RAG queries
                if request.action_type in ['species', 'biophysical', 'hope']:
                    # Ecology-focused actions
                    rag_context = rag_system.ecology_rag.get_restoration_context(
                        location_description=location_description,
                        severity_level=severity_level,
                        k=5
                    )
                    print(f"   [RAG] Retrieved ecological context")
                    
                elif request.action_type in ['legal', 'ownership']:
                    # Legal-focused actions
                    rag_context = rag_system.legal_rag.get_legal_context(
                        location_description=location_description,
                        activity_type="restoration"
                    )
                    print(f"   [RAG] Retrieved legal context")
                    
                elif request.action_type == 'monitoring':
                    # Combined context for monitoring
                    rag_context = rag_system.get_full_context(
                        location_description=location_description,
                        severity_level=severity_level,
                        activity_type="monitoring"
                    )
                    print(f"   [RAG] Retrieved combined context")
                    
                else:
                    # General queries get combined context
                    rag_context = rag_system.get_full_context(
                        location_description=location_description,
                        severity_level=severity_level,
                        activity_type="restoration"
                    )
                    print(f"   [RAG] Retrieved general context")
                    
            except Exception as e:
                print(f"   [RAG] Retrieval failed: {e}, proceeding without RAG")
                rag_context = ""

        # =====================================================
        # BUILD RAG-AUGMENTED PROMPT
        # =====================================================
        
        # System context with RAG grounding instruction and reasoning guardrails
        system_context = f"""You are a Senior Restoration Ecologist and RAG-Augmented Reasoning Engine for EcoRevive.

{geographic_context}

{biome_fire_ecology}

{reasoning_constraints}

SITE ANALYSIS DATA (Derived from Satellite):
- Location: {abs(center_lat):.4f}°{lat_dir}, {abs(center_lon):.4f}°{lon_dir}
- Area: {area_km2:.1f} km² ({area_km2 * 100:.0f} hectares)
- Mean burn severity (within burn scar): {mean_sev:.0f}%
- High severity coverage (within burn scar): {high_sev:.0f}%
- User type: {request.user_type}

VISUAL ANALYSIS EVIDENCE:
- Detected Area Type: {layer2.get('location_enhancement', {}).get('area_type', 'unknown')}
- Terrain Features: {', '.join(layer2.get('location_enhancement', {}).get('terrain_features', []))}
- Soil Inference: {layer2.get('characteristics', {}).get('soil_type', 'unknown')}
- Slope/Drainage: {layer2.get('characteristics', {}).get('slope_category', 'unknown')}, {layer2.get('characteristics', {}).get('drainage_pattern', 'unknown')}
- Land Use History: {layer2.get('characteristics', {}).get('land_use_history', 'unknown')}
- Ecosystem State: {layer2.get('ecosystem', {}).get('current_state', 'unknown')}
- Vegetation Types: {', '.join(layer2.get('ecosystem', {}).get('vegetation_types', []))}
- Key Species Observed/Likely: {', '.join(layer2.get('ecosystem', {}).get('key_species', []))}

ZONATION & HAZARDS:
- Zones: {', '.join([f"{z.get('zone_type')} ({z.get('severity_category', '')}, {z.get('area_estimate_pct', 0)}%)" for z in layer2.get('visual_zones', [])])}
- Hazards: {', '.join([f"{h.get('hazard_type')} ({h.get('severity', '')})" for h in layer2.get('visual_hazards', [])])}

ASSESSMENT SIGNALS:
- Restoration Potential: {layer2.get('signals', {}).get('restoration_potential_score', 0.0)}/1.0
- Intervention Urgency: {layer2.get('signals', {}).get('intervention_urgency_score', 0.0)}/1.0
- Ecological Complexity: {layer2.get('signals', {}).get('ecological_complexity_score', 0.0)}/1.0

{rag_context}

EVIDENCE-GROUNDED INSTRUCTIONS:
1. Speak naturally as an expert. Do NOT cite the JSON structure (e.g. avoid saying "(JSON: ...)" or "Visual Analysis: ...").
2. Use the provided data as facts. For example, say "The site has rocky soil" instead of "The JSON indicates rocky soil".
3. ONLY discuss risks that are BOTH: (a) eligible for this area type AND (b) supported by evidence.
4. Scale severity language to evidence strength (minor claims for low values, significant for high).
5. For missing data: remain silent or state "insufficient data" - do NOT speculate.
6. Do NOT list generic hazards or encyclopedic risks without evidence.
7. For professional users: include scientific names, quantitative data, uncertainty bounds.
8. For personal users: be approachable but still evidence-grounded.
9. **CRITICAL: NEVER state 'Data confidence is 0%' or similar.** If data is limited, simply advise verifying on the ground. Do not mention internal confidence scores.
"""

        # Build action-specific prompts with explicit grounding instructions
        if request.action_type:
            action_prompts = {
                # PERSONAL - grounded prompts
                'safety': f"""Generate a comprehensive safety checklist for volunteers at this {high_sev:.0f}% high-severity burn site.

Structure your response:
## Immediate Hazards
(List hazards specific to burn severity level)

## Required Safety Gear
(Table format: Item | Purpose | Priority)

## Zones to Avoid
(Based on severity map and retrieved ecological data)

## Emergency Contacts
(Relevant ranger districts from retrieved data if available)
""",

                'hope': f"""Generate an evidence-based recovery timeline for this burn site.

Use the retrieved ecological data to provide realistic estimates. Structure:

## Current State (Year 0)
## Year 5 Projection
## Year 10 Projection  
## Year 15 Projection

For each period include:
- Expected species succession (use scientific names from retrieved data)
- Estimated vegetation cover %
- Carbon sequestration potential
- Key milestones

Be encouraging but scientifically grounded.
""",

                'ownership': f"""Provide a land ownership and jurisdiction analysis for this California restoration site.

Use retrieved legal knowledge to identify:

## Land Ownership Determination
- Likely managing agency based on location
- How to verify ownership (specific steps)

## Permit Requirements
(From retrieved permit data)

## Key Contacts
(Ranger districts, agency offices from retrieved data)

## Timeline for Approval
""",

                'supplies': f"""Generate a detailed supply list and budget for a 10-person restoration event at this {area_km2:.1f} km² site.

## Equipment List
| Item | Quantity | Unit Cost | Total |
|------|----------|-----------|-------|

## Safety Equipment

## Planting Materials
(Based on retrieved species recommendations for {severity_level} severity)

## Logistics

## Total Budget Estimate
""",

                # PROFESSIONAL - grounded prompts
                'legal': f"""Provide a professional-grade legal and tenure analysis for a restoration grant application.

Use retrieved legal framework data to address:

## Ownership Verification Protocol
## Protected Status Assessment
(Endangered species, cultural resources, watershed protections)

## Regulatory Compliance
(CEQA, NEPA, Clean Water Act as applicable)

## Encumbrances and Easements
## Timeline and Cost Estimates
## Required Certifications

Include uncertainty where verification is needed.
""",

                'biophysical': f"""Generate a professional biophysical site characterization.

Use retrieved ecoregion and ecological data:

## Ecoregion Classification
(From retrieved California ecoregion data)

## Soil Characteristics
## Hydrology Assessment
## Topographic Analysis
## Historical Land Use
## Fire Regime Analysis
(From retrieved fire ecology data)

## Recommended Species Palette
(Scientific names, source: retrieved species catalog)

## Microsite Matching Matrix
""",

                'species': f"""Generate a comprehensive native species palette for this California burn site.

Use retrieved species catalog and ecoregion data. Structure:

## Pioneer Species (Year 0-2)
| Scientific Name | Common Name | Planting Density | Survival Rate | Source |
|-----------------|-------------|------------------|---------------|--------|

## Mid-Succession Species (Year 2-5)
(Table format)

## Climax Community Species (Year 5-10)
(Table format)

## Invasive Species to Monitor
(From retrieved invasive warnings)

## Planting Prescriptions by Microsite
""",

                'monitoring': f"""Generate a professional monitoring framework for restoration verification.

## Baseline Metrics (Year 0)
| Metric | Measurement Protocol | Target Value | Uncertainty |
|--------|---------------------|--------------|-------------|

## Monitoring Schedule
| Year | Metrics | Method | Estimated Cost |
|------|---------|--------|----------------|

## Carbon Protocol Eligibility
(CAR, ACR, VCS based on retrieved data)

## Uncertainty Quantification
## Adaptive Management Triggers
## Reporting Requirements
"""
            }

            prompt = action_prompts.get(request.action_type, request.message)
        else:
            prompt = f"""Answer this question using the site analysis data and retrieved knowledge:

{request.message}

Provide an expert-level response with citations where applicable.
"""

        full_prompt = f"{system_context}\n\n{prompt}"

        # Generate response
        response = client.analyze_multimodal(
            prompt=full_prompt,
            use_json=False
        )

        rag_status = "with RAG" if rag_context else "no RAG"
        print(f"   [OK] Generated response ({response['usage']['response_tokens']} tokens, {rag_status})")

        return ChatResponse(
            success=True,
            response=response['text']
        )

    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/export/pdf", response_model=PDFExportResponse)
async def export_pdf(request: PDFExportRequest):
    """
    Generate a PDF report for burn severity analysis.

    Supports two report types:
    - "personal": 1-2 page Impact Card (shareable, emotional)
    - "professional": 5-10 page grant-ready document (technical, comprehensive)
    """
    try:
        from reasoning.pdf_export import generate_pdf
        import base64

        print(f"[INFO] PDF Export request: type={request.report_type}, user={request.user_type}")

        # Determine report type (use user_type if report_type not explicitly set)
        report_type = request.report_type
        if report_type not in ["personal", "professional"]:
            report_type = request.user_type

        # Generate PDF
        pdf_bytes, metadata = generate_pdf(
            report_type=report_type,
            satellite_image=request.satellite_image,
            severity_image=request.severity_image,
            severity_stats=request.severity_stats,
            bbox=request.bbox,
            layer2_output=request.layer2_output,
            layer3_context=request.layer3_context,
            carbon_analysis=request.carbon_analysis,
            location_name=request.location_name,
            analysis_id=request.analysis_id,
        )

        # Encode to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        print(f"   [OK] PDF generated: {metadata['filename']} ({metadata['file_size_bytes']} bytes)")

        return PDFExportResponse(
            success=True,
            pdf_base64=pdf_base64,
            filename=metadata['filename']
        )

    except Exception as e:
        print(f"[ERROR] PDF export failed: {e}")
        import traceback
        traceback.print_exc()
        return PDFExportResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/export/word", response_model=WordExportResponse)
async def export_word(request: WordExportRequest):
    """
    Generate a Word document report for burn severity analysis.
    Matches PDF content so users can edit and customize.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import base64
        import io
        from PIL import Image
        from datetime import datetime

        print(f"[INFO] Word Export request: type={request.report_type}")

        # Create document
        doc = Document()

        # Set document margins
        for section in doc.sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)

        # Calculate values
        location_name = request.location_name or "Analysis Site"
        timestamp = datetime.now().strftime("%B %d, %Y")
        area_km2 = _calculate_area_km2(request.bbox)
        area_hectares = area_km2 * 100

        # Safely extract severity values
        high_sev = float(request.severity_stats.get('high_severity_ratio', 0) or 0) * 100
        mod_sev = float(request.severity_stats.get('moderate_severity_ratio', 0) or 0) * 100
        low_sev = float(request.severity_stats.get('low_severity_ratio', 0) or 0) * 100
        mean_sev = float(request.severity_stats.get('mean_severity', 0) or 0) * 100

        # ==================== TITLE ====================
        title = doc.add_heading('EcoRevive Burn Severity Analysis', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Location header
        header_para = doc.add_paragraph()
        header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_para.add_run(f"{location_name}\n").bold = True

        # Coordinates
        center_lat = (request.bbox['north'] + request.bbox['south']) / 2
        center_lon = (request.bbox['west'] + request.bbox['east']) / 2
        lat_dir = 'N' if center_lat >= 0 else 'S'
        lon_dir = 'E' if center_lon >= 0 else 'W'
        coords = f"{abs(center_lat):.4f}°{lat_dir}, {abs(center_lon):.4f}°{lon_dir}"
        header_para.add_run(f"{coords}  |  {area_km2:.2f} km² ({area_hectares:.0f} ha)  |  {timestamp}")

        doc.add_paragraph()

        # ==================== IMAGES ====================
        doc.add_heading('Satellite Analysis', level=1)

        # Add images side by side if available
        if request.satellite_image and request.severity_image:
            # Create a table for side-by-side images
            img_table = doc.add_table(rows=2, cols=2)

            # Add satellite image
            try:
                sat_data = request.satellite_image.split(',')[1] if ',' in request.satellite_image else request.satellite_image
                sat_bytes = base64.b64decode(sat_data)
                sat_stream = io.BytesIO(sat_bytes)
                cell = img_table.rows[0].cells[0]
                cell_para = cell.paragraphs[0]
                run = cell_para.add_run()
                run.add_picture(sat_stream, width=Inches(2.8))
                cell_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            except:
                img_table.rows[0].cells[0].text = "[Satellite Image]"

            # Add severity image
            try:
                sev_data = request.severity_image.split(',')[1] if ',' in request.severity_image else request.severity_image
                sev_bytes = base64.b64decode(sev_data)
                sev_stream = io.BytesIO(sev_bytes)
                cell = img_table.rows[0].cells[1]
                cell_para = cell.paragraphs[0]
                run = cell_para.add_run()
                run.add_picture(sev_stream, width=Inches(2.8))
                cell_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            except:
                img_table.rows[0].cells[1].text = "[Severity Map]"

            # Captions
            img_table.rows[1].cells[0].text = "Sentinel-2 Satellite View"
            img_table.rows[1].cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            img_table.rows[1].cells[1].text = "AI Burn Severity Map"
            img_table.rows[1].cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_paragraph()

        # ==================== EXECUTIVE SUMMARY ====================
        doc.add_heading('Executive Summary', level=1)

        summary = doc.add_paragraph()
        summary.add_run(f"This {area_km2:.1f} km² site shows ")
        if mean_sev > 60:
            run = summary.add_run("SEVERE IMPACT")
            run.bold = True
        elif mean_sev > 35:
            run = summary.add_run("MODERATE IMPACT")
            run.bold = True
        else:
            run = summary.add_run("LOW-MODERATE IMPACT")
            run.bold = True
        summary.add_run(f" with {mean_sev:.0f}% average burn severity and {high_sev:.0f}% high-severity areas.")

        doc.add_paragraph()

        # ==================== SEVERITY TABLE ====================
        doc.add_heading('Burn Severity Breakdown', level=1)

        table = doc.add_table(rows=4, cols=4)
        table.style = 'Table Grid'

        # Header row
        headers = ['Category', 'Percentage', 'Area (km²)', 'What This Means']
        for i, header in enumerate(headers):
            cell = table.rows[0].cells[i]
            cell.text = header
            cell.paragraphs[0].runs[0].bold = True

        # Data rows
        severity_data = [
            ['High Severity', f'{high_sev:.1f}%', f'{high_sev * area_km2 / 100:.2f}', 'Complete vegetation loss, may need replanting'],
            ['Moderate', f'{mod_sev:.1f}%', f'{mod_sev * area_km2 / 100:.2f}', 'Partial damage, natural recovery likely'],
            ['Low/Unburned', f'{low_sev:.1f}%', f'{low_sev * area_km2 / 100:.2f}', 'Minimal impact, seed source preserved'],
        ]

        for row_idx, row_data in enumerate(severity_data):
            for col_idx, cell_data in enumerate(row_data):
                table.rows[row_idx + 1].cells[col_idx].text = cell_data

        doc.add_paragraph()

        # ==================== CARBON IMPACT ====================
        if request.carbon_analysis:
            doc.add_heading('Restoration Impact Potential', level=1)

            if request.user_type == 'personal' and request.carbon_analysis.get('personal'):
                personal = request.carbon_analysis['personal']
                total_co2 = int(personal.get('total_co2_capture_20yr', 0))

                carbon_para = doc.add_paragraph()
                run = carbon_para.add_run(f"{total_co2:,} tons CO₂")
                run.bold = True
                run.font.size = Pt(18)
                carbon_para.add_run(" captured over 20 years of restoration")
                carbon_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

                equivs = personal.get('equivalencies', {})
                if equivs:
                    doc.add_paragraph()
                    doc.add_paragraph("That's equivalent to:")
                    doc.add_paragraph(f"    • {int(equivs.get('cars_off_road_for_year', 0)):,} cars removed from road for 1 year")
                    doc.add_paragraph(f"    • {int(equivs.get('tree_seedlings_grown_10yr', 0)):,} tree seedlings grown for 10 years")
                    doc.add_paragraph(f"    • {int(equivs.get('round_trip_flights_nyc_la', 0)):,} round-trip flights NYC to LA offset")
                    doc.add_paragraph(f"    • {int(equivs.get('homes_electricity_year', 0)):,} homes powered for 1 year")

            elif request.carbon_analysis.get('professional'):
                pro = request.carbon_analysis['professional']

                # Methodology badge
                method_para = doc.add_paragraph()
                run = method_para.add_run(f"Methodology: {pro.get('methodology', 'IPCC Tier 2')}")
                run.bold = True
                method_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

                doc.add_paragraph()

                # Carbon Stock Summary Table
                doc.add_heading('Carbon Stock Summary', level=2)
                carbon_table = doc.add_table(rows=5, cols=3)
                carbon_table.style = 'Table Grid'

                # Headers
                carbon_headers = ['Metric', 'Value', 'Unit']
                for i, header in enumerate(carbon_headers):
                    cell = carbon_table.rows[0].cells[i]
                    cell.text = header
                    cell.paragraphs[0].runs[0].bold = True

                # Data
                carbon_data = [
                    ['Baseline Carbon Stock', f"{pro.get('baseline_carbon_tc', 0):,.1f}", 'tC'],
                    ['Carbon Lost to Fire', f"-{pro.get('carbon_lost_tc', 0):,.1f}", 'tC'],
                    ['Current Carbon Stock', f"{pro.get('current_carbon_tc', 0):,.1f}", 'tC'],
                    ['Annual Sequestration Rate', f"{pro.get('annual_sequestration_tco2e', 0):,.1f}", 'tCO₂e/year'],
                ]

                for row_idx, row_data in enumerate(carbon_data):
                    for col_idx, cell_data in enumerate(row_data):
                        carbon_table.rows[row_idx + 1].cells[col_idx].text = cell_data

                doc.add_paragraph()

                # Sequestration Projections
                doc.add_heading('Sequestration Projections', level=2)

                projections = pro.get('projections', [])
                if projections:
                    proj_table = doc.add_table(rows=len(projections) + 1, cols=3)
                    proj_table.style = 'Table Grid'

                    # Headers
                    proj_headers = ['Timeframe', 'Cumulative tCO₂e', 'Annual Rate']
                    for i, header in enumerate(proj_headers):
                        cell = proj_table.rows[0].cells[i]
                        cell.text = header
                        cell.paragraphs[0].runs[0].bold = True

                    # Data
                    for row_idx, proj in enumerate(projections):
                        proj_table.rows[row_idx + 1].cells[0].text = f"{proj.get('years', 0)} years"
                        proj_table.rows[row_idx + 1].cells[1].text = f"{proj.get('cumulative_tco2e', 0):,.0f}"
                        proj_table.rows[row_idx + 1].cells[2].text = f"{proj.get('annual_rate_tco2e', 0):,.1f} tCO₂e/yr"
                else:
                    # Generate default projections
                    annual_rate = pro.get('annual_sequestration_tco2e', 0)
                    proj_table = doc.add_table(rows=5, cols=3)
                    proj_table.style = 'Table Grid'

                    proj_headers = ['Timeframe', 'Cumulative tCO₂e', 'Annual Rate']
                    for i, header in enumerate(proj_headers):
                        cell = proj_table.rows[0].cells[i]
                        cell.text = header
                        cell.paragraphs[0].runs[0].bold = True

                    for row_idx, years in enumerate([5, 10, 15, 20]):
                        proj_table.rows[row_idx + 1].cells[0].text = f"{years} years"
                        proj_table.rows[row_idx + 1].cells[1].text = f"{(annual_rate * years):,.0f}"
                        proj_table.rows[row_idx + 1].cells[2].text = f"{annual_rate:,.1f} tCO₂e/yr"

                doc.add_paragraph()

                # Protocol Eligibility
                doc.add_heading('Carbon Credit Protocol Eligibility', level=2)

                protocols = pro.get('protocols', {})
                if protocols:
                    proto_table = doc.add_table(rows=len(protocols) + 1, cols=2)
                    proto_table.style = 'Table Grid'

                    proto_table.rows[0].cells[0].text = 'Protocol'
                    proto_table.rows[0].cells[0].paragraphs[0].runs[0].bold = True
                    proto_table.rows[0].cells[1].text = 'Eligibility'
                    proto_table.rows[0].cells[1].paragraphs[0].runs[0].bold = True

                    for row_idx, (protocol, eligible) in enumerate(protocols.items()):
                        proto_name = protocol.replace('_', ' ').replace('eligible', '').strip().title()
                        proto_table.rows[row_idx + 1].cells[0].text = proto_name
                        proto_table.rows[row_idx + 1].cells[1].text = 'Eligible' if eligible else 'Not Eligible'
                else:
                    doc.add_paragraph("Protocol eligibility assessment pending. Contact EcoRevive for detailed analysis.")

                doc.add_paragraph()

                # Uncertainty Quantification
                doc.add_heading('Uncertainty Analysis', level=2)

                ci_low = pro.get('confidence_interval_low', 0)
                ci_high = pro.get('confidence_interval_high', 0)
                uncertainty = pro.get('uncertainty_pct', 25)

                uncert_para = doc.add_paragraph()
                uncert_para.add_run("95% Confidence Interval (20-year projection): ").bold = True
                uncert_para.add_run(f"{ci_low:,.0f} - {ci_high:,.0f} tCO₂e")
                doc.add_paragraph(f"Combined Uncertainty: ±{uncertainty}%")

                doc.add_paragraph()

                # Limitations
                doc.add_heading('Methodological Limitations', level=2)

                limitations = pro.get('limitations', [
                    'Remote sensing estimates require ground-truth verification',
                    'Carbon stock calculations assume typical forest composition',
                    'Sequestration rates may vary based on management practices',
                    'Does not account for natural disturbance risk',
                    'Baseline estimates derived from regional averages'
                ])

                for lim in limitations:
                    doc.add_paragraph(f"• {lim}")

                doc.add_paragraph()

                # Data Sources
                doc.add_heading('Data Sources', level=2)

                sources = pro.get('data_sources', [
                    'Sentinel-2 MSI multispectral imagery (10m resolution)',
                    'U-Net deep learning model trained on California wildfires',
                    'IPCC Guidelines for National GHG Inventories (2006, 2019 Refinement)',
                    'FIA forest inventory regional parameters',
                    'Gemini multimodal AI for contextual analysis'
                ])

                for src in sources:
                    doc.add_paragraph(f"• {src}")

        doc.add_paragraph()

        # ==================== SITE CONTEXT ====================
        if request.layer3_context and request.layer3_context.get('land_use'):
            doc.add_heading('Site Context', level=1)
            land_use = request.layer3_context.get('land_use', {})
            land_type = land_use.get('land_use_type', 'Unknown').title()
            doc.add_paragraph(f"Land Classification: {land_type}")
            if land_use.get('land_use_description'):
                doc.add_paragraph(land_use.get('land_use_description'))

        # ==================== SAFETY NOTICE ====================
        doc.add_heading('Safety Notice', level=1)

        if high_sev > 40:
            level = "HIGH"
            message = "This site has significant high-severity burn areas. Before visiting, be aware of hazards like standing dead trees (widowmakers), unstable slopes, and ash pits. Always go with a buddy and inform someone of your plans."
        elif high_sev > 20:
            level = "MODERATE"
            message = "Exercise caution when visiting this site. Some areas may have standing dead trees and loose soil. Wear sturdy boots and bring plenty of water."
        else:
            level = "LOW"
            message = "This site appears relatively safe for visits, but always be aware of your surroundings. Standard outdoor safety precautions apply."

        safety_para = doc.add_paragraph()
        run = safety_para.add_run(f"SAFETY LEVEL: {level}")
        run.bold = True
        doc.add_paragraph(message)

        doc.add_paragraph()

        # ==================== HOW YOU CAN HELP ====================
        doc.add_heading('How You Can Help', level=1)

        actions = [
            ("Learn About Native Species", "Research which native plants thrive in your region. Focus on fire-adapted species."),
            ("Join Restoration Events", "Connect with local conservation groups that organize volunteer planting days."),
            ("Support Organizations", "Donate to or volunteer with groups like One Tree Planted or local land trusts."),
            ("Monitor & Document", "If you visit the site, document recovery with photos for citizen science."),
            ("Reduce Fire Risk", "Create defensible space around homes and practice fire-safe behaviors."),
            ("Spread Awareness", "Share information about wildfire recovery with your community."),
        ]

        for i, (title_text, desc) in enumerate(actions, 1):
            action_para = doc.add_paragraph()
            action_para.add_run(f"{i}. {title_text}: ").bold = True
            action_para.add_run(desc)

        doc.add_paragraph()

        # ==================== RECOVERY TIMELINE ====================
        doc.add_heading('Recovery Timeline', level=1)

        timeline_table = doc.add_table(rows=5, cols=2)
        timeline_table.style = 'Table Grid'

        timeline_data = [
            ['Timeframe', 'What to Expect'],
            ['Year 1-2', 'Ground cover begins returning. Pioneer grasses and shrubs establish.'],
            ['Year 3-5', 'Shrub layer develops. Tree seedlings visible. Wildlife returns.'],
            ['Year 5-10', 'Young forest structure emerges. Canopy begins closing.'],
            ['Year 10-20', 'Maturing forest. Carbon sequestration accelerates.'],
        ]

        for row_idx, row_data in enumerate(timeline_data):
            for col_idx, cell_data in enumerate(row_data):
                cell = timeline_table.rows[row_idx].cells[col_idx]
                cell.text = cell_data
                if row_idx == 0:
                    cell.paragraphs[0].runs[0].bold = True

        doc.add_paragraph()

        # ==================== FOOTER ====================
        doc.add_paragraph()
        footer = doc.add_paragraph()
        footer.add_run(f"Generated by EcoRevive on {timestamp}").italic = True
        footer.add_run("\nData sources: Sentinel-2 satellite imagery, U-Net deep learning model, Gemini AI analysis")
        footer.add_run("\nThis report is for informational purposes. Verify with ground-truth data before making decisions.")
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Save to bytes
        docx_buffer = io.BytesIO()
        doc.save(docx_buffer)
        docx_buffer.seek(0)
        docx_bytes = docx_buffer.read()

        # Encode to base64
        docx_base64 = base64.b64encode(docx_bytes).decode('utf-8')

        filename = f"EcoRevive_{location_name.replace(' ', '_').replace(',', '')}_{datetime.now().strftime('%Y%m%d')}.docx"

        print(f"   [OK] Word document generated: {filename}")

        return WordExportResponse(
            success=True,
            docx_base64=docx_base64,
            filename=filename
        )

    except ImportError:
        return WordExportResponse(
            success=False,
            error="python-docx library not installed. Install with: pip install python-docx"
        )
    except Exception as e:
        print(f"[ERROR] Word export failed: {e}")
        import traceback
        traceback.print_exc()
        return WordExportResponse(
            success=False,
            error=str(e)
        )


def _calculate_area_km2(bbox: Dict[str, float]) -> float:
    """Calculate area in km² from bbox."""
    import math
    lat_mid = (bbox['north'] + bbox['south']) / 2
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_mid))
    width = abs(bbox['east'] - bbox['west']) * km_per_deg_lon
    height = abs(bbox['north'] - bbox['south']) * km_per_deg_lat
    return width * height


@app.post("/api/hope-visualization", response_model=HopeVisualizationResponse)
async def generate_hope_visualization(request: HopeVisualizationRequest):
    """
    Generate hope visualization using Imagen 3.

    Creates AI-generated photorealistic images showing ecosystem recovery
    at different time points (5, 10, 15 years).
    """
    try:
        from reasoning.gemini_hope import HopeVisualizer
        from reasoning import create_client
        import io

        print(f"[INFO] Hope visualization request: {request.ecosystem_type}, {request.years_in_future} years")

        # Create Gemini client and visualizer
        client = create_client()
        visualizer = HopeVisualizer(gemini_client=client)

        # Generate recovery forecast first (text-based timeline)
        forecast = visualizer.forecast_recovery(
            ecosystem_type=request.ecosystem_type,
            mean_severity=request.mean_severity,
            restoration_method="combination",
            area_hectares=request.area_hectares
        )

        # Try to generate image with Imagen 3
        image_result = visualizer.generate_hope_image(
            ecosystem_type=request.ecosystem_type,
            years_in_future=request.years_in_future,
        )

        image_base64 = None
        description = None

        # Check if we got an image
        if image_result.get("image"):
            # Convert PIL image to base64
            img = image_result["image"]
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_base64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            print(f"   [OK] Generated Imagen 3 visualization")
        elif image_result.get("fallback_description"):
            # Use text description as fallback
            description = image_result["fallback_description"]
            print(f"   [WARNING] Using text fallback (Imagen unavailable)")

        # Extract hope message from forecast
        hope_message = forecast.get("hope_message", "")
        if not description and hope_message:
            description = hope_message

        return HopeVisualizationResponse(
            success=True,
            image_base64=image_base64,
            forecast=forecast,
            description=description
        )

    except Exception as e:
        print(f"[ERROR] Hope visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return HopeVisualizationResponse(
            success=False,
            error=str(e)
        )


# Run with: uvicorn server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

