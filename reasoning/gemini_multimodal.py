"""
EcoRevive Gemini Multimodal Module
===================================
True multimodal integration between U-Net segmentation and Gemini 3 Flash.

This module provides:
1. Multi-image pack generation (RGB + Severity Overlay)
2. Semantic severity colormap (restoration-aligned)
3. Spatial metrics computation
4. Complete multimodal analysis prompt
5. Output schema validation

The key insight: Gemini must see BOTH the original satellite image AND
the U-Net predictions to perform meaningful spatial reasoning.
"""

import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import PIL.Image
    from PIL import Image
except ImportError:
    PIL = None
    Image = None


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SeverityLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNBURNED = "unburned"


class SegmentationQuality(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class InterventionType(str, Enum):
    ACTIVE_REFORESTATION = "active_reforestation"
    EROSION_CONTROL = "erosion_control"
    ASSISTED_REGENERATION = "assisted_regeneration"
    MONITOR_ONLY = "monitor_only"


class Urgency(str, Enum):
    IMMEDIATE = "immediate"
    SIX_MONTHS = "6_months"
    ONE_YEAR = "1_year"
    TWO_THREE_YEARS = "2_3_years"


@dataclass
class SeverityThresholds:
    """Configurable thresholds for severity classification."""
    high: float = 0.66
    moderate: float = 0.27
    low: float = 0.1


@dataclass 
class OverlayColors:
    """RGBA colors for severity overlay visualization."""
    high: Tuple[int, int, int, int] = (255, 0, 0, 180)      # Red, semi-transparent
    moderate: Tuple[int, int, int, int] = (255, 165, 0, 150)  # Orange
    low: Tuple[int, int, int, int] = (255, 255, 0, 120)       # Yellow
    unburned: Tuple[int, int, int, int] = (0, 0, 0, 0)        # Transparent


# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def normalize_rgb_tile(
    rgb_tile: np.ndarray,
    percentile_low: float = 2,
    percentile_high: float = 98
) -> np.ndarray:
    """
    Normalize RGB tile to 0-255 with percentile contrast stretch.
    
    Args:
        rgb_tile: (3, H, W) or (H, W, 3) RGB array, any dtype
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping
        
    Returns:
        (H, W, 3) uint8 array normalized to 0-255
    """
    # Ensure (H, W, 3) format
    if rgb_tile.ndim == 3 and rgb_tile.shape[0] == 3:
        rgb_tile = np.moveaxis(rgb_tile, 0, -1)
    
    # Handle per-channel normalization for better visual quality
    result = np.zeros_like(rgb_tile, dtype=np.float32)
    for i in range(3):
        channel = rgb_tile[:, :, i].astype(np.float32)
        p_low = np.percentile(channel, percentile_low)
        p_high = np.percentile(channel, percentile_high)
        if p_high > p_low:
            result[:, :, i] = (channel - p_low) / (p_high - p_low)
        else:
            result[:, :, i] = 0
    
    # Clip and scale to 0-255
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def create_severity_overlay(
    severity_map: np.ndarray,
    thresholds: SeverityThresholds = None,
    colors: OverlayColors = None
) -> np.ndarray:
    """
    Create RGBA severity overlay with ecologically-meaningful colors.
    
    Colors are chosen for restoration planning context:
    - High severity (Red): Active restoration needed
    - Moderate (Orange): Monitor/light intervention
    - Low (Yellow): Natural regeneration likely
    - Unburned (Transparent): Not relevant
    
    Args:
        severity_map: 2D numpy array with values 0-1
        thresholds: Custom severity thresholds
        colors: Custom RGBA colors
        
    Returns:
        (H, W, 4) RGBA uint8 array
    """
    thresholds = thresholds or SeverityThresholds()
    colors = colors or OverlayColors()
    
    h, w = severity_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # High severity
    high_mask = severity_map > thresholds.high
    rgba[high_mask] = colors.high
    
    # Moderate severity
    moderate_mask = (severity_map > thresholds.moderate) & (severity_map <= thresholds.high)
    rgba[moderate_mask] = colors.moderate
    
    # Low severity
    low_mask = (severity_map > thresholds.low) & (severity_map <= thresholds.moderate)
    rgba[low_mask] = colors.low
    
    # Unburned remains transparent (default zeros)
    
    return rgba


def create_probability_heatmap(
    probability_map: np.ndarray,
    colormap: str = "viridis"
) -> 'Image.Image':
    """
    Create a continuous probability heatmap image.
    
    Args:
        probability_map: 2D array with values 0-1 (raw sigmoid output)
        colormap: Matplotlib colormap name
        
    Returns:
        PIL Image with colorized probability map
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Ensure 0-1 range
    prob_normalized = np.clip(probability_map, 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(prob_normalized)
    
    # Convert to 8-bit RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(rgb)


def create_image_pack(
    rgb_tile: np.ndarray,
    severity_map: np.ndarray,
    probability_map: np.ndarray = None,
    include_probability: bool = False
) -> List['Image.Image']:
    """
    Create the image pack for Gemini multimodal analysis.
    
    This is the core function that prepares images for true multimodal
    reasoning. Gemini receives:
    1. Original RGB satellite image (landscape context)
    2. Severity overlay (U-Net predictions in context)
    3. Optional: Probability heatmap (uncertainty visualization)
    
    Args:
        rgb_tile: (3, H, W) or (H, W, 3) RGB bands from Sentinel-2 (B4, B3, B2)
        severity_map: (H, W) float32 burn severity predictions 0-1
        probability_map: Optional raw sigmoid output for uncertainty
        include_probability: Whether to include probability heatmap
        
    Returns:
        List of 2-3 PIL Images for Gemini analysis
    """
    if Image is None:
        raise ImportError("PIL is required for image pack creation")
    
    images = []
    
    # Image 1: Original RGB (normalized for visualization)
    rgb_normalized = normalize_rgb_tile(rgb_tile)
    original_image = Image.fromarray(rgb_normalized)
    images.append(original_image)
    
    # Image 2: Severity Overlay (RGB + severity with transparency)
    severity_rgba = create_severity_overlay(severity_map)
    
    base = original_image.convert('RGBA')
    overlay = Image.fromarray(severity_rgba)
    composite = Image.alpha_composite(base, overlay)
    images.append(composite.convert('RGB'))
    
    # Image 3 (Optional): Probability heatmap for uncertainty
    if include_probability and probability_map is not None:
        prob_image = create_probability_heatmap(probability_map)
        images.append(prob_image)
    
    return images


# =============================================================================
# SPATIAL METRICS COMPUTATION
# =============================================================================

def compute_severity_statistics(severity_map: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics from burn severity map.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        
    Returns:
        Dict with severity statistics
    """
    thresholds = SeverityThresholds()
    
    return {
        "mean_severity": float(np.mean(severity_map)),
        "max_severity": float(np.max(severity_map)),
        "min_severity": float(np.min(severity_map)),
        "std_severity": float(np.std(severity_map)),
        "high_severity_ratio": float(np.mean(severity_map > thresholds.high)),
        "moderate_severity_ratio": float(
            np.mean((severity_map > thresholds.moderate) & (severity_map <= thresholds.high))
        ),
        "low_severity_ratio": float(
            np.mean((severity_map > thresholds.low) & (severity_map <= thresholds.moderate))
        ),
        "unburned_ratio": float(np.mean(severity_map <= thresholds.low)),
    }


def compute_spatial_metrics(
    severity_map: np.ndarray,
    pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Compute spatial metrics for fragmentation analysis.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        pixel_size_m: Pixel size in meters (Sentinel-2 = 10m)
        
    Returns:
        Dict with spatial metrics
    """
    from scipy import ndimage
    
    thresholds = SeverityThresholds()
    
    # Binary mask of burned area (> low threshold)
    burned_mask = (severity_map > thresholds.low).astype(np.int32)
    
    # Label connected components
    labeled, num_patches = ndimage.label(burned_mask)
    
    if num_patches == 0:
        return {
            "num_distinct_patches": 0,
            "largest_patch_hectares": 0.0,
            "edge_density_m_per_ha": 0.0,
            "mean_patch_shape_index": 0.0,
            "landscape_fragmentation_index": 0.0
        }
    
    # Compute patch sizes
    patch_sizes = ndimage.sum(burned_mask, labeled, range(1, num_patches + 1))
    largest_patch_pixels = max(patch_sizes) if len(patch_sizes) > 0 else 0
    
    # Convert to hectares (1 ha = 10,000 m²)
    pixel_area_m2 = pixel_size_m ** 2
    largest_patch_ha = (largest_patch_pixels * pixel_area_m2) / 10000
    
    # Calculate edge pixels (simple approximation)
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(burned_mask)
    edge_pixels = np.sum(burned_mask) - np.sum(interior)
    
    # Edge density (m per hectare)
    total_area_ha = (np.sum(burned_mask) * pixel_area_m2) / 10000
    edge_length_m = edge_pixels * pixel_size_m
    edge_density = edge_length_m / total_area_ha if total_area_ha > 0 else 0
    
    # Fragmentation index (0 = single connected patch, 1 = highly fragmented)
    fragmentation = 1 - (largest_patch_pixels / np.sum(burned_mask)) if np.sum(burned_mask) > 0 else 0
    
    return {
        "num_distinct_patches": int(num_patches),
        "largest_patch_hectares": float(largest_patch_ha),
        "edge_density_m_per_ha": float(edge_density),
        "mean_patch_shape_index": 1.5,  # Placeholder - needs perimeter calculation
        "landscape_fragmentation_index": float(fragmentation)
    }


def build_gemini_context(
    location: Tuple[float, float],
    severity_map: np.ndarray,
    metadata: Dict[str, Any] = None,
    unet_confidence: float = None,
    pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Build the complete structured context JSON for Gemini.
    
    Args:
        location: (latitude, longitude) tuple
        severity_map: 2D numpy array with values 0-1
        metadata: Additional metadata (ecosystem, legal, temporal)
        unet_confidence: Model confidence score
        pixel_size_m: Pixel size in meters
        
    Returns:
        Complete context dict for Gemini prompt
    """
    metadata = metadata or {}
    
    # Compute area in hectares
    h, w = severity_map.shape
    area_ha = (h * w * (pixel_size_m ** 2)) / 10000
    
    context = {
        "location": {
            "lat": location[0],
            "lon": location[1],
            "elevation_m": metadata.get("elevation_m"),
            "aspect_degrees": metadata.get("aspect_degrees"),
            "slope_percent": metadata.get("slope_percent")
        },
        "ecosystem_context": {
            "ecoregion": metadata.get("ecoregion"),
            "biome": metadata.get("biome"),
            "pre_fire_ndvi": metadata.get("pre_fire_ndvi"),
            "current_ndvi": metadata.get("current_ndvi")
        },
        "burn_statistics": {
            "area_hectares": float(area_ha),
            **compute_severity_statistics(severity_map)
        },
        "spatial_metrics": compute_spatial_metrics(severity_map, pixel_size_m),
        "temporal_context": {
            "fire_date": metadata.get("fire_date"),
            "days_since_fire": metadata.get("days_since_fire"),
            "season": metadata.get("season")
        },
        "legal_context": {
            "land_ownership": metadata.get("land_ownership"),
            "protected_status": metadata.get("protected_status"),
            "known_constraints": metadata.get("known_constraints", [])
        },
        "model_metadata": {
            "unet_version": metadata.get("unet_version", "california_fire_v2"),
            "confidence_score": unet_confidence,
            "cloud_mask_applied": metadata.get("cloud_mask_applied", False)
        }
    }
    
    # Remove None values for cleaner JSON
    return _clean_none_values(context)


def _clean_none_values(d: Dict) -> Dict:
    """Recursively remove None values from dict."""
    if not isinstance(d, dict):
        return d
    return {
        k: _clean_none_values(v) 
        for k, v in d.items() 
        if v is not None and v != []
    }


# =============================================================================
# GEMINI PROMPT
# =============================================================================

MULTIMODAL_ANALYSIS_PROMPT = '''You are an expert restoration ecologist with remote sensing expertise.

## YOUR TASK
Analyze the provided satellite imagery and U-Net burn severity prediction.
Your analysis bridges pixel-level predictions to restoration-actionable intelligence.

## IMAGES PROVIDED
1. **Original Satellite Image** (RGB true color composite from Sentinel-2)
2. **Severity Overlay** (U-Net predictions overlaid: Red=high severity, Orange=moderate, Yellow=low, Transparent=unburned)
{probability_image_clause}

## LOCATION & CONTEXT
```json
{context_json}
```

---

## ANALYSIS REQUIREMENTS

### STEP 1: Ground the Prediction in Reality
BEFORE analyzing the severity map, describe what you SEE in the original image:
- What land cover types are visible? (forest, grassland, shrubland, developed, water, bare soil)
- Are there visible terrain features? (ridgelines, valleys, roads, water bodies)
- What is the pre-fire vegetation density/texture?

### STEP 2: Evaluate U-Net Prediction Quality
Examine the overlay image for segmentation artifacts:
- [ ] Are severity boundaries crisp or noisy/speckled?
- [ ] Do high-severity zones align with visible burned areas?
- [ ] Are there unexpected patterns (linear artifacts, edge effects, cloud shadows misclassified)?
- [ ] Does the severity gradient look ecologically plausible?

**FLAG any issues explicitly. Do NOT proceed with high confidence if the segmentation looks problematic.**

### STEP 3: Spatial Pattern Analysis
From the overlay image, analyze:

**Fragmentation Geometry**
- How many distinct burn patches are visible?
- Are patches connected or isolated?
- What is the approximate shape complexity?

**Edge Characteristics**
- Are burn edges sharp or diffuse?
- Do edges follow topographic features?
- Are there unburned inclusions (green islands)?

**Severity Gradients**
- Is there a dominant direction to severity transitions?
- Are there abrupt transitions that might indicate fire behavior?

### STEP 4: Ecological Interpretation
Cross-reference the original image with the severity overlay:

**Differential Impact Assessment**
- Which vegetation types received HIGH severity?
- Are there surviving seed sources near high severity areas?
- What is the regeneration potential based on visible landscape structure?

**Priority Zone Identification**
Identify 3-5 priority zones for restoration intervention.

### STEP 5: Uncertainty and Limitations
Be explicit about what you CANNOT determine from these images.

---

## OUTPUT FORMAT
Return ONLY valid JSON matching this schema (no markdown, no explanation outside JSON):

{{
  "visual_grounding": {{
    "observed_land_cover": ["list of visible land cover types"],
    "terrain_features": ["visible terrain features"],
    "pre_fire_vegetation_description": "string"
  }},
  "segmentation_quality": {{
    "overall_quality": "good | acceptable | poor | unusable",
    "boundary_sharpness": "crisp | somewhat_diffuse | noisy",
    "artifact_flags": [],
    "confidence_in_prediction": 0.0,
    "quality_notes": "string"
  }},
  "spatial_patterns": {{
    "fragmentation_assessment": {{
      "patch_count_visual": "integer or many",
      "connectivity": "connected | partially_connected | fragmented | highly_fragmented",
      "shape_complexity": "compact | moderate | irregular | highly_irregular"
    }},
    "edge_characteristics": {{
      "edge_sharpness": "sharp | diffuse | mixed",
      "topographic_alignment": "yes | partial | no",
      "unburned_inclusions": true,
      "inclusion_significance": "string"
    }},
    "gradient_analysis": {{
      "dominant_direction": "N | NE | E | SE | S | SW | W | NW | none | radial",
      "transition_type": "gradual | abrupt | mixed",
      "fire_behavior_inference": "string"
    }}
  }},
  "ecological_interpretation": {{
    "differential_impacts": [
      {{
        "vegetation_type": "string",
        "severity_level": "high | moderate | low",
        "ecological_significance": "string"
      }}
    ],
    "seed_source_availability": "abundant | limited | none_visible",
    "natural_regeneration_potential": "high | moderate | low | very_low",
    "regeneration_rationale": "string"
  }},
  "priority_zones": [
    {{
      "zone_id": "A",
      "location_description": "string",
      "area_percent_estimate": 0.0,
      "severity": "high | moderate | low",
      "priority_reason": "string",
      "recommended_intervention": "active_reforestation | erosion_control | assisted_regeneration | monitor_only",
      "urgency": "immediate | 6_months | 1_year | 2_3_years"
    }}
  ],
  "uncertainty_flags": {{
    "cannot_determine": [],
    "requests_additional_data": []
  }},
  "signals_for_final_model": {{
    "restoration_potential_score": 0.0,
    "intervention_urgency_score": 0.0,
    "ecological_complexity_score": 0.0,
    "risk_score": 0.0,
    "confidence_weighted_severity": 0.0
  }},
  "reasoning_trace": "Brief explanation of key analytical steps taken"
}}
'''


def build_multimodal_prompt(
    context: Dict[str, Any],
    include_probability_image: bool = False
) -> str:
    """
    Build the complete Gemini analysis prompt with context.
    
    Args:
        context: Structured context from build_gemini_context()
        include_probability_image: Whether probability heatmap is included
        
    Returns:
        Complete prompt string
    """
    probability_clause = ""
    if include_probability_image:
        probability_clause = (
            "3. **Probability Heatmap** (Continuous 0-1 prediction confidence. "
            "Bright=high confidence burn, dark=uncertain/unburned)\n"
            "   Pay attention to mid-range values (0.3-0.7) as these indicate model uncertainty."
        )
    
    context_json = json.dumps(context, indent=2)
    
    return MULTIMODAL_ANALYSIS_PROMPT.format(
        probability_image_clause=probability_clause,
        context_json=context_json
    )


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_gemini_output(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate Gemini output against expected schema.
    
    Args:
        response: Parsed JSON response from Gemini
        
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    required_keys = [
        "visual_grounding",
        "segmentation_quality", 
        "spatial_patterns",
        "ecological_interpretation",
        "priority_zones",
        "signals_for_final_model"
    ]
    
    for key in required_keys:
        if key not in response:
            errors.append(f"Missing required key: {key}")
    
    # Validate segmentation quality
    if "segmentation_quality" in response:
        sq = response["segmentation_quality"]
        valid_qualities = ["good", "acceptable", "poor", "unusable"]
        if sq.get("overall_quality") not in valid_qualities:
            errors.append(f"Invalid overall_quality: {sq.get('overall_quality')}")
        
        confidence = sq.get("confidence_in_prediction", 0)
        if not (0 <= confidence <= 1):
            errors.append(f"confidence_in_prediction must be 0-1, got {confidence}")
    
    # Validate signals
    if "signals_for_final_model" in response:
        signals = response["signals_for_final_model"]
        signal_keys = [
            "restoration_potential_score",
            "intervention_urgency_score",
            "ecological_complexity_score",
            "risk_score",
            "confidence_weighted_severity"
        ]
        for key in signal_keys:
            if key in signals:
                val = signals[key]
                if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                    errors.append(f"{key} must be 0-1, got {val}")
    
    return (len(errors) == 0, errors)


def should_trigger_human_review(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Determine if Gemini output requires human verification.
    
    Args:
        response: Parsed Gemini response
        
    Returns:
        (should_review, list of trigger reasons)
    """
    triggers = []
    
    sq = response.get("segmentation_quality", {})
    
    # Quality triggers
    if sq.get("overall_quality") in ["poor", "unusable"]:
        triggers.append("segmentation_quality_low")
    
    if len(sq.get("artifact_flags", [])) > 0:
        triggers.append("artifacts_detected")
    
    if sq.get("confidence_in_prediction", 1.0) < 0.6:
        triggers.append("gemini_low_confidence")
    
    # Risk triggers
    signals = response.get("signals_for_final_model", {})
    if signals.get("risk_score", 0) > 0.7:
        triggers.append("high_risk_site")
    
    # Urgency triggers
    for zone in response.get("priority_zones", []):
        if zone.get("urgency") == "immediate":
            triggers.append(f"immediate_urgency_{zone.get('zone_id', 'unknown')}")
    
    return (len(triggers) > 0, triggers)


# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================

class MultimodalAnalyzer:
    """
    True multimodal analyzer for U-Net → Gemini integration.
    
    This class properly implements the multimodal pipeline where:
    1. Original RGB + Severity overlay are passed as images
    2. Structured context provides metadata
    3. Gemini performs spatial reasoning on actual imagery
    4. Output is validated and includes machine-readable signals
    """
    
    def __init__(self, gemini_client: 'EcoReviveGemini'):
        """
        Initialize the multimodal analyzer.
        
        Args:
            gemini_client: Configured EcoReviveGemini client
        """
        self.client = gemini_client
    
    def analyze(
        self,
        rgb_tile: np.ndarray,
        severity_map: np.ndarray,
        location: Tuple[float, float],
        metadata: Dict[str, Any] = None,
        unet_confidence: float = None,
        include_probability: bool = False,
        probability_map: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Perform true multimodal analysis of U-Net predictions.
        
        Args:
            rgb_tile: (3, H, W) or (H, W, 3) RGB bands from Sentinel-2
            severity_map: (H, W) float32 burn severity predictions 0-1
            location: (latitude, longitude) tuple
            metadata: Additional context (ecosystem, legal, temporal)
            unet_confidence: U-Net model confidence score
            include_probability: Whether to include probability heatmap
            probability_map: Raw sigmoid output for uncertainty visualization
            
        Returns:
            Complete analysis result with signals for Layer 3
        """
        # Create image pack
        images = create_image_pack(
            rgb_tile=rgb_tile,
            severity_map=severity_map,
            probability_map=probability_map,
            include_probability=include_probability
        )
        
        # Build context
        context = build_gemini_context(
            location=location,
            severity_map=severity_map,
            metadata=metadata,
            unet_confidence=unet_confidence
        )
        
        # Build prompt
        prompt = build_multimodal_prompt(
            context=context,
            include_probability_image=include_probability
        )
        
        # Call Gemini with multimodal input
        print(f"🔬 Running multimodal analysis at ({location[0]:.4f}, {location[1]:.4f})...")
        print(f"   Sending {len(images)} images to Gemini")
        
        response = self.client.analyze_multimodal(
            prompt=prompt,
            images=images,
            use_json=True
        )
        
        if not response.get('parsed'):
            return {
                'status': 'error',
                'error': 'Failed to parse Gemini response',
                'raw_response': response.get('text', '')[:500],
                'tokens_used': response.get('usage')
            }
        
        gemini_output = response['parsed']
        
        # Validate output
        is_valid, validation_errors = validate_gemini_output(gemini_output)
        
        # Check if human review needed
        needs_review, review_triggers = should_trigger_human_review(gemini_output)
        
        result = {
            'status': 'complete',
            'analysis': gemini_output,
            'validation': {
                'is_valid': is_valid,
                'errors': validation_errors
            },
            'human_review': {
                'required': needs_review,
                'triggers': review_triggers
            },
            'metadata': {
                'location': location,
                'images_sent': len(images),
                'tokens_used': response.get('usage')
            }
        }
        
        # Quality gate check
        sq = gemini_output.get('segmentation_quality', {})
        if sq.get('overall_quality') == 'unusable':
            result['status'] = 'quality_gate_failed'
            result['reason'] = sq.get('quality_notes', 'Segmentation unusable')
        
        print(f"   ✅ Analysis complete. Quality: {sq.get('overall_quality', 'unknown')}")
        if needs_review:
            print(f"   ⚠️ Human review required: {', '.join(review_triggers)}")
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_with_multimodal(
    rgb_tile: np.ndarray,
    severity_map: np.ndarray,
    location: Tuple[float, float],
    gemini_client: 'EcoReviveGemini' = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for multimodal analysis.
    
    Args:
        rgb_tile: RGB bands from Sentinel-2
        severity_map: U-Net severity predictions
        location: (lat, lon) tuple
        gemini_client: Optional pre-configured client
        metadata: Additional context
        **kwargs: Additional arguments for analyze()
        
    Returns:
        Multimodal analysis result
    """
    if gemini_client is None:
        from .gemini_client import create_client
        gemini_client = create_client()
    
    analyzer = MultimodalAnalyzer(gemini_client)
    return analyzer.analyze(
        rgb_tile=rgb_tile,
        severity_map=severity_map,
        location=location,
        metadata=metadata,
        **kwargs
    )


if __name__ == "__main__":
    # Demo mode
    print("=" * 60)
    print("🔬 EcoRevive Multimodal Analyzer Demo")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Synthetic RGB tile (simulating Sentinel-2 B4, B3, B2)
    rgb = np.random.randint(100, 200, size=(3, 256, 256), dtype=np.uint8)
    
    # Synthetic severity map with spatial structure
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    severity = 0.5 * np.exp(-((x - 0.5)**2 + (y - 0.3)**2) / 0.1)
    severity += 0.3 * np.random.beta(2, 1, size=(256, 256))
    severity = np.clip(severity, 0, 1).astype(np.float32)
    
    location = (40.05, -121.20)  # Dixie Fire location
    
    # Test image pack creation
    print("\n📸 Testing image pack creation...")
    images = create_image_pack(rgb, severity)
    print(f"   Created {len(images)} images:")
    for i, img in enumerate(images):
        print(f"   - Image {i+1}: {img.size}, mode={img.mode}")
    
    # Test context building
    print("\n📋 Testing context building...")
    context = build_gemini_context(
        location=location,
        severity_map=severity,
        metadata={"ecoregion": "sierra_nevada"},
        unet_confidence=0.87
    )
    print(f"   Context keys: {list(context.keys())}")
    print(f"   Burn stats: mean={context['burn_statistics']['mean_severity']:.2%}")
    
    # Test prompt building
    print("\n📝 Testing prompt building...")
    prompt = build_multimodal_prompt(context)
    print(f"   Prompt length: {len(prompt)} characters")
    
    print("\n✅ All module components working correctly!")
    print("\nTo run full analysis, set GOOGLE_API_KEY and use:")
    print("   from reasoning.gemini_multimodal import analyze_with_multimodal")
    print("   result = analyze_with_multimodal(rgb, severity, location)")
