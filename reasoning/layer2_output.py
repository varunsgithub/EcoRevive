"""
EcoRevive Layer 2 Structured Output Module
==========================================
Produces structured, machine-readable spatial analysis for Layer 3 consumption.

SCHEMA VERSION: 2.0

COORDINATE CONVENTION: Internal functions use (lat, lon) order.
GeoJSON outputs use (lon, lat) order per spec. Check each function's docstring.

Layer 2 responsibilities:
- Spatial zone extraction (burn scar, healthy, degraded, etc.)
- Risk grid generation (H3-indexed cells)
- Hazard annotation (dead trees, steep slopes, erosion)
- Site characteristic extraction (soil, terrain, ecosystem)
- Baseline metric computation (NDVI, canopy %, burn %)
- Confidence calibration

All outputs are JSON-serializable for storage/transport to Layer 3.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# SCHEMA VERSION & CONFIGURATION CONSTANTS
# =============================================================================

SCHEMA_VERSION = "2.0"

# Zone detection thresholds (empirically tuned for U-Net burn severity model)
# Source: Calibrated against CAL FIRE severity classifications
ZONE_THRESHOLDS = {
    "high_severity": 0.66,      # >66% = high severity burn
    "moderate_severity": 0.27,  # 27-66% = moderate severity
    "low_severity": 0.10,       # 10-27% = low severity
}

# Minimum patch size to include (based on Sentinel-2 10m resolution)
# 10 pixels = 1000m² = 0.1 hectares = minimum ecologically significant patch
MIN_PATCH_PIXELS = 10

# Maximum patches per severity class (to prevent JSON bloat)
MAX_PATCHES_PER_CLASS = 5

# Confidence blending weights (model vs Gemini)
# TODO: Validate optimal weights through cross-validation study
# Current 60/40 weighting is a preliminary estimate:
# - U-Net: Higher spatial precision from trained segmentation
# - Gemini: Better contextual/semantic understanding from VLM
MODEL_CONFIDENCE_WEIGHT = 0.6
GEMINI_CONFIDENCE_WEIGHT = 0.4

# NOTE: Recovery time estimates removed - these require scientific citations
# and should be computed by Layer 3 using RAG with peer-reviewed literature.

# Rate limiting for external APIs (Nominatim ToS compliance)
_last_geocode_time = 0.0
NOMINATIM_RATE_LIMIT_SECONDS = 1.1  # >1 req/sec per Nominatim ToS

# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_severity_map(severity_map: np.ndarray) -> None:
    """
    Validate severity map input.
    
    Raises:
        ValueError: If severity_map is invalid
    """
    if severity_map is None or severity_map.size == 0:
        raise ValueError("severity_map cannot be None or empty")
    
    if severity_map.ndim != 2:
        raise ValueError(f"severity_map must be 2D, got {severity_map.ndim}D")
    
    min_val, max_val = severity_map.min(), severity_map.max()
    if min_val < 0 or max_val > 1:
        logger.warning(f"severity_map values outside [0,1]: min={min_val:.3f}, max={max_val:.3f}")
        # Don't raise - just warn and clip later


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validate geographic coordinates.
    
    Raises:
        ValueError: If coordinates are out of valid range
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be in [-90, 90], got {lat}")
    
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude must be in [-180, 180], got {lon}")


def validate_bbox(bbox: Dict[str, float]) -> None:
    """
    Validate bounding box has required keys and valid values.
    
    Raises:
        ValueError: If bbox is missing keys or has invalid values
    """
    required_keys = {'west', 'south', 'east', 'north'}
    if not required_keys.issubset(bbox.keys()):
        raise ValueError(f"bbox missing required keys: {required_keys - set(bbox.keys())}")
    
    if bbox['south'] > bbox['north']:
        raise ValueError(f"bbox south ({bbox['south']}) > north ({bbox['north']})")
    
    if bbox['west'] > bbox['east']:
        # Allow for antimeridian crossing
        logger.warning("bbox west > east - possible antimeridian crossing")


# =============================================================================
# ENUMS
# =============================================================================

class ZoneType(str, Enum):
    """Types of spatial zones detected in imagery."""
    BURN_SCAR = "burn_scar"
    HEALTHY_VEGETATION = "healthy_vegetation"
    DEGRADED_VEGETATION = "degraded_vegetation"
    BARE_SOIL = "bare_soil"
    WATER = "water"
    STRUCTURE = "structure"
    UNCERTAIN = "uncertain"


class HazardType(str, Enum):
    """Types of safety hazards."""
    STANDING_DEAD_TREE = "standing_dead_tree"
    STEEP_SLOPE = "steep_slope"
    EROSION_CHANNEL = "erosion_channel"
    UNSTABLE_TERRAIN = "unstable_terrain"
    WATER_HAZARD = "water_hazard"
    WIDOWMAKER = "widowmaker"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk levels for safety assessment."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class SuccessionStage(str, Enum):
    """Ecological succession stages."""
    PIONEER = "pioneer"
    EARLY = "early"
    MID = "mid"
    LATE_CLIMAX = "late_climax"
    DISTURBED = "disturbed"


# =============================================================================
# DATACLASSES - Location & Terrain
# =============================================================================

@dataclass
class LocationContext:
    """Geographic context for the analysis site."""
    latitude: float  # Geometric center
    longitude: float  # Geometric center
    area_name: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    elevation_m: Optional[float] = None
    bounding_box: Optional[Dict[str, float]] = None  # west, south, east, north
    
    # Severity-weighted damage centroid (professional restoration focus)
    damage_centroid_lat: Optional[float] = None  # Center of damage, not box
    damage_centroid_lon: Optional[float] = None
    centroid_type: str = "geometric"  # "geometric" or "severity_weighted"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TerrainData:
    """Terrain characteristics extracted from imagery/DEM."""
    slope_degrees_mean: Optional[float] = None
    slope_degrees_max: Optional[float] = None
    aspect: Optional[str] = None  # N, NE, E, SE, S, SW, W, NW, flat
    instability_score: Optional[float] = None  # 0-1
    drainage_pattern: Optional[str] = None  # well_drained, moderate, poor, waterlogged
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# DATACLASSES - Site Characteristics
# =============================================================================

@dataclass
class SiteCharacteristics:
    """Biophysical site characteristics for restoration planning."""
    # Soil properties
    soil_type: Optional[str] = None
    soil_texture: Optional[str] = None  # sandy, loamy, clay, etc.
    nutrient_profile: Optional[str] = None  # low, moderate, high
    ph_level: Optional[float] = None
    
    # Terrain
    terrain: Optional[TerrainData] = None
    
    # History
    land_use_history: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                if isinstance(v, dict):
                    result[k] = {kk: vv for kk, vv in v.items() if vv is not None}
                else:
                    result[k] = v
        return result


# =============================================================================
# DATACLASSES - Ecosystem
# =============================================================================

@dataclass
class EcosystemInfo:
    """Ecosystem classification and reference matching."""
    current_state: str  # e.g., "severely_burned_mixed_conifer"
    pre_degradation_ecosystem: Optional[str] = None  # e.g., "mixed_conifer_forest"
    reference_ecosystem_match: Optional[str] = None  # closest healthy analog
    match_confidence: float = 0.0
    succession_stage: str = "disturbed"
    vegetation_types: List[str] = field(default_factory=list)
    key_species: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None and v != []}


# =============================================================================
# DATACLASSES - Metrics
# =============================================================================

@dataclass
class SiteMetrics:
    """Computed metrics for the site."""
    # Vegetation indices
    ndvi_mean: float = 0.0
    ndvi_std: float = 0.0
    ndvi_min: float = 0.0
    ndvi_max: float = 0.0
    
    # Cover percentages
    canopy_cover_pct: float = 0.0
    burn_pct: float = 0.0
    high_severity_pct: float = 0.0
    moderate_severity_pct: float = 0.0
    low_severity_pct: float = 0.0
    unburned_pct: float = 0.0
    
    # Coverage analytics (professional restoration metrics)
    total_area_km2: float = 0.0
    damaged_area_km2: float = 0.0  # Area with >10% severity
    high_severity_area_km2: float = 0.0  # Area with >60% severity
    damage_coverage_pct: float = 0.0  # What % of selection is damaged
    
    # Recovery estimates - set by Layer 3 using RAG with scientific literature
    estimated_healing_years: Optional[Dict[str, float]] = None
    
    # Model outputs
    mean_severity: float = 0.0
    max_severity: float = 0.0
    model_confidence: float = 0.0
    
    # Area
    total_area_hectares: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# DATACLASSES - Spatial Primitives
# =============================================================================

@dataclass
class ZonePrimitive:
    """A discrete spatial zone with classification."""
    zone_id: str
    zone_type: str  # ZoneType value
    geometry_type: str = "polygon"  # polygon, point
    # GeoJSON coordinates (simplified - can be expanded)
    bbox: Optional[Dict[str, float]] = None  # west, south, east, north
    centroid: Optional[Tuple[float, float]] = None  # (lon, lat)
    area_hectares: float = 0.0
    confidence: float = 0.0
    uncertainty_radius_m: float = 0.0
    
    # Zone-specific metrics
    severity_mean: Optional[float] = None
    ndvi_mean: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class HazardAnnotation:
    """A detected safety hazard with location."""
    hazard_id: str
    hazard_type: str  # HazardType value
    geometry_type: str = "point"  # point, polygon
    location: Optional[Tuple[float, float]] = None  # (lon, lat)
    bbox: Optional[Dict[str, float]] = None
    severity: str = "moderate"  # low, moderate, high
    detection_confidence: float = 0.0
    buffer_radius_m: float = 10.0
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class RiskCell:
    """A single cell in the risk grid."""
    cell_id: str  # H3 index or simple grid ID
    risk_level: str  # RiskLevel value
    risk_score: float  # 0-1
    contributing_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskGrid:
    """Risk assessment grid for the site."""
    resolution: int = 9  # Grid resolution (H3 level or grid size)
    grid_type: str = "h3"  # h3 or simple_grid
    cells: List[RiskCell] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "grid_type": self.grid_type,
            "cells": [c.to_dict() for c in self.cells]
        }


# =============================================================================
# DATACLASSES - Confidence & Metadata
# =============================================================================

@dataclass
class ConfidenceMetadata:
    """Confidence scores and quality flags."""
    overall_confidence: float = 0.0
    segmentation_quality: str = "unknown"  # good, acceptable, poor, unusable
    classification_reliability: float = 0.0
    data_quality_flags: List[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_triggers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingMetadata:
    """Processing pipeline metadata."""
    layer1_model_version: str = "california_fire_v2"
    layer2_gemini_model: str = "gemini-2.0-flash"
    processing_time_ms: int = 0
    imagery_source: str = "sentinel2"
    imagery_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# MAIN LAYER 2 OUTPUT CLASS
# =============================================================================

@dataclass
class Layer2Output:
    """
    Complete Layer 2 structured output.
    
    This is the main output class that contains all structured data
    for Layer 3 consumption. All fields are JSON-serializable.
    """
    # Identification (required fields first)
    analysis_id: str
    timestamp: str  # ISO 8601
    
    # Location & Context
    location: LocationContext
    
    # Site Characteristics
    characteristics: SiteCharacteristics
    
    # Ecosystem Classification
    ecosystem: EcosystemInfo
    
    # Computed Metrics
    metrics: SiteMetrics
    
    # --- Fields with defaults below ---
    
    # Schema version for backward compatibility
    schema_version: str = SCHEMA_VERSION
    
    # Spatial Primitives
    zones: List[ZonePrimitive] = field(default_factory=list)
    hazards: List[HazardAnnotation] = field(default_factory=list)
    risk_grid: Optional[RiskGrid] = None
    
    # Confidence & Quality
    confidence: ConfidenceMetadata = field(default_factory=ConfidenceMetadata)
    
    # Processing Metadata
    processing: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    
    # Signals for Layer 3 (machine-readable scores)
    signals: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict with consistent null handling."""
        result = {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "location": self.location.to_dict(),
            "characteristics": self.characteristics.to_dict(),
            "ecosystem": self.ecosystem.to_dict(),
            "metrics": self.metrics.to_dict(),
            "zones": [z.to_dict() for z in self.zones],
            "hazards": [h.to_dict() for h in self.hazards],
            "risk_grid": self.risk_grid.to_dict() if self.risk_grid else None,
            "confidence": self.confidence.to_dict(),
            "processing": self.processing.to_dict(),
            "signals": self.signals
        }
        # Consistent null handling: remove None values and empty lists
        return {k: v for k, v in result.items() if v is not None and v != []}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def create_empty(cls, lat: float, lon: float) -> 'Layer2Output':
        """Create an empty Layer2Output with minimal required fields."""
        return cls(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            location=LocationContext(latitude=lat, longitude=lon),
            characteristics=SiteCharacteristics(),
            ecosystem=EcosystemInfo(current_state="unknown"),
            metrics=SiteMetrics()
        )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_metrics_from_severity(
    severity_map: np.ndarray,
    pixel_size_m: float = 10.0,
    thresholds: Dict[str, float] = None
) -> SiteMetrics:
    """
    Compute SiteMetrics from a severity map.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        pixel_size_m: Pixel size in meters
        thresholds: Custom thresholds for classification
        
    Returns:
        SiteMetrics object with computed values
    """
    if thresholds is None:
        thresholds = {"high": 0.66, "moderate": 0.27, "low": 0.1}
    
    # Basic stats
    mean_sev = float(np.mean(severity_map))
    max_sev = float(np.max(severity_map))
    
    # Area calculation
    h, w = severity_map.shape
    pixel_area_m2 = pixel_size_m ** 2
    total_area_ha = (h * w * pixel_area_m2) / 10000
    
    # Severity percentages
    high_pct = float(np.mean(severity_map > thresholds["high"]))
    moderate_pct = float(np.mean(
        (severity_map > thresholds["moderate"]) & (severity_map <= thresholds["high"])
    ))
    low_pct = float(np.mean(
        (severity_map > thresholds["low"]) & (severity_map <= thresholds["moderate"])
    ))
    unburned_pct = float(np.mean(severity_map <= thresholds["low"]))
    
    # Burn percentage (anything above low threshold)
    burn_pct = float(np.mean(severity_map > thresholds["low"]))
    
    # NOTE: Recovery time estimates removed - requires peer-reviewed citations
    # Layer 3 should compute this using RAG with scientific literature
    
    return SiteMetrics(
        ndvi_mean=0.0,  # Would require NIR band calculation
        ndvi_std=0.0,
        ndvi_min=0.0,
        ndvi_max=0.0,
        canopy_cover_pct=unburned_pct * 100,  # Rough approximation
        burn_pct=burn_pct * 100,
        high_severity_pct=high_pct * 100,
        moderate_severity_pct=moderate_pct * 100,
        low_severity_pct=low_pct * 100,
        unburned_pct=unburned_pct * 100,
        estimated_healing_years=None,  # Set by Layer 3 with RAG
        mean_severity=mean_sev,
        max_severity=max_sev,
        model_confidence=0.85,  # Default - override with actual value
        total_area_hectares=total_area_ha
    )


# =============================================================================
# PROFESSIONAL ANALYTICS FUNCTIONS
# =============================================================================

def compute_severity_weighted_centroid(
    severity_map: np.ndarray,
    bbox: Dict[str, float]
) -> Tuple[float, float]:
    """
    Compute the CENTER OF DAMAGE, not the center of the selection box.
    
    For restoration professionals: "Where is the core of the problem?"
    This weights the centroid by burn severity, so the returned location
    represents where damage is concentrated.
    
    Args:
        severity_map: 2D array (H, W) with severity values 0-1
        bbox: Bounding box dict with west, south, east, north
        
    Returns:
        (latitude, longitude) of severity-weighted centroid
    """
    h, w = severity_map.shape
    
    # Create coordinate grids
    lats = np.linspace(bbox['north'], bbox['south'], h)  # N to S
    lons = np.linspace(bbox['west'], bbox['east'], w)    # W to E
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Weight by severity (only areas with damage)
    weights = np.maximum(severity_map, 0)
    total_weight = np.sum(weights)
    
    if total_weight < 0.01:
        # No significant damage - fall back to geometric center
        center_lat = (bbox['north'] + bbox['south']) / 2
        center_lon = (bbox['east'] + bbox['west']) / 2
        return center_lat, center_lon
    
    # Compute severity-weighted centroid (center of mass)
    center_lat = float(np.sum(lat_grid * weights) / total_weight)
    center_lon = float(np.sum(lon_grid * weights) / total_weight)
    
    return center_lat, center_lon


def compute_coverage_analytics(
    severity_map: np.ndarray,
    bbox: Dict[str, float],
    pixel_size_m: float = 10.0,
    damage_threshold: float = 0.1,
    high_severity_threshold: float = 0.6
) -> Dict[str, float]:
    """
    Compute professional coverage metrics for restoration planning.
    
    For restoration managers: "What percentage of the selection is damaged?"
    
    Args:
        severity_map: 2D array (H, W) with severity values 0-1
        bbox: Bounding box dict
        pixel_size_m: Pixel resolution (default 10m for Sentinel-2)
        damage_threshold: Severity above which is considered damaged (0.1)
        high_severity_threshold: Severity above which is high severity (0.6)
        
    Returns:
        Dict with total_area_km2, damaged_area_km2, high_severity_area_km2, damage_coverage_pct
    """
    total_pixels = severity_map.size
    damaged_pixels = np.sum(severity_map > damage_threshold)
    high_damage_pixels = np.sum(severity_map > high_severity_threshold)
    
    # Convert pixel count to km²
    pixel_area_km2 = (pixel_size_m * pixel_size_m) / 1e6
    
    total_km2 = float(total_pixels * pixel_area_km2)
    damaged_km2 = float(damaged_pixels * pixel_area_km2)
    high_severity_km2 = float(high_damage_pixels * pixel_area_km2)
    
    coverage_pct = (damaged_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    
    return {
        'total_area_km2': round(total_km2, 3),
        'damaged_area_km2': round(damaged_km2, 3),
        'high_severity_area_km2': round(high_severity_km2, 3),
        'damage_coverage_pct': round(coverage_pct, 1)
    }


def reverse_geocode(lat: float, lon: float, retries: int = 2) -> Dict[str, Optional[str]]:
    """
    Dynamically get location info from coordinates using Nominatim API.
    
    NO HARDCODED LOCATIONS - all data comes from the coordinates.
    Uses OpenStreetMap's Nominatim service with ToS-compliant rate limiting.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        retries: Number of retry attempts on failure
        
    Returns:
        Dict with country, state, area_name (all may be None if lookup fails)
    """
    import requests
    import time
    global _last_geocode_time
    
    result = {
        'country': None,
        'state': None,
        'area_name': None
    }
    
    # Validate coordinates first
    try:
        validate_coordinates(lat, lon)
    except ValueError as e:
        logger.error(f"Invalid coordinates for reverse geocode: {e}")
        return result
    
    # Rate limiting - Nominatim ToS requires max 1 request/second
    time_since_last = time.time() - _last_geocode_time
    if time_since_last < NOMINATIM_RATE_LIMIT_SECONDS:
        time.sleep(NOMINATIM_RATE_LIMIT_SECONDS - time_since_last)
    
    for attempt in range(retries + 1):
        try:
            _last_geocode_time = time.time()
            
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 10
            }
            headers = {
                'User-Agent': 'EcoRevive/2.0 (Restoration Analysis Tool)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            address = data.get('address', {})
            
            result['country'] = address.get('country')
            result['state'] = address.get('state', address.get('province', address.get('region')))
            result['area_name'] = address.get('county', address.get('city', address.get('town', address.get('village'))))
            
            logger.info(f"Reverse geocode: {result['area_name']}, {result['state']}, {result['country']}")
            return result
            
        except requests.exceptions.Timeout:
            logger.warning(f"Reverse geocode timeout (attempt {attempt + 1}/{retries + 1})")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                logger.warning("Nominatim rate limited - backing off")
            else:
                logger.warning(f"Reverse geocode HTTP error: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Reverse geocode request failed: {e}")
                
        except Exception as e:
            logger.error(f"Reverse geocode unexpected error: {e}")
            break
        
        # Exponential backoff for retries
        if attempt < retries:
            backoff = (2 ** attempt) * NOMINATIM_RATE_LIMIT_SECONDS
            time.sleep(backoff)
    
    logger.warning(f"Reverse geocode failed after {retries + 1} attempts")
    return result


def extract_zones_from_severity(
    severity_map: np.ndarray,
    location: Tuple[float, float],
    pixel_size_m: float = 10.0,
    thresholds: Dict[str, float] = None,
    bbox: Optional[Dict[str, float]] = None
) -> List[ZonePrimitive]:
    """
    Extract discrete zones from severity map WITH BOUNDING BOXES.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        location: (lat, lon) center point
        pixel_size_m: Pixel size in meters
        thresholds: Custom thresholds
        bbox: Bounding box for coordinate conversion (west, south, east, north)
        
    Returns:
        List of ZonePrimitive objects with bbox and centroid coordinates
    """
    if thresholds is None:
        thresholds = {"high": 0.66, "moderate": 0.27, "low": 0.1}
    
    from scipy import ndimage
    
    zones = []
    h, w = severity_map.shape
    pixel_area_m2 = pixel_size_m ** 2
    lat, lon = location
    
    # Create coordinate grids for pixel -> geo conversion
    if bbox:
        lats = np.linspace(bbox['north'], bbox['south'], h)  # N to S
        lons = np.linspace(bbox['west'], bbox['east'], w)    # W to E
    
    # Create masks for each severity level
    zone_configs = [
        ("high_severity", severity_map > thresholds["high"], ZoneType.BURN_SCAR.value),
        ("moderate_severity", (severity_map > thresholds["moderate"]) & (severity_map <= thresholds["high"]), ZoneType.DEGRADED_VEGETATION.value),
        ("unburned", severity_map <= thresholds["low"], ZoneType.HEALTHY_VEGETATION.value),
    ]
    
    zone_counter = 0
    for name_prefix, mask, zone_type in zone_configs:
        if not np.any(mask):
            continue
            
        # Label connected components
        labeled, num_patches = ndimage.label(mask.astype(np.int32))
        
        for patch_id in range(1, min(num_patches + 1, MAX_PATCHES_PER_CLASS + 1)):
            patch_mask = labeled == patch_id
            patch_pixels = np.sum(patch_mask)
            
            if patch_pixels < MIN_PATCH_PIXELS:
                continue
            
            area_ha = (patch_pixels * pixel_area_m2) / 10000
            
            # Compute patch severity stats
            patch_severity = severity_map[patch_mask]
            sev_mean = float(np.mean(patch_severity))
            
            # Confidence based on size and homogeneity
            confidence = min(0.95, 0.5 + (patch_pixels / (h * w)) * 2)
            
            # Compute actual bounding box and centroid from pixel coordinates
            zone_bbox = None
            zone_centroid = (lon, lat)  # Fallback
            
            if bbox:
                # Find pixel bounds of this patch
                rows, cols = np.where(patch_mask)
                if len(rows) > 0:
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Convert pixel bounds to geographic coordinates
                    zone_bbox = {
                        'north': float(lats[min_row]),
                        'south': float(lats[max_row]),
                        'west': float(lons[min_col]),
                        'east': float(lons[max_col])
                    }
                    
                    # Compute severity-weighted centroid for this zone
                    zone_severity = severity_map * patch_mask
                    total_weight = np.sum(zone_severity)
                    if total_weight > 0:
                        lon_grid, lat_grid = np.meshgrid(lons, lats)
                        centroid_lat = float(np.sum(lat_grid * zone_severity) / total_weight)
                        centroid_lon = float(np.sum(lon_grid * zone_severity) / total_weight)
                        zone_centroid = (centroid_lon, centroid_lat)  # GeoJSON order: (lon, lat)
            
            zones.append(ZonePrimitive(
                zone_id=f"zone_{zone_counter:03d}",
                zone_type=zone_type,
                geometry_type="polygon",
                bbox=zone_bbox,
                centroid=zone_centroid,
                area_hectares=area_ha,
                confidence=confidence,
                uncertainty_radius_m=15.0,
                severity_mean=sev_mean
            ))
            zone_counter += 1
    
    return zones


def generate_risk_grid(
    severity_map: np.ndarray,
    location: Tuple[float, float],
    grid_size: int = 8
) -> RiskGrid:
    """
    Generate a simple risk grid from severity map.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        location: (lat, lon) center point
        grid_size: Number of cells per dimension
        
    Returns:
        RiskGrid object
    """
    h, w = severity_map.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * cell_h
            y_end = min((row + 1) * cell_h, h)
            x_start = col * cell_w
            x_end = min((col + 1) * cell_w, w)
            
            cell_data = severity_map[y_start:y_end, x_start:x_end]
            cell_mean = float(np.mean(cell_data))
            
            # Determine risk level
            if cell_mean > 0.75:
                risk_level = RiskLevel.EXTREME.value
            elif cell_mean > 0.5:
                risk_level = RiskLevel.HIGH.value
            elif cell_mean > 0.25:
                risk_level = RiskLevel.MODERATE.value
            else:
                risk_level = RiskLevel.LOW.value
            
            factors = []
            if cell_mean > 0.6:
                factors.append("high_burn_severity")
            if cell_mean > 0.8:
                factors.append("potential_erosion")
            
            cells.append(RiskCell(
                cell_id=f"r{row}_c{col}",
                risk_level=risk_level,
                risk_score=cell_mean,
                contributing_factors=factors
            ))
    
    return RiskGrid(
        resolution=grid_size,
        grid_type="simple_grid",
        cells=cells
    )


# NOTE: extract_hazards_from_severity() was REMOVED 
# Reason: Hazard detection (dead trees, erosion) cannot be reliably determined
# from burn severity values alone. This requires RGB imagery analysis via
# Gemini vision in Layer 3. The previous implementation produced misleading
# confidence scores for hazards that could not actually be detected.


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_layer2_analysis(
    severity_map: np.ndarray,
    location: Tuple[float, float],
    bbox: Optional[Dict[str, float]] = None,
    gemini_analysis: Optional[Dict[str, Any]] = None,
    model_confidence: float = 0.85,
    pixel_size_m: float = 10.0,
    imagery_date: Optional[str] = None
) -> Layer2Output:
    """
    Run complete Layer 2 analysis.
    
    This is the main entry point for generating structured Layer 2 output.
    It combines severity map analysis with optional Gemini spatial reasoning.
    
    Args:
        severity_map: 2D numpy array with burn severity predictions (0-1)
        location: (latitude, longitude) center point - NOTE: (lat, lon) order!
        bbox: Optional bounding box dict with keys: west, south, east, north
        gemini_analysis: Optional parsed Gemini multimodal analysis
        model_confidence: U-Net model confidence score (0-1)
        pixel_size_m: Pixel size in meters (default 10m for Sentinel-2)
        imagery_date: Date of satellite imagery (ISO format)
        
    Returns:
        Layer2Output with complete structured data
        
    Raises:
        ValueError: If inputs fail validation
    """
    import time
    start_time = time.time()
    
    # --- Input Validation ---
    validate_severity_map(severity_map)
    
    lat, lon = location
    validate_coordinates(lat, lon)
    
    if bbox is not None:
        validate_bbox(bbox)
    
    # Handle out-of-range severity values
    # Small floating-point errors (±0.01) are acceptable and clipped
    # Larger deviations indicate data issues and should be raised
    min_val, max_val = float(severity_map.min()), float(severity_map.max())
    TOLERANCE = 0.01
    
    if min_val < -TOLERANCE or max_val > 1 + TOLERANCE:
        raise ValueError(
            f"Severity map values significantly out of range: "
            f"min={min_val:.4f}, max={max_val:.4f}. Expected [0, 1]."
        )
    
    # Clip minor floating-point errors silently
    if min_val < 0 or max_val > 1:
        severity_map = np.clip(severity_map, 0, 1)
    
    # --- Create Base Output ---
    output = Layer2Output.create_empty(lat, lon)
    output.schema_version = SCHEMA_VERSION  # Add schema version
    
    # Update location
    output.location.bounding_box = bbox
    if bbox:
        # Dynamic reverse geocoding - no hardcoded locations!
        geo_info = reverse_geocode(lat, lon)
        output.location.country = geo_info.get('country')
        output.location.state = geo_info.get('state')
        output.location.area_name = geo_info.get('area_name')
        
        # Compute severity-weighted damage centroid (professional analytics)
        damage_lat, damage_lon = compute_severity_weighted_centroid(severity_map, bbox)
        output.location.damage_centroid_lat = damage_lat
        output.location.damage_centroid_lon = damage_lon
        output.location.centroid_type = "severity_weighted"
    
    # Compute metrics from severity map
    output.metrics = compute_metrics_from_severity(severity_map, pixel_size_m)
    output.metrics.model_confidence = model_confidence
    
    # Add coverage analytics (professional restoration metrics)
    if bbox:
        coverage = compute_coverage_analytics(severity_map, bbox, pixel_size_m)
        output.metrics.total_area_km2 = coverage['total_area_km2']
        output.metrics.damaged_area_km2 = coverage['damaged_area_km2']
        output.metrics.high_severity_area_km2 = coverage['high_severity_area_km2']
        output.metrics.damage_coverage_pct = coverage['damage_coverage_pct']
    
    # Extract spatial zones
    output.zones = extract_zones_from_severity(severity_map, location, pixel_size_m, bbox=bbox)
    
    # Generate risk grid
    output.risk_grid = generate_risk_grid(severity_map, location)
    
    # NOTE: Hazard extraction removed - requires RGB imagery analysis in Layer 3
    # output.hazards remains empty list (default) - Layer 3 should populate via Gemini vision

    
    # If Gemini analysis provided, enhance output
    if gemini_analysis:
        _enhance_from_gemini(output, gemini_analysis)
    
    # Compute signals for Layer 3
    output.signals = {
        "restoration_potential_score": max(0, 1 - output.metrics.mean_severity),
        "intervention_urgency_score": min(1, output.metrics.high_severity_pct / 50),
        "risk_score": output.metrics.mean_severity,
        "confidence_weighted_severity": output.metrics.mean_severity * model_confidence,
        "ecological_complexity_score": min(1, len(output.zones) / 10)
    }
    
    # Update confidence
    output.confidence.overall_confidence = model_confidence
    output.confidence.segmentation_quality = (
        "good" if model_confidence > 0.8 else
        "acceptable" if model_confidence > 0.6 else "poor"
    )
    output.confidence.classification_reliability = model_confidence * 0.9
    
    # Check if human review needed
    if model_confidence < 0.6 or output.metrics.high_severity_pct > 60:
        output.confidence.requires_human_review = True
        if model_confidence < 0.6:
            output.confidence.review_triggers.append("low_model_confidence")
        if output.metrics.high_severity_pct > 60:
            output.confidence.review_triggers.append("high_severity_area")
    
    # Update processing metadata
    processing_time = int((time.time() - start_time) * 1000)
    output.processing.processing_time_ms = processing_time
    output.processing.imagery_date = imagery_date
    
    return output


def _enhance_from_gemini(output: Layer2Output, gemini_analysis: Dict[str, Any]):
    """
    Enhance Layer2Output with Gemini multimodal analysis results.
    
    Supports both:
    1. New Layer 2 structured output (analyze_for_layer2)
    2. Legacy multimodal output (analyze)
    
    Args:
        output: Layer2Output to enhance
        gemini_analysis: Parsed Gemini response (from layer2_data or raw)
    """
    # Check if this is new Layer 2 structured format
    if 'layer2_data' in gemini_analysis:
        data = gemini_analysis['layer2_data']
        _apply_layer2_structured_data(output, data)
        return
    
    # Check if this came from _transform_to_layer2_schema
    if 'characteristics' in gemini_analysis and 'ecosystem' in gemini_analysis:
        _apply_layer2_structured_data(output, gemini_analysis)
        return
    
    # Legacy format - handle old schema
    _apply_legacy_gemini_data(output, gemini_analysis)


def _apply_layer2_structured_data(output: Layer2Output, data: Dict[str, Any]):
    """Apply new Layer 2 structured Gemini data to output."""
    
    # Site Characteristics
    chars = data.get('characteristics', {})
    if chars:
        output.characteristics.soil_type = chars.get('soil_type', output.characteristics.soil_type)
        output.characteristics.land_use_history = chars.get('land_use_history')
        if output.characteristics.terrain is None:
            output.characteristics.terrain = TerrainData()
        # Map slope category to degrees
        slope_map = {'flat': 2, 'gentle': 8, 'moderate': 18, 'steep': 35}
        slope_cat = chars.get('slope_category', 'unknown')
        if slope_cat in slope_map:
            output.characteristics.terrain.slope_degrees_mean = slope_map[slope_cat]
        output.characteristics.terrain.drainage_pattern = chars.get('drainage_pattern')
    
    # Ecosystem Info
    eco = data.get('ecosystem', {})
    if eco:
        output.ecosystem.current_state = eco.get('current_state', output.ecosystem.current_state)
        output.ecosystem.pre_degradation_ecosystem = eco.get('pre_degradation_ecosystem')
        output.ecosystem.vegetation_types = eco.get('vegetation_types', [])
        output.ecosystem.key_species = eco.get('key_species', [])
        output.ecosystem.succession_stage = eco.get('succession_stage', 'disturbed')
    
    # Visual zones from Gemini (merge with computed zones)
    visual_zones = data.get('visual_zones', [])
    for vz in visual_zones:
        # Add Gemini-detected zones as additional zones
        output.zones.append(ZonePrimitive(
            zone_id=f"gemini_{vz.get('zone_id', 'unknown')}",
            zone_type=vz.get('zone_type', 'uncertain'),
            area_hectares=vz.get('area_estimate_pct', 0) * output.metrics.total_area_hectares / 100,
            confidence=vz.get('confidence', 0.5),
            severity_mean=_severity_to_float(vz.get('severity_category', 'moderate'))
        ))
    
    # Visual hazards from Gemini (merge with computed hazards)
    visual_hazards = data.get('visual_hazards', [])
    for i, vh in enumerate(visual_hazards):
        output.hazards.append(HazardAnnotation(
            hazard_id=f"gemini_haz_{i:03d}",
            hazard_type=vh.get('hazard_type', 'other'),
            severity=vh.get('severity', 'moderate'),
            detection_confidence=vh.get('detection_confidence', 0.5),
            description=vh.get('location_description')
        ))
    
    # Quality assessment
    quality = data.get('quality', {})
    if quality:
        output.confidence.segmentation_quality = quality.get('segmentation_quality', 'unknown')
        gemini_conf = quality.get('confidence_in_prediction', 0.5)
        # Blend model confidence with Gemini confidence
        output.confidence.overall_confidence = (
            output.confidence.overall_confidence * 0.6 + gemini_conf * 0.4
        )
        output.confidence.data_quality_flags = quality.get('artifact_flags', [])
        output.confidence.requires_human_review = quality.get('requires_human_review', False)
        output.confidence.review_triggers = quality.get('review_triggers', [])
    
    # Signals from Gemini
    gemini_signals = data.get('signals', {})
    for key, value in gemini_signals.items():
        if isinstance(value, (int, float)):
            output.signals[f"gemini_{key}"] = value


def _apply_legacy_gemini_data(output: Layer2Output, gemini_analysis: Dict[str, Any]):
    """Apply legacy multimodal Gemini data to output."""
    
    # Visual grounding → Ecosystem info
    vg = gemini_analysis.get("visual_grounding", {})
    if vg:
        output.ecosystem.vegetation_types = vg.get("observed_land_cover", [])
        if vg.get("pre_fire_vegetation_description"):
            output.characteristics.land_use_history = vg["pre_fire_vegetation_description"]
    
    # Ecological interpretation → Ecosystem info
    ei = gemini_analysis.get("ecological_interpretation", {})
    if ei:
        regen = ei.get("natural_regeneration_potential", "unknown")
        if regen == "high":
            output.ecosystem.succession_stage = "early"
        elif regen == "low" or regen == "very_low":
            output.ecosystem.succession_stage = "disturbed"
    
    # Segmentation quality → Confidence
    sq = gemini_analysis.get("segmentation_quality", {})
    if sq:
        output.confidence.segmentation_quality = sq.get("overall_quality", "unknown")
        gemini_conf = sq.get("confidence_in_prediction", 0)
        output.confidence.overall_confidence = (
            output.confidence.overall_confidence * 0.6 + gemini_conf * 0.4
        )
        if sq.get("artifact_flags"):
            output.confidence.data_quality_flags.extend(sq["artifact_flags"])
    
    # Signals from Gemini
    signals = gemini_analysis.get("signals_for_final_model", {})
    if signals:
        for key, value in signals.items():
            if key not in output.signals:
                output.signals[key] = value


def _severity_to_float(severity: str) -> float:
    """Convert severity category to float value."""
    mapping = {'high': 0.8, 'moderate': 0.5, 'low': 0.2, 'none': 0.0}
    return mapping.get(severity.lower(), 0.5)


def _map_severity_to_zone_type(severity: str) -> str:
    """Map severity string to ZoneType."""
    mapping = {
        "high": ZoneType.BURN_SCAR.value,
        "moderate": ZoneType.DEGRADED_VEGETATION.value,
        "low": ZoneType.HEALTHY_VEGETATION.value
    }
    return mapping.get(severity.lower(), ZoneType.UNCERTAIN.value)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_layer2_response(
    severity_map: np.ndarray,
    location: Tuple[float, float],
    bbox: Optional[Dict[str, float]] = None,
    gemini_analysis: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function that returns Layer2Output as a dict.
    
    This is the function to call from the server endpoint.
    """
    output = run_layer2_analysis(
        severity_map=severity_map,
        location=location,
        bbox=bbox,
        gemini_analysis=gemini_analysis,
        **kwargs
    )
    return output.to_dict()
