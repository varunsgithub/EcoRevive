"""
EcoRevive Layer 3 Context Analysis Module
==========================================
Provides land use classification and contextual warnings for burn severity analysis.

Layer 3 responsibilities:
- Land use classification (urban, suburban, rural, forest, etc.)
- Contextual warnings for areas where model may be less reliable
- Data quality assessment and user guidance

This layer helps users understand:
1. What type of area they're analyzing
2. Whether the burn severity model is appropriate for this area
3. Cautions and limitations to be aware of
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class LandUseType(str, Enum):
    """Types of land use detected in imagery."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    AGRICULTURAL = "agricultural"
    FOREST = "forest"
    GRASSLAND = "grassland"
    SHRUBLAND = "shrubland"
    WETLAND = "wetland"
    WATER = "water"
    BARREN = "barren"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CautionLevel(str, Enum):
    """Caution levels for analysis reliability."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# Model reliability notes by land use type
LAND_USE_CAUTIONS = {
    LandUseType.URBAN: {
        "level": CautionLevel.HIGH,
        "message": "This appears to be an urban/developed area. The burn severity model was trained on wildland fires and may not accurately assess urban fire damage. Building materials, pavement, and infrastructure can produce false positives.",
        "recommendations": [
            "Consider using structural damage assessment tools instead",
            "Verify results with on-ground inspection",
            "Be aware that impervious surfaces may appear as burn scars"
        ]
    },
    LandUseType.SUBURBAN: {
        "level": CautionLevel.MODERATE,
        "message": "This area contains mixed urban and vegetated land. The burn severity model may have reduced accuracy in developed portions. Results are more reliable for vegetated areas.",
        "recommendations": [
            "Focus analysis on vegetated portions",
            "Verify urban-adjacent areas with ground truth",
            "Consider separate assessment for structures"
        ]
    },
    LandUseType.AGRICULTURAL: {
        "level": CautionLevel.LOW,
        "message": "This appears to be agricultural land. Harvested fields or fallow land may appear similar to low-severity burns. Results are most reliable for permanent vegetation.",
        "recommendations": [
            "Check imagery date relative to harvest season",
            "Bare soil may be misclassified as burned",
            "Focus on perennial vegetation for best accuracy"
        ]
    },
    LandUseType.WATER: {
        "level": CautionLevel.MODERATE,
        "message": "This area contains significant water bodies. Water surfaces are excluded from burn analysis but shoreline vegetation assessment should be reliable.",
        "recommendations": [
            "Focus on riparian and shoreline vegetation",
            "Water pixels are masked from severity calculation"
        ]
    },
    LandUseType.FOREST: {
        "level": CautionLevel.NONE,
        "message": "This is forested land - the model's primary training environment. Burn severity predictions should be highly reliable.",
        "recommendations": []
    },
    LandUseType.SHRUBLAND: {
        "level": CautionLevel.NONE,
        "message": "This is shrubland/chaparral - well represented in model training data. Burn severity predictions should be reliable.",
        "recommendations": []
    },
    LandUseType.GRASSLAND: {
        "level": CautionLevel.LOW,
        "message": "This is grassland. The model handles grass fires well, though rapid regrowth may affect post-fire assessment timing.",
        "recommendations": [
            "Grass recovers quickly - timing of imagery matters",
            "Best assessed within weeks of fire event"
        ]
    },
    LandUseType.WETLAND: {
        "level": CautionLevel.LOW,
        "message": "This area contains wetland vegetation. Water saturation may affect spectral signatures but vegetated areas should be reliably assessed.",
        "recommendations": []
    },
    LandUseType.BARREN: {
        "level": CautionLevel.MODERATE,
        "message": "This area is predominantly barren/rocky terrain. Limited vegetation means burn severity assessment has reduced meaning.",
        "recommendations": [
            "Minimal vegetation to assess",
            "Focus on vegetated patches within the area"
        ]
    },
    LandUseType.MIXED: {
        "level": CautionLevel.LOW,
        "message": "This area has mixed land cover. Results are most reliable for vegetated portions.",
        "recommendations": []
    },
    LandUseType.UNKNOWN: {
        "level": CautionLevel.MODERATE,
        "message": "Unable to classify land use type. Please verify results with additional context.",
        "recommendations": []
    }
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class LandUseContext:
    """Land use classification and contextual information."""
    land_use_type: str  # LandUseType value
    land_use_description: str
    is_urban: bool
    urban_percentage: float  # 0-100
    vegetation_percentage: float  # 0-100

    # Caution information
    caution_level: str  # CautionLevel value
    caution_message: str
    recommendations: List[str]

    # Confidence
    classification_confidence: float  # 0-1

    # Additional context
    detected_features: List[str]  # e.g., ["buildings", "roads", "parking lots"]
    suitable_for_analysis: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Layer3Output:
    """Complete Layer 3 contextual analysis output with RAG context."""
    land_use: LandUseContext
    analysis_suitable: bool
    overall_caution_level: str
    user_guidance: str
    # RAG-retrieved context
    ecological_context: Optional[str] = None
    legal_context: Optional[str] = None
    ecoregion_info: Optional[List[Dict[str, Any]]] = None
    species_recommendations: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "land_use": self.land_use.to_dict(),
            "analysis_suitable": self.analysis_suitable,
            "overall_caution_level": self.overall_caution_level,
            "user_guidance": self.user_guidance
        }
        # Include RAG context if available
        if self.ecological_context:
            result["ecological_context"] = self.ecological_context
        if self.legal_context:
            result["legal_context"] = self.legal_context
        if self.ecoregion_info:
            result["ecoregion_info"] = self.ecoregion_info
        if self.species_recommendations:
            result["species_recommendations"] = self.species_recommendations
        return result


# =============================================================================
# GEMINI LAND USE CLASSIFICATION
# =============================================================================

def classify_land_use_with_gemini(
    client,
    rgb_image: np.ndarray,
    location: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Use Gemini to classify land use type from satellite imagery.

    Args:
        client: Gemini client instance
        rgb_image: RGB satellite image as numpy array (H, W, 3) or (3, H, W)
        location: (latitude, longitude) tuple

    Returns:
        Dict with land use classification results
    """
    # Convert numpy array to PIL Image
    if rgb_image.ndim == 3:
        if rgb_image.shape[0] == 3:  # (3, H, W) format
            rgb_image = np.moveaxis(rgb_image, 0, -1)
        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
    else:
        raise ValueError(f"Expected 3D RGB array, got shape {rgb_image.shape}")

    lat, lon = location

    prompt = f"""Analyze this satellite image and classify the land use type.

Location: {abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}

Examine the image carefully and identify:
1. The PRIMARY land use type
2. Approximate percentage of urban/developed features (buildings, roads, parking lots)
3. Approximate percentage of vegetation cover
4. Specific features you can see

Respond in this exact JSON format:
{{
    "land_use_type": "<one of: urban, suburban, rural, agricultural, forest, grassland, shrubland, wetland, water, barren, mixed>",
    "land_use_description": "<brief 1-2 sentence description of what you see>",
    "urban_percentage": <0-100 number>,
    "vegetation_percentage": <0-100 number>,
    "detected_features": ["<feature1>", "<feature2>", ...],
    "classification_confidence": <0.0-1.0>,
    "reasoning": "<brief explanation of your classification>"
}}

Be conservative with urban classification - only classify as "urban" if you see clear evidence of dense development (buildings, extensive pavement, grid patterns). Suburban should have mixed development and vegetation. Rural areas have sparse development.

Focus on what you can actually see in the satellite imagery."""

    try:
        response = client.analyze_multimodal(
            prompt=prompt,
            images=[pil_image],
            use_json=True
        )

        if response.get('json'):
            return response['json']
        elif response.get('text'):
            # Try to parse JSON from text
            import json
            text = response['text']
            # Find JSON in the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])

        logger.warning("Gemini did not return valid JSON for land use classification")
        return None

    except Exception as e:
        logger.error(f"Land use classification failed: {e}")
        return None


def classify_land_use_heuristic(
    severity_map: np.ndarray,
    location: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Heuristic-based land use estimation when Gemini is unavailable.

    Uses severity map patterns and location to make educated guesses.
    This is a fallback - Gemini classification is preferred.
    """
    # Basic heuristics based on severity patterns
    mean_severity = float(np.mean(severity_map))
    std_severity = float(np.std(severity_map))
    high_ratio = float(np.mean(severity_map > 0.66))

    # Very uniform high values might indicate urban (false positives from pavement)
    # Natural fires typically have more variation

    # If severity is very uniform and high, might be urban false positive
    if std_severity < 0.1 and mean_severity > 0.6:
        return {
            "land_use_type": "unknown",
            "land_use_description": "Uniform severity pattern detected - may indicate developed area or sensor artifacts",
            "urban_percentage": 50,  # Unknown, assume moderate risk
            "vegetation_percentage": 50,
            "detected_features": ["uniform_pattern"],
            "classification_confidence": 0.3,
            "reasoning": "Heuristic fallback - uniform severity patterns may indicate non-natural land cover"
        }

    # Default assumption: natural vegetation (model's target domain)
    return {
        "land_use_type": "mixed",
        "land_use_description": "Mixed vegetation cover - classification requires visual analysis",
        "urban_percentage": 10,  # Assume mostly natural
        "vegetation_percentage": 80,
        "detected_features": [],
        "classification_confidence": 0.4,
        "reasoning": "Heuristic fallback - assuming natural vegetation (model's target domain)"
    }


# =============================================================================
# MAIN LAYER 3 ANALYSIS
# =============================================================================

def run_layer3_analysis(
    client,
    rgb_image: np.ndarray,
    severity_map: np.ndarray,
    location: Tuple[float, float],
    use_gemini: bool = True,
    use_rag: bool = True
) -> Layer3Output:
    """
    Run Layer 3 contextual analysis.

    This analyzes the land use type and provides appropriate warnings
    to help users understand model reliability for their selected area.

    Args:
        client: Gemini client instance (can be None if use_gemini=False)
        rgb_image: RGB satellite image
        severity_map: Burn severity predictions from Layer 1
        location: (latitude, longitude) center point
        use_gemini: Whether to use Gemini for classification

    Returns:
        Layer3Output with land use context and cautions
    """
    # Unpack location
    lat, lon = location

    # Step 1: Classify land use
    if use_gemini and client is not None:
        classification = classify_land_use_with_gemini(client, rgb_image, location)
        if classification is None:
            classification = classify_land_use_heuristic(severity_map, location)
    else:
        classification = classify_land_use_heuristic(severity_map, location)

    # Step 2: Map classification to LandUseType
    land_use_str = classification.get('land_use_type', 'unknown').lower()
    try:
        land_use_type = LandUseType(land_use_str)
    except ValueError:
        land_use_type = LandUseType.UNKNOWN

    # Step 3: Get caution information for this land use type
    caution_info = LAND_USE_CAUTIONS.get(land_use_type, LAND_USE_CAUTIONS[LandUseType.UNKNOWN])

    # Step 4: Determine if area is urban
    urban_pct = float(classification.get('urban_percentage', 0))
    veg_pct = float(classification.get('vegetation_percentage', 100 - urban_pct))
    conf = float(classification.get('classification_confidence', 0.5))
    is_urban = bool(urban_pct > 30 or land_use_type in [LandUseType.URBAN, LandUseType.SUBURBAN])
    is_suitable = bool(caution_info['level'] in [CautionLevel.NONE, CautionLevel.LOW])

    # Step 5: Build land use context
    land_use_context = LandUseContext(
        land_use_type=land_use_type.value,
        land_use_description=classification.get('land_use_description', ''),
        is_urban=is_urban,
        urban_percentage=urban_pct,
        vegetation_percentage=veg_pct,
        caution_level=caution_info['level'].value,
        caution_message=caution_info['message'],
        recommendations=caution_info['recommendations'],
        classification_confidence=conf,
        detected_features=classification.get('detected_features', []),
        suitable_for_analysis=is_suitable
    )

    # Step 6: Generate user guidance
    if land_use_context.suitable_for_analysis:
        user_guidance = "This area is well-suited for burn severity analysis. Results should be reliable."
    elif land_use_context.caution_level == CautionLevel.HIGH.value:
        user_guidance = (
            "CAUTION: This area may not be suitable for wildland fire analysis. "
            "The burn severity model was trained on natural vegetation and may produce "
            "unreliable results for developed/urban areas. Please verify results carefully."
        )
    else:
        user_guidance = (
            "Some caution advised. This area has characteristics that may affect model accuracy. "
            "Results are usable but should be verified for the specific conditions present."
        )

    # Step 7: RAG Retrieval (if enabled)
    ecological_context = None
    legal_context = None
    ecoregion_info = None
    species_recommendations = None
    
    if use_rag:
        try:
            from .rag.ecology_rag import CombinedRAG
            
            lat_dir = 'N' if lat >= 0 else 'S'
            lon_dir = 'E' if lon >= 0 else 'W'
            location_desc = f"Site at {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}, land use: {land_use_type.value}"
            
            # Determine severity level
            mean_severity = float(np.mean(severity_map))
            if mean_severity > 0.6:
                severity_level = "high"
            elif mean_severity > 0.3:
                severity_level = "moderate"
            else:
                severity_level = "low"
            
            # Initialize RAG and retrieve context
            rag = CombinedRAG()
            rag.initialize()
            
            # Get ecological context
            ecological_context = rag.ecology_rag.get_restoration_context(
                location_description=location_desc,
                severity_level=severity_level,
                k=3
            )
            
            # Get legal context
            legal_context = rag.legal_rag.get_legal_context(
                location_description=location_desc,
                activity_type="restoration"
            )
            
            # Get specific ecoregion info
            ecoregion_info = rag.ecology_rag.get_ecoregion_info(
                query=location_desc,
                k=2
            )
            
            # Get species recommendations if suitable for analysis
            if is_suitable:
                species_recommendations = rag.ecology_rag.get_species_recommendations(
                    query=f"{severity_level} severity burn site in {location_desc}",
                    k=5
                )
            
            logger.info(f"Layer 3 RAG retrieval complete: {len(ecoregion_info or [])} ecoregions, {len(species_recommendations or [])} species")
            
        except Exception as e:
            logger.warning(f"Layer 3 RAG retrieval failed: {e}")
            # Continue without RAG context
    
    # Step 8: Build output
    return Layer3Output(
        land_use=land_use_context,
        analysis_suitable=land_use_context.suitable_for_analysis,
        overall_caution_level=land_use_context.caution_level,
        user_guidance=user_guidance,
        ecological_context=ecological_context,
        legal_context=legal_context,
        ecoregion_info=ecoregion_info,
        species_recommendations=species_recommendations
    )


def create_layer3_response(
    client,
    rgb_image: np.ndarray,
    severity_map: np.ndarray,
    location: Tuple[float, float],
    use_gemini: bool = True,
    use_rag: bool = True
) -> Dict[str, Any]:
    """
    Convenience function that returns Layer3Output as a dict.

    This is the function to call from the server endpoint.
    Includes RAG-retrieved ecological and legal context when available.
    """
    output = run_layer3_analysis(
        client=client,
        rgb_image=rgb_image,
        severity_map=severity_map,
        location=location,
        use_gemini=use_gemini,
        use_rag=use_rag
    )
    return output.to_dict()

