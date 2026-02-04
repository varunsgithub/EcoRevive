"""
EcoRevive Ecosystem Classification Module
==========================================
Uses Gemini's multimodal capabilities to:
1. Analyze burn severity maps WITH original satellite imagery (true multimodal)
2. Classify ecosystem type from location + visual patterns
3. Recommend native species with Google Search grounding
4. Determine restoration approach based on spatial reasoning

This module showcases:
- TRUE multimodal input (RGB image + severity overlay + structured context)
- Spatial pattern analysis (fragmentation, edges, gradients)
- Structured JSON output with machine-readable signals
- Grounding with Google Search for species data

IMPORTANT: For true multimodal analysis, use classify_multimodal() which requires
the original RGB tile. The legacy classify() method is preserved for backward
compatibility but only sends the severity colormap without landscape context.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

try:
    import PIL.Image
except ImportError:
    PIL = None

from .gemini_client import EcoReviveGemini, create_client
from .gemini_multimodal import (
    MultimodalAnalyzer,
    create_image_pack,
    build_gemini_context,
    build_multimodal_prompt,
    validate_gemini_output,
    should_trigger_human_review,
    compute_severity_statistics,
    compute_spatial_metrics,
)


# California Ecoregion reference data
CALIFORNIA_ECOREGIONS = {
    "sierra_nevada": {
        "name": "Sierra Nevada",
        "bbox": [-121.5, 35.5, -118.0, 41.0],
        "vegetation": ["Mixed Conifer", "Red Fir", "Subalpine"],
        "key_species": ["Ponderosa Pine", "White Fir", "Black Oak", "Manzanita"]
    },
    "coast_ranges": {
        "name": "California Coast Ranges",
        "bbox": [-124.0, 34.5, -121.0, 40.5],
        "vegetation": ["Coast Redwood", "Mixed Evergreen", "Oak Woodland"],
        "key_species": ["Coast Redwood", "Douglas Fir", "Tan Oak", "California Bay"]
    },
    "klamath": {
        "name": "Klamath Mountains",
        "bbox": [-124.5, 40.5, -122.0, 42.5],
        "vegetation": ["Mixed Conifer", "Mixed Evergreen"],
        "key_species": ["Douglas Fir", "Tanoak", "Pacific Madrone", "Canyon Live Oak"]
    },
    "modoc": {
        "name": "Modoc Plateau",
        "bbox": [-121.5, 41.0, -120.0, 42.0],
        "vegetation": ["Juniper Woodland", "Sagebrush Steppe"],
        "key_species": ["Western Juniper", "Sagebrush", "Bitterbrush"]
    },
    "central_valley": {
        "name": "Central Valley",
        "bbox": [-122.5, 35.0, -119.0, 40.0],
        "vegetation": ["Valley Oak Savanna", "Riparian Forest"],
        "key_species": ["Valley Oak", "Blue Oak", "Cottonwood", "Willow"]
    }
}


def get_ecoregion_from_location(lat: float, lon: float) -> Optional[str]:
    """
    Determine the California ecoregion based on coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Ecoregion key or None if not in California
    """
    for key, region in CALIFORNIA_ECOREGIONS.items():
        west, south, east, north = region["bbox"]
        if west <= lon <= east and south <= lat <= north:
            return key
    return None


def severity_map_to_image(
    severity_map: np.ndarray,
    colormap: str = "hot"
) -> 'PIL.Image.Image':
    """
    Convert a numpy severity map (0-1) to a PIL Image for Gemini analysis.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        colormap: Matplotlib colormap name
        
    Returns:
        PIL Image with colorized severity
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Normalize to 0-1 if needed
    if severity_map.max() > 1:
        severity_map = severity_map / severity_map.max()
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(severity_map)
    
    # Convert to 8-bit RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return PIL.Image.fromarray(rgb)


def compute_severity_stats(severity_map: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics from a burn severity map.
    
    Args:
        severity_map: 2D numpy array with values 0-1
        
    Returns:
        Dict with severity statistics
    """
    return {
        "mean_severity": float(np.mean(severity_map)),
        "max_severity": float(np.max(severity_map)),
        "min_severity": float(np.min(severity_map)),
        "std_severity": float(np.std(severity_map)),
        "high_severity_ratio": float(np.mean(severity_map > 0.66)),
        "moderate_severity_ratio": float(np.mean((severity_map > 0.27) & (severity_map <= 0.66))),
        "low_severity_ratio": float(np.mean((severity_map > 0.1) & (severity_map <= 0.27))),
        "unburned_ratio": float(np.mean(severity_map <= 0.1)),
    }


ECOSYSTEM_PROMPT = """You are an expert restoration ecologist specializing in California ecosystems.

You are analyzing a burn severity map from a wildfire site. The map shows:
- RED/WHITE areas = High severity burn (tree mortality, soil damage)
- ORANGE/YELLOW = Moderate burn
- DARK/BLACK = Low or no burn

LOCATION: {lat_display}
ECOREGION HINT: {ecoregion}

BURN STATISTICS:
{stats}

Based on the visual pattern of the burn severity map, the location, and the statistics:

1. ECOSYSTEM CLASSIFICATION: What type of ecosystem was this before the fire? Consider elevation, latitude, and typical California vegetation patterns.

2. REFERENCE ECOSYSTEM: What should this area look like when fully recovered? Describe the target state.

3. SPECIES RECOMMENDATIONS: List 5-8 native species appropriate for restoration, with:
   - Common name and scientific name
   - Role in ecosystem (canopy, understory, ground cover, etc.)
   - Recommended planting density (trees/hectare or kg seed/hectare)
   - Priority (1=essential, 2=important, 3=beneficial)

4. RESTORATION APPROACH: Given the burn severity pattern, recommend:
   - Active reforestation OR natural regeneration OR combination
   - Key priority zones based on the severity pattern
   - Estimated timeline for canopy recovery

5. SPECIAL CONSIDERATIONS: Any concerns about:
   - Invasive species risk (post-fire invasion is common)
   - Erosion control needs
   - Wildlife corridor preservation

Return your analysis as structured JSON matching this schema:
{{
  "ecosystem_type": "string",
  "reference_ecosystem": "string",
  "climate_zone": "string",
  "elevation_estimate_m": number,
  "species_palette": [
    {{
      "common_name": "string",
      "scientific_name": "string",
      "role": "string",
      "density_per_hectare": number,
      "density_unit": "trees" | "kg_seed",
      "priority": 1 | 2 | 3,
      "notes": "string"
    }}
  ],
  "restoration_method": "active_reforestation" | "natural_regeneration" | "combination",
  "restoration_rationale": "string",
  "priority_zones": [
    {{
      "zone": "string",
      "action": "string",
      "urgency": "immediate" | "year_1" | "year_2_3"
    }}
  ],
  "recovery_timeline": {{
    "ground_cover_years": number,
    "shrub_establishment_years": number,
    "canopy_closure_years": number
  }},
  "special_considerations": {{
    "invasive_species_risk": "low" | "moderate" | "high",
    "invasive_species_to_watch": ["string"],
    "erosion_risk": "low" | "moderate" | "high",
    "erosion_mitigation": "string",
    "wildlife_notes": "string"
  }},
  "confidence": "low" | "medium" | "high",
  "reasoning": "string"
}}
"""


class EcosystemClassifier:
    """
    Gemini-powered ecosystem classification for restoration planning.
    
    Uses multimodal analysis to understand burn patterns and
    recommend appropriate restoration strategies.
    
    Two analysis modes are available:
    - classify_multimodal(): TRUE multimodal with RGB + severity overlay (RECOMMENDED)
    - classify(): Legacy mode with severity colormap only (preserved for backward compatibility)
    """
    
    def __init__(self, gemini_client: Optional[EcoReviveGemini] = None):
        """
        Initialize the classifier.
        
        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self.client = gemini_client or create_client()
        self._multimodal_analyzer = None  # Lazy initialization
    
    @property
    def multimodal_analyzer(self) -> MultimodalAnalyzer:
        """Get or create the multimodal analyzer instance."""
        if self._multimodal_analyzer is None:
            self._multimodal_analyzer = MultimodalAnalyzer(self.client)
        return self._multimodal_analyzer
    
    def classify_multimodal(
        self,
        location: Tuple[float, float],
        rgb_tile: np.ndarray,
        severity_map: np.ndarray,
        metadata: Optional[Dict] = None,
        unet_confidence: float = None,
        include_probability: bool = False,
        probability_map: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        TRUE multimodal ecosystem classification with RGB + severity overlay.
        
        This is the RECOMMENDED method for ecosystem classification. It sends
        BOTH the original satellite image AND the severity overlay to Gemini,
        enabling proper spatial reasoning about what is actually burned.
        
        Args:
            location: (latitude, longitude) tuple
            rgb_tile: (3, H, W) or (H, W, 3) RGB bands from Sentinel-2 (B4, B3, B2)
            severity_map: 2D numpy array with burn severity values (0-1)
            metadata: Optional additional metadata (ecosystem, legal, temporal context)
            unet_confidence: U-Net model confidence score
            include_probability: Whether to include probability heatmap as 3rd image
            probability_map: Raw sigmoid output for uncertainty visualization
            
        Returns:
            Complete multimodal analysis with:
            - visual_grounding: What Gemini observes in the landscape
            - segmentation_quality: Assessment of U-Net prediction quality
            - spatial_patterns: Fragmentation, edges, gradients
            - ecological_interpretation: Differential impacts, regeneration potential
            - priority_zones: 3-5 priority areas for restoration
            - signals_for_final_model: Machine-readable scores for Layer 3
        """
        # Enrich metadata with ecoregion info
        lat, lon = location
        ecoregion_key = get_ecoregion_from_location(lat, lon)
        
        enriched_metadata = metadata.copy() if metadata else {}
        if ecoregion_key:
            enriched_metadata['ecoregion'] = ecoregion_key
            enriched_metadata['biome'] = CALIFORNIA_ECOREGIONS[ecoregion_key]['name']
        
        # Run true multimodal analysis
        result = self.multimodal_analyzer.analyze(
            rgb_tile=rgb_tile,
            severity_map=severity_map,
            location=location,
            metadata=enriched_metadata,
            unet_confidence=unet_confidence,
            include_probability=include_probability,
            probability_map=probability_map
        )
        
        # Add ecoregion info to result
        if result.get('status') == 'complete':
            result['ecoregion'] = {
                'key': ecoregion_key,
                'info': CALIFORNIA_ECOREGIONS.get(ecoregion_key) if ecoregion_key else None
            }
        
        return result
    
    def classify(
        self,
        location: Tuple[float, float],
        severity_map: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        LEGACY ecosystem classification with severity colormap only.
        
        [DEPRECATED] This method only sends the severity colormap to Gemini,
        without the original RGB satellite image. Gemini cannot understand
        WHAT is burned, only that something is burned. Use classify_multimodal()
        for true spatial reasoning.
        
        This method is preserved for backward compatibility only.
        
        Args:
            location: (latitude, longitude) tuple
            severity_map: 2D numpy array with burn severity values (0-1)
            metadata: Optional additional metadata from Layer 1
            
        Returns:
            Structured ecosystem analysis with species recommendations
        """
        lat, lon = location
        
        # Get ecoregion hint
        ecoregion_key = get_ecoregion_from_location(lat, lon)
        if ecoregion_key:
            ecoregion_info = CALIFORNIA_ECOREGIONS[ecoregion_key]
            ecoregion_hint = f"{ecoregion_info['name']} (typical vegetation: {', '.join(ecoregion_info['vegetation'])})"
        else:
            ecoregion_hint = "Unknown - may be outside California coverage"
        
        # Compute severity statistics
        stats = compute_severity_stats(severity_map)
        stats_str = "\n".join([f"  - {k}: {v:.2%}" if 'ratio' in k else f"  - {k}: {v:.3f}" 
                               for k, v in stats.items()])
        
        # Convert severity map to image for multimodal analysis
        severity_image = severity_map_to_image(severity_map)
        
        # Format coordinates correctly for display
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        lat_display = f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"
        
        # Build prompt
        prompt = ECOSYSTEM_PROMPT.format(
            lat_display=lat_display,
            ecoregion=ecoregion_hint,
            stats=stats_str
        )
        
        # Call Gemini with multimodal input (image + text) and JSON output
        print(f"[LEGACY] Analyzing ecosystem at ({lat:.4f}, {lon:.4f})...")
        print(f"   [WARNING] Using legacy mode - consider classify_multimodal() for better results")
        print(f"   Ecoregion: {ecoregion_hint}")
        print(f"   High severity: {stats['high_severity_ratio']:.1%}")
        
        response = self.client.analyze_multimodal(
            prompt=prompt,
            images=[severity_image],
            use_json=True
        )
        
        if response.get('parsed'):
            result = response['parsed']
            result['_metadata'] = {
                'location': {'lat': lat, 'lon': lon},
                'ecoregion_key': ecoregion_key,
                'severity_stats': stats,
                'tokens_used': response['usage'],
                'analysis_mode': 'legacy_severity_only'
            }
            
            print(f"   [OK] Classified as: {result.get('ecosystem_type', 'Unknown')}")
            print(f"   Restoration method: {result.get('restoration_method', 'Unknown')}")
            
            return result
        else:
            # Return raw text if JSON parsing failed
            return {
                'error': 'Failed to parse structured response',
                'raw_response': response['text'],
                '_metadata': {
                    'location': {'lat': lat, 'lon': lon},
                    'severity_stats': stats,
                    'analysis_mode': 'legacy_severity_only'
                }
            }
    
    def get_species_with_grounding(
        self,
        ecosystem_type: str,
        location: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Get species recommendations with Google Search grounding.
        
        Uses Gemini's grounding feature to search for current
        best practices and native plant availability.
        
        Args:
            ecosystem_type: The classified ecosystem type
            location: (lat, lon) tuple
            
        Returns:
            Grounded species recommendations with sources
        """
        lat, lon = location
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        query = f"""
        I need native plant species recommendations for post-fire restoration in a 
        {ecosystem_type} ecosystem near {abs(lat):.2f}°{lat_dir}, {abs(lon):.2f}°{lon_dir}.
        
        Please search for:
        1. Current CAL FIRE or USFS recommendations for this ecosystem type
        2. Native plant nurseries in this region of California
        3. Recent research on post-fire restoration success rates
        
        Provide specific species with scientific names and any data on 
        survival rates in post-fire conditions.
        """
        
        print(f"Searching for grounded species data...")
        
        response = self.client.search_grounded(query)
        
        return {
            'recommendations': response['text'],
            'sources': response['sources'],
            'grounding_metadata': response.get('grounding_metadata')
        }


def classify_ecosystem(
    location: Tuple[float, float],
    severity_map: np.ndarray,
    metadata: Optional[Dict] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    LEGACY convenience function for ecosystem classification.
    
    [DEPRECATED] Use classify_ecosystem_multimodal() for true spatial reasoning.
    
    Args:
        location: (latitude, longitude) tuple
        severity_map: 2D numpy array with burn severity values (0-1)
        metadata: Optional additional metadata
        api_key: Optional Google API key
        
    Returns:
        Ecosystem classification results
    """
    client = create_client(api_key) if api_key else None
    classifier = EcosystemClassifier(gemini_client=client)
    return classifier.classify(location, severity_map, metadata)


def classify_ecosystem_multimodal(
    location: Tuple[float, float],
    rgb_tile: np.ndarray,
    severity_map: np.ndarray,
    metadata: Optional[Dict] = None,
    unet_confidence: float = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    TRUE multimodal ecosystem classification (RECOMMENDED).
    
    This function properly sends BOTH the original satellite image AND
    the severity overlay to Gemini, enabling spatial reasoning about
    what is actually burned in the landscape.
    
    Args:
        location: (latitude, longitude) tuple
        rgb_tile: (3, H, W) or (H, W, 3) RGB bands from Sentinel-2 (B4, B3, B2)
        severity_map: 2D numpy array with burn severity values (0-1)
        metadata: Optional additional metadata (ecosystem, legal, temporal)
        unet_confidence: U-Net model confidence score
        api_key: Optional Google API key
        **kwargs: Additional arguments (include_probability, probability_map)
        
    Returns:
        Complete multimodal analysis with:
        - visual_grounding: What Gemini observes in the landscape
        - segmentation_quality: Assessment of U-Net prediction quality
        - spatial_patterns: Fragmentation, edges, gradients
        - ecological_interpretation: Differential impacts, regeneration potential
        - priority_zones: 3-5 priority areas for restoration
        - signals_for_final_model: Machine-readable scores for Layer 3
    """
    client = create_client(api_key) if api_key else None
    classifier = EcosystemClassifier(gemini_client=client)
    return classifier.classify_multimodal(
        location=location,
        rgb_tile=rgb_tile,
        severity_map=severity_map,
        metadata=metadata,
        unet_confidence=unet_confidence,
        **kwargs
    )


if __name__ == "__main__":
    # Demo with synthetic data
    print("=" * 60)
    print("EcoRevive Ecosystem Classifier Demo")
    print("=" * 60)
    
    # Create synthetic data (simulating Dixie Fire)
    np.random.seed(42)
    
    # Synthetic RGB tile (simulating Sentinel-2 B4, B3, B2)
    rgb_tile = np.random.randint(100, 200, size=(3, 256, 256), dtype=np.uint8)
    
    # Synthetic severity map with spatial structure
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    severity_map = 0.5 * np.exp(-((x - 0.5)**2 + (y - 0.3)**2) / 0.1)
    severity_map += 0.3 * np.random.beta(2, 1, size=(256, 256))
    severity_map = np.clip(severity_map, 0, 1).astype(np.float32)
    
    # Dixie Fire location
    location = (40.05, -121.20)
    
    print("\n" + "=" * 40)
    print("Testing TRUE MULTIMODAL analysis (recommended)")
    print("=" * 40)
    
    try:
        result = classify_ecosystem_multimodal(
            location=location,
            rgb_tile=rgb_tile,
            severity_map=severity_map,
            unet_confidence=0.87
        )
        
        print("\nMultimodal Analysis Results:")
        if result.get('status') == 'complete':
            analysis = result.get('analysis', {})
            sq = analysis.get('segmentation_quality', {})
            print(f"   Segmentation Quality: {sq.get('overall_quality', 'N/A')}")
            print(f"   Confidence: {sq.get('confidence_in_prediction', 0):.0%}")
            
            signals = analysis.get('signals_for_final_model', {})
            print(f"   Restoration Potential: {signals.get('restoration_potential_score', 0):.2f}")
            print(f"   Intervention Urgency: {signals.get('intervention_urgency_score', 0):.2f}")
            print(f"   Risk Score: {signals.get('risk_score', 0):.2f}")
            
            zones = analysis.get('priority_zones', [])
            print(f"   Priority Zones: {len(zones)}")
            for zone in zones[:3]:
                print(f"     - Zone {zone.get('zone_id')}: {zone.get('urgency')} - {zone.get('recommended_intervention')}")
            
            if result.get('human_review', {}).get('required'):
                print(f"\n   [WARNING] Human review required: {result['human_review']['triggers']}")
        else:
            print(f"   Status: {result.get('status')}")
            if result.get('error'):
                print(f"   Error: {result.get('error')}")
        
    except ValueError as e:
        print(f"\n[WARNING] {e}")
        print("Set GOOGLE_API_KEY to run the demo.")

    print("\n" + "=" * 40)
    print("Testing LEGACY analysis (for comparison)")
    print("=" * 40)
    
    try:
        legacy_result = classify_ecosystem(location, severity_map)
        
        print("\nLegacy Results:")
        print(f"   Ecosystem Type: {legacy_result.get('ecosystem_type', 'N/A')}")
        print(f"   Restoration Method: {legacy_result.get('restoration_method', 'N/A')}")
        print(f"   Mode: {legacy_result.get('_metadata', {}).get('analysis_mode', 'N/A')}")
        
    except ValueError as e:
        print(f"\n[WARNING] {e}")
        print("Set GOOGLE_API_KEY to run the demo.")

