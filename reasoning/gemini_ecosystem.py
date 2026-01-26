"""
EcoRevive Ecosystem Classification Module
==========================================
Uses Gemini's multimodal capabilities to:
1. Analyze burn severity maps as images
2. Classify ecosystem type from location + visual patterns
3. Recommend native species with Google Search grounding
4. Determine restoration approach

This module showcases:
- Multimodal input (image + text)
- Structured JSON output
- Grounding with Google Search for species data
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

LOCATION: {lat:.4f}°N, {lon:.4f}°W
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
    """
    
    def __init__(self, gemini_client: Optional[EcoReviveGemini] = None):
        """
        Initialize the classifier.
        
        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self.client = gemini_client or create_client()
    
    def classify(
        self,
        location: Tuple[float, float],
        severity_map: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Classify ecosystem and generate restoration recommendations.
        
        This is the main entry point that demonstrates:
        - Multimodal input (severity map as image)
        - Structured JSON output
        - Location-aware analysis
        
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
        
        # Build prompt
        prompt = ECOSYSTEM_PROMPT.format(
            lat=lat,
            lon=lon,
            ecoregion=ecoregion_hint,
            stats=stats_str
        )
        
        # Call Gemini with multimodal input (image + text) and JSON output
        print(f"🌲 Analyzing ecosystem at ({lat:.4f}, {lon:.4f})...")
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
                'tokens_used': response['usage']
            }
            
            print(f"   ✅ Classified as: {result.get('ecosystem_type', 'Unknown')}")
            print(f"   Restoration method: {result.get('restoration_method', 'Unknown')}")
            
            return result
        else:
            # Return raw text if JSON parsing failed
            return {
                'error': 'Failed to parse structured response',
                'raw_response': response['text'],
                '_metadata': {
                    'location': {'lat': lat, 'lon': lon},
                    'severity_stats': stats
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
        
        query = f"""
        I need native plant species recommendations for post-fire restoration in a 
        {ecosystem_type} ecosystem near {lat:.2f}°N, {lon:.2f}°W in California.
        
        Please search for:
        1. Current CAL FIRE or USFS recommendations for this ecosystem type
        2. Native plant nurseries in this region of California
        3. Recent research on post-fire restoration success rates
        
        Provide specific species with scientific names and any data on 
        survival rates in post-fire conditions.
        """
        
        print(f"🔍 Searching for grounded species data...")
        
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
    Convenience function for ecosystem classification.
    
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


if __name__ == "__main__":
    # Demo with synthetic data
    print("=" * 60)
    print("🌲 EcoRevive Ecosystem Classifier Demo")
    print("=" * 60)
    
    # Create synthetic severity map (simulating Dixie Fire pattern)
    np.random.seed(42)
    severity_map = np.random.beta(2, 1, size=(256, 256))  # Skewed towards higher severity
    
    # Dixie Fire location
    location = (40.05, -121.20)
    
    try:
        result = classify_ecosystem(location, severity_map)
        
        print("\n📊 Classification Results:")
        print(json.dumps(result, indent=2, default=str))
        
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("Set GOOGLE_API_KEY to run the demo.")
