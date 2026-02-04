"""
EcoRevive Hope Visualizer
=========================
Uses Gemini's Imagen 3 to generate inspiring visualizations of 
ecosystem recovery - the "Hope Visualizer" that shows communities
what their restoration work will accomplish.

Also includes recovery timeline forecasting using Gemini 1.5 Pro.

Key features:
- Imagen 3 for photorealistic future landscape generation
- Recovery timeline predictions
- Carbon sequestration estimates
- Shareable impact cards
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    import PIL.Image
except ImportError:
    PIL = None

from .gemini_client import EcoReviveGemini, create_client


# Recovery parameters by ecosystem type
RECOVERY_PROFILES = {
    "mixed_conifer": {
        "name": "Mixed Conifer Forest",
        "ground_cover_years": 1,
        "shrub_years": 3,
        "canopy_closure_years": 25,
        "carbon_rate_tonnes_per_ha_year": 4.5,
        "typical_species": ["Ponderosa Pine", "White Fir", "Black Oak", "Manzanita"],
        "visual_description": "tall conifer forest with mixed pine and fir trees, golden grasses beneath"
    },
    "coast_redwood": {
        "name": "Coast Redwood Forest",
        "ground_cover_years": 1,
        "shrub_years": 2,
        "canopy_closure_years": 40,
        "carbon_rate_tonnes_per_ha_year": 8.0,
        "typical_species": ["Coast Redwood", "Douglas Fir", "Tan Oak", "Sword Fern"],
        "visual_description": "towering redwood forest with lush fern understory, misty atmosphere"
    },
    "oak_woodland": {
        "name": "Oak Woodland",
        "ground_cover_years": 1,
        "shrub_years": 5,
        "canopy_closure_years": 30,
        "carbon_rate_tonnes_per_ha_year": 2.5,
        "typical_species": ["Blue Oak", "Valley Oak", "California Buckeye"],
        "visual_description": "golden rolling hills with scattered majestic oak trees, grassland"
    },
    "chaparral": {
        "name": "Chaparral Shrubland",
        "ground_cover_years": 1,
        "shrub_years": 5,
        "canopy_closure_years": 15,
        "carbon_rate_tonnes_per_ha_year": 1.5,
        "typical_species": ["Chamise", "Ceanothus", "Manzanita", "Toyon"],
        "visual_description": "dense shrubland covering hillsides, flowering ceanothus in spring"
    }
}


RECOVERY_PROMPT = """You are an expert restoration ecologist forecasting post-fire ecosystem recovery.

ECOSYSTEM TYPE: {ecosystem_type}
BURN SEVERITY: {severity:.1%} mean severity
RESTORATION METHOD: {method}
AREA: {area_hectares} hectares

Based on scientific literature and recovery patterns from similar California fires, provide a detailed recovery forecast.

Consider:
1. The specific ecosystem's natural recovery rate
2. How burn severity affects regeneration
3. Whether active restoration accelerates recovery
4. Regional climate factors
5. Carbon sequestration potential

Return a detailed JSON forecast with this structure:
{{
    "recovery_timeline": {{
        "year_1": {{
            "description": "string",
            "ground_cover_percent": number,
            "key_changes": ["string"]
        }},
        "year_3": {{
            "description": "string",
            "shrub_cover_percent": number,
            "tree_seedling_height_m": number,
            "key_changes": ["string"]
        }},
        "year_5": {{
            "description": "string",
            "vegetation_cover_percent": number,
            "key_changes": ["string"]
        }},
        "year_10": {{
            "description": "string", 
            "canopy_cover_percent": number,
            "average_tree_height_m": number,
            "key_changes": ["string"]
        }},
        "year_15": {{
            "description": "string",
            "canopy_cover_percent": number,
            "biodiversity_recovery_percent": number,
            "key_changes": ["string"]
        }}
    }},
    "carbon_sequestration": {{
        "total_year_15_tonnes_co2": number,
        "annual_rate_tonnes_per_hectare": number,
        "equivalent_cars_offset": number
    }},
    "success_factors": ["string"],
    "risks": ["string"],
    "monitoring_indicators": [
        {{
            "indicator": "string",
            "target_year_5": "string",
            "target_year_15": "string"
        }}
    ],
    "hope_message": "An inspiring 2-sentence message for community volunteers about what they're building toward"
}}
"""


IMAGEN_PROMPT_TEMPLATE = """
Photorealistic landscape photograph of a {visual_description} in California.
{years} years after a wildfire, showing healthy ecosystem recovery.
{species_phrase}
{lighting}
Natural, documentary style photography.
No text, labels, watermarks, or human figures.
Inspiring and hopeful mood showing nature's resilience.
"""


class HopeVisualizer:
    """
    Generates hope visualizations and recovery forecasts.
    
    Uses:
    - Gemini 1.5 Pro for recovery forecasting
    - Imagen 3 for photorealistic future landscape images
    """
    
    def __init__(self, gemini_client: Optional[EcoReviveGemini] = None):
        """
        Initialize the visualizer.
        
        Args:
            gemini_client: Optional pre-configured client
        """
        self.client = gemini_client or create_client()
    
    def forecast_recovery(
        self,
        ecosystem_type: str,
        mean_severity: float,
        restoration_method: str = "combination",
        area_hectares: float = 100.0
    ) -> Dict[str, Any]:
        """
        Generate a detailed recovery forecast using Gemini.
        
        Args:
            ecosystem_type: Type of ecosystem (e.g., "mixed_conifer")
            mean_severity: Mean burn severity (0-1)
            restoration_method: "active_reforestation", "natural_regeneration", or "combination"
            area_hectares: Size of restoration area
            
        Returns:
            Detailed recovery forecast with timeline and carbon estimates
        """
        # Normalize ecosystem type
        ecosystem_key = ecosystem_type.lower().replace(" ", "_")
        if ecosystem_key not in RECOVERY_PROFILES:
            ecosystem_key = "mixed_conifer"  # Default
        
        profile = RECOVERY_PROFILES[ecosystem_key]
        
        prompt = RECOVERY_PROMPT.format(
            ecosystem_type=profile["name"],
            severity=mean_severity,
            method=restoration_method,
            area_hectares=area_hectares
        )
        
        print(f"Forecasting recovery for {profile['name']}...")
        print(f"   Severity: {mean_severity:.1%}, Area: {area_hectares} ha")
        
        response = self.client.analyze_multimodal(prompt, use_json=True)
        
        if response.get("parsed"):
            result = response["parsed"]
            result["_metadata"] = {
                "ecosystem_profile": profile,
                "tokens_used": response["usage"]
            }
            
            print(f"   [OK] Forecast generated")
            return result
        else:
            return {
                "error": "Failed to generate forecast",
                "raw_response": response.get("text")
            }
    
    def generate_hope_image(
        self,
        ecosystem_type: str,
        years_in_future: int = 15,
        species_list: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a hope visualization image using Imagen 3.
        
        Shows what the restored ecosystem will look like at a future time point.
        
        Args:
            ecosystem_type: Type of ecosystem
            years_in_future: How many years to project (default 15)
            species_list: Optional list of species to include
            save_path: Optional path to save the generated image
            
        Returns:
            Dict with image data and generation info
        """
        # Get ecosystem profile
        ecosystem_key = ecosystem_type.lower().replace(" ", "_")
        if ecosystem_key not in RECOVERY_PROFILES:
            ecosystem_key = "mixed_conifer"
        
        profile = RECOVERY_PROFILES[ecosystem_key]
        
        # Build species phrase
        species = species_list or profile["typical_species"][:3]
        species_phrase = f"Healthy {', '.join(species)} visible in the scene."
        
        # Vary lighting based on ecosystem
        lighting_options = {
            "mixed_conifer": "Golden hour sunlight filtering through trees, warm tones",
            "coast_redwood": "Soft misty morning light, ethereal atmosphere",
            "oak_woodland": "Late afternoon golden light, rolling hills",
            "chaparral": "Bright clear day, blue sky, spring wildflowers"
        }
        lighting = lighting_options.get(ecosystem_key, "Golden hour lighting")
        
        # Build prompt
        prompt = IMAGEN_PROMPT_TEMPLATE.format(
            visual_description=profile["visual_description"],
            years=years_in_future,
            species_phrase=species_phrase,
            lighting=lighting
        )
        
        print(f"Generating hope visualization...")
        print(f"   Ecosystem: {profile['name']}")
        print(f"   Years in future: {years_in_future}")
        
        # Generate with Imagen
        images = self.client.generate_image(
            prompt=prompt,
            aspect_ratio="16:9",
            num_images=1
        )
        
        result = {
            "prompt": prompt,
            "ecosystem_type": profile["name"],
            "years_in_future": years_in_future,
            "species_shown": species,
            "images_generated": len(images)
        }
        
        if images:
            result["image"] = images[0]
            
            # Save if path provided
            if save_path and PIL:
                images[0].save(save_path)
                result["saved_to"] = save_path
                print(f"   [OK] Saved to: {save_path}")
            
            print(f"   [OK] Generated {len(images)} image(s)")
        else:
            print(f"   [WARNING] Image generation failed (Imagen may require additional access)")
            result["fallback_description"] = self._generate_text_visualization(
                profile, years_in_future, species
            )
        
        return result
    
    def _generate_text_visualization(
        self,
        profile: Dict,
        years: int,
        species: List[str]
    ) -> str:
        """Generate a text description as fallback when Imagen is unavailable."""
        
        prompt = f"""
        Describe in vivid, inspiring detail what a {profile['name']} in California 
        will look like {years} years after a wildfire, following active restoration.
        
        Include sensory details: what you see, hear, and smell.
        Mention these species: {', '.join(species)}
        
        Write 2-3 paragraphs that would inspire a community volunteer about
        what their restoration work will create.
        """
        
        response = self.client.analyze_multimodal(prompt, use_json=False)
        return response["text"]
    
    def generate_impact_card(
        self,
        trees_planted: int,
        area_hectares: float,
        volunteers: int,
        ecosystem_type: str = "mixed_conifer"
    ) -> Dict[str, Any]:
        """
        Generate data for a shareable impact card.
        
        Creates the content for a "We planted X trees today!" style
        shareable graphic for social media.
        
        Args:
            trees_planted: Number of trees planted
            area_hectares: Area covered
            volunteers: Number of volunteers
            ecosystem_type: Type of ecosystem restored
            
        Returns:
            Impact card data with stats and messaging
        """
        # Calculate impact metrics
        ecosystem_key = ecosystem_type.lower().replace(" ", "_")
        profile = RECOVERY_PROFILES.get(ecosystem_key, RECOVERY_PROFILES["mixed_conifer"])
        
        # Carbon calculation (rough estimate based on mature forest)
        carbon_per_tree_lifetime = 0.5  # tonnes CO2 over 40 years
        total_carbon = trees_planted * carbon_per_tree_lifetime
        cars_equivalent = total_carbon / 4.6  # Average car emits 4.6 tonnes/year
        
        prompt = f"""
        Create an inspiring impact message for social media about a restoration event:
        
        - {trees_planted} native trees planted
        - {area_hectares} hectares restored  
        - {volunteers} community volunteers
        - Ecosystem: {profile['name']}
        - Estimated lifetime carbon capture: {total_carbon:.0f} tonnes CO2
        
        Write:
        1. A compelling headline (under 10 words)
        2. A 2-sentence body that's inspiring but not preachy
        3. A call-to-action for others to join
        
        Return as JSON with keys: headline, body, call_to_action
        """
        
        response = self.client.analyze_multimodal(prompt, use_json=True)
        
        result = {
            "stats": {
                "trees_planted": trees_planted,
                "area_hectares": area_hectares,
                "volunteers": volunteers,
                "carbon_tonnes_lifetime": round(total_carbon, 1),
                "cars_offset_equivalent": round(cars_equivalent, 1)
            },
            "ecosystem": profile["name"],
            "messaging": response.get("parsed", {})
        }
        
        return result
    
    def create_before_after_slider_data(
        self,
        current_severity_image: Any,
        ecosystem_type: str,
        years_in_future: int = 15
    ) -> Dict[str, Any]:
        """
        Create data for a before/after slider visualization.
        
        Args:
            current_severity_image: Current burn severity image
            ecosystem_type: Target ecosystem type
            years_in_future: How far to project
            
        Returns:
            Data structure for rendering before/after slider
        """
        # Generate "after" image
        after_result = self.generate_hope_image(
            ecosystem_type=ecosystem_type,
            years_in_future=years_in_future
        )
        
        return {
            "before": {
                "image": current_severity_image,
                "label": "Current Condition",
                "description": "Post-fire burn severity"
            },
            "after": {
                "image": after_result.get("image"),
                "label": f"Year {years_in_future} Vision",
                "description": after_result.get("fallback_description", 
                    f"Restored {ecosystem_type} ecosystem")
            },
            "years_projected": years_in_future
        }


def forecast_recovery(
    ecosystem_type: str,
    mean_severity: float,
    restoration_method: str = "combination",
    area_hectares: float = 100.0,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for recovery forecasting."""
    client = create_client(api_key) if api_key else None
    visualizer = HopeVisualizer(gemini_client=client)
    return visualizer.forecast_recovery(
        ecosystem_type, mean_severity, restoration_method, area_hectares
    )


def generate_hope_visualization(
    ecosystem_type: str,
    years_in_future: int = 15,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for hope image generation."""
    client = create_client(api_key) if api_key else None
    visualizer = HopeVisualizer(gemini_client=client)
    return visualizer.generate_hope_image(ecosystem_type, years_in_future)


if __name__ == "__main__":
    print("=" * 60)
    print("EcoRevive Hope Visualizer Demo")
    print("=" * 60)
    
    try:
        visualizer = HopeVisualizer()
        
        # Test recovery forecast
        print("\n--- Recovery Forecast ---")
        forecast = visualizer.forecast_recovery(
            ecosystem_type="Mixed Conifer",
            mean_severity=0.72,
            restoration_method="combination",
            area_hectares=100
        )
        print(json.dumps(forecast, indent=2, default=str)[:1000] + "...")
        
        # Test impact card
        print("\n--- Impact Card ---")
        card = visualizer.generate_impact_card(
            trees_planted=500,
            area_hectares=2.5,
            volunteers=25,
            ecosystem_type="mixed_conifer"
        )
        print(json.dumps(card, indent=2))
        
    except ValueError as e:
        print(f"\n[WARNING] {e}")
        print("Set GOOGLE_API_KEY to run the demo.")
