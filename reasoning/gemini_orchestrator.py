"""
EcoRevive Gemini Orchestrator
=============================
Demonstrates Gemini's Function Calling capabilities to coordinate
multiple analysis modules into a unified restoration assessment.

The orchestrator lets Gemini decide:
1. Which analyses to run based on user needs
2. What order to run them in
3. How to synthesize results into actionable recommendations

This showcases agentic behavior where Gemini acts as the "brain"
coordinating specialized tools.
"""

import json
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np

from .gemini_client import EcoReviveGemini, create_client


# Tool declarations for Gemini function calling
RESTORATION_TOOLS = [
    {
        "name": "analyze_ecosystem",
        "description": "Classify the ecosystem type and recommend native species for restoration based on burn severity patterns and location. Returns ecosystem type, species palette, and restoration method.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the burn site"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the burn site"
                },
                "mean_severity": {
                    "type": "number",
                    "description": "Mean burn severity (0-1 scale)"
                },
                "high_severity_ratio": {
                    "type": "number",
                    "description": "Fraction of area with high severity burn"
                }
            },
            "required": ["latitude", "longitude", "mean_severity"]
        }
    },
    {
        "name": "check_safety_hazards",
        "description": "Identify safety hazards at the burn site including widowmaker trees, landslide risk areas, and access restrictions. Critical for community volunteer safety.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the site"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the site"
                },
                "severity_pattern": {
                    "type": "string",
                    "description": "Description of burn severity pattern"
                },
                "terrain_info": {
                    "type": "string",
                    "description": "Any known terrain information (slope, aspect)"
                }
            },
            "required": ["latitude", "longitude"]
        }
    },
    {
        "name": "lookup_land_ownership",
        "description": "Look up land ownership information including federal/state/private status, managing agency, and contact information for obtaining restoration permits.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the parcel"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the parcel"
                }
            },
            "required": ["latitude", "longitude"]
        }
    },
    {
        "name": "forecast_recovery",
        "description": "Generate a recovery timeline forecast showing expected vegetation recovery at different time horizons (1, 5, 10, 15 years) along with carbon sequestration estimates.",
        "parameters": {
            "type": "object",
            "properties": {
                "ecosystem_type": {
                    "type": "string",
                    "description": "Type of ecosystem being restored"
                },
                "mean_severity": {
                    "type": "number",
                    "description": "Mean burn severity (0-1)"
                },
                "restoration_method": {
                    "type": "string",
                    "enum": ["active_reforestation", "natural_regeneration", "combination"],
                    "description": "Planned restoration approach"
                },
                "area_hectares": {
                    "type": "number",
                    "description": "Size of restoration area in hectares"
                }
            },
            "required": ["ecosystem_type", "mean_severity", "restoration_method"]
        }
    },
    {
        "name": "generate_supply_checklist",
        "description": "Generate a supplies and equipment checklist for restoration volunteers based on the site conditions and planned activities.",
        "parameters": {
            "type": "object",
            "properties": {
                "restoration_activities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of planned restoration activities"
                },
                "team_size": {
                    "type": "integer",
                    "description": "Expected number of volunteers"
                },
                "duration_days": {
                    "type": "integer",
                    "description": "Duration of restoration event in days"
                },
                "safety_hazards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known safety hazards at the site"
                }
            },
            "required": ["restoration_activities", "team_size"]
        }
    },
    {
        "name": "generate_hope_visualization",
        "description": "Generate an AI image showing what the restored site will look like at a future time point. Uses Imagen to create inspiring before/after visualizations.",
        "parameters": {
            "type": "object",
            "properties": {
                "ecosystem_type": {
                    "type": "string",
                    "description": "Target ecosystem type"
                },
                "species_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key species that will be present"
                },
                "years_in_future": {
                    "type": "integer",
                    "description": "How many years into the future to visualize"
                }
            },
            "required": ["ecosystem_type", "years_in_future"]
        }
    }
]

# User type prompts
USER_TYPE_PROMPTS = {
    "professional": """
You are coordinating a restoration assessment for a professional NGO project manager.

They need to make budget allocation decisions and defend them to donors. Focus on:
- Detailed ecosystem classification with species prescriptions
- Land ownership and legal requirements
- Recovery timeline with measurable milestones
- Prioritization for maximum impact

The site is at coordinates ({lat}, {lon}) with mean burn severity of {severity:.1%}.

Analyze what tools you need to call to produce a comprehensive restoration plan.
Think step by step about what information is needed and in what order.
""",
    
    "community": """
You are helping a community organizer who wants to bring volunteers to help restore 
a burned area near their home.

They need practical, safety-first guidance. Focus on:
- Safety hazards (this is the MOST important - don't let anyone get hurt)
- Who owns the land and who to contact for permission
- What supplies to bring
- A hopeful visualization of what their work will accomplish

The site is at coordinates ({lat}, {lon}) with mean burn severity of {severity:.1%}.

Analyze what tools you need to call to help this community organizer.
Safety first, then logistics, then inspiration.
"""
}


class RestorationOrchestrator:
    """
    Gemini-powered orchestrator that coordinates restoration analysis.
    
    Uses function calling to let Gemini decide which analyses to run
    and in what order, demonstrating agentic AI behavior.
    """
    
    def __init__(self, gemini_client: Optional[EcoReviveGemini] = None):
        """
        Initialize the orchestrator.
        
        Args:
            gemini_client: Optional pre-configured client
        """
        self.client = gemini_client or create_client()
        self.function_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default function handlers."""
        # These are placeholder implementations
        # In production, these would call the actual modules
        
        self.function_handlers["analyze_ecosystem"] = self._handle_ecosystem
        self.function_handlers["check_safety_hazards"] = self._handle_safety
        self.function_handlers["lookup_land_ownership"] = self._handle_ownership
        self.function_handlers["forecast_recovery"] = self._handle_forecast
        self.function_handlers["generate_supply_checklist"] = self._handle_supplies
        self.function_handlers["generate_hope_visualization"] = self._handle_hope
    
    def register_handler(self, name: str, handler: Callable):
        """Register a custom function handler."""
        self.function_handlers[name] = handler
    
    def orchestrate(
        self,
        location: Tuple[float, float],
        severity_stats: Dict[str, float],
        user_type: str = "professional",
        area_hectares: float = 100.0
    ) -> Dict[str, Any]:
        """
        Run a full orchestrated restoration analysis.
        
        Gemini will decide which tools to call based on:
        - The user type (professional vs community)
        - The site characteristics
        - What information is needed
        
        Args:
            location: (lat, lon) tuple
            severity_stats: Dict with burn severity statistics
            user_type: "professional" or "community"
            area_hectares: Size of the restoration area
            
        Returns:
            Complete restoration assessment with all analyses
        """
        lat, lon = location
        severity = severity_stats.get("mean_severity", 0.5)
        
        # Get appropriate prompt
        prompt_template = USER_TYPE_PROMPTS.get(user_type, USER_TYPE_PROMPTS["professional"])
        prompt = prompt_template.format(lat=lat, lon=lon, severity=severity)
        
        print(f"🎯 Starting orchestrated analysis for {user_type} user")
        print(f"   Location: ({lat:.4f}, {lon:.4f})")
        print(f"   Mean severity: {severity:.1%}")
        print()
        
        # First call: Let Gemini decide what to do
        response = self.client.generate_with_functions(
            prompt=prompt,
            tools=RESTORATION_TOOLS,
            auto_execute=False
        )
        
        results = {
            "user_type": user_type,
            "location": {"lat": lat, "lon": lon},
            "severity_stats": severity_stats,
            "area_hectares": area_hectares,
            "analyses": {},
            "function_calls": []
        }
        
        # Execute requested functions
        for call in response.get("function_calls", []):
            func_name = call["name"]
            func_args = call["args"]
            
            print(f"   📞 Gemini requested: {func_name}")
            
            # Add location to args if not present
            if "latitude" not in func_args and "longitude" not in func_args:
                func_args["latitude"] = lat
                func_args["longitude"] = lon
            
            # Execute handler
            if func_name in self.function_handlers:
                try:
                    result = self.function_handlers[func_name](**func_args)
                    results["analyses"][func_name] = result
                    results["function_calls"].append({
                        "name": func_name,
                        "args": func_args,
                        "status": "success"
                    })
                    print(f"      ✅ {func_name} completed")
                except Exception as e:
                    results["function_calls"].append({
                        "name": func_name,
                        "args": func_args,
                        "status": "error",
                        "error": str(e)
                    })
                    print(f"      ❌ {func_name} failed: {e}")
            else:
                print(f"      ⚠️ No handler for {func_name}")
        
        # Synthesize final report
        results["synthesis"] = self._synthesize_results(results, user_type)
        
        return results
    
    def _synthesize_results(
        self,
        results: Dict[str, Any],
        user_type: str
    ) -> Dict[str, Any]:
        """
        Use Gemini to synthesize all analyses into a final report.
        """
        analyses_summary = json.dumps(results["analyses"], indent=2, default=str)
        
        synthesis_prompt = f"""
        You have completed the following analyses for a {user_type} user:
        
        {analyses_summary}
        
        Synthesize these results into a clear, actionable summary.
        
        For a professional user, structure as an executive summary with:
        - Key findings
        - Recommended actions with priorities
        - Budget considerations
        - Timeline
        
        For a community user, structure as a simple action plan with:
        - Safety briefing (most important!)
        - Who to contact first
        - What to bring
        - What to expect
        
        Be specific and actionable.
        """
        
        response = self.client.analyze_multimodal(synthesis_prompt, use_json=False)
        
        return {
            "summary": response["text"],
            "tokens_used": response["usage"]
        }
    
    # Placeholder function handlers
    # These would be replaced with actual implementations
    
    def _handle_ecosystem(self, **kwargs) -> Dict:
        """Handle ecosystem analysis request."""
        # In production, this would call gemini_ecosystem.py
        from .gemini_ecosystem import classify_ecosystem
        
        severity_map = np.random.beta(2, 1, size=(256, 256))  # Placeholder
        return classify_ecosystem(
            location=(kwargs["latitude"], kwargs["longitude"]),
            severity_map=severity_map
        )
    
    def _handle_safety(self, **kwargs) -> Dict:
        """Handle safety hazard check."""
        # Placeholder - would call gemini_safety.py
        return {
            "hazards_identified": True,
            "widowmaker_zones": 12,
            "landslide_risk_areas": 3,
            "recommended_ppe": ["Hard hat", "Steel-toe boots", "Eye protection"],
            "no_go_zones": ["Steep slopes >60%", "Standing dead trees >50ft"]
        }
    
    def _handle_ownership(self, **kwargs) -> Dict:
        """Handle land ownership lookup."""
        # Placeholder - would call gemini_legal.py with grounding
        return {
            "owner_type": "federal",
            "agency": "USDA Forest Service",
            "unit": "Plumas National Forest",
            "contact": "Feather River Ranger District",
            "phone": "(530) 534-6500",
            "permit_required": True,
            "permit_type": "Special Use Permit for Restoration Activities"
        }
    
    def _handle_forecast(self, **kwargs) -> Dict:
        """Handle recovery forecast."""
        # Placeholder - would call gemini_hope.py
        ecosystem = kwargs.get("ecosystem_type", "Mixed Conifer")
        severity = kwargs.get("mean_severity", 0.7)
        method = kwargs.get("restoration_method", "combination")
        area = kwargs.get("area_hectares", 100)
        
        return {
            "timeline": {
                "year_1": "Ground cover establishment (grasses, forbs)",
                "year_5": "Shrub layer developing, planted seedlings 2-3m tall",
                "year_10": "Young forest structure emerging",
                "year_15": "Closed canopy in moderate severity zones"
            },
            "carbon_sequestration": {
                "year_15_tonnes_co2": area * 50,  # Rough estimate
                "annual_rate_tonnes_per_hectare": 3.5
            },
            "success_probability": 0.85 if method == "combination" else 0.70
        }
    
    def _handle_supplies(self, **kwargs) -> Dict:
        """Handle supply checklist generation."""
        activities = kwargs.get("restoration_activities", ["planting"])
        team_size = kwargs.get("team_size", 20)
        
        return {
            "safety_equipment": [
                f"{team_size}x Hard hats",
                f"{team_size}x Work gloves",
                f"{team_size}x Eye protection",
                "4x First aid kits",
                "Emergency communication device"
            ],
            "planting_supplies": [
                f"{team_size * 10}x Native seedlings",
                f"{team_size}x Planting tubes",
                f"{team_size}x Dibble bars or augers",
                "500 lbs native seed mix",
                "Straw wattles for erosion control"
            ],
            "logistics": [
                "Water - 1 gallon per person per day",
                "Shade structures",
                "Portable toilets",
                "Trash bags for debris"
            ]
        }
    
    def _handle_hope(self, **kwargs) -> Dict:
        """Handle hope visualization generation."""
        ecosystem = kwargs.get("ecosystem_type", "forest")
        years = kwargs.get("years_in_future", 15)
        species = kwargs.get("species_list", ["Ponderosa Pine", "Black Oak"])
        
        # Build Imagen prompt
        prompt = f"""
        Photorealistic landscape photograph of a healthy {ecosystem} 
        in California, {years} years after wildfire recovery.
        Thriving young forest with {', '.join(species[:3])}.
        Golden hour lighting, mountains in background.
        Inspiring and hopeful mood showing nature's resilience.
        No text, labels, or human elements.
        """
        
        # Try to generate with Imagen
        images = self.client.generate_image(prompt)
        
        return {
            "prompt_used": prompt,
            "images_generated": len(images),
            "years_in_future": years,
            "description": f"A vision of the restored {ecosystem} in {years} years"
        }


def orchestrate_analysis(
    location: Tuple[float, float],
    severity_stats: Dict[str, float],
    user_type: str = "professional",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for orchestrated analysis.
    
    Args:
        location: (lat, lon) tuple
        severity_stats: Dict with severity statistics
        user_type: "professional" or "community"
        api_key: Optional Google API key
        
    Returns:
        Complete orchestrated analysis results
    """
    client = create_client(api_key) if api_key else None
    orchestrator = RestorationOrchestrator(gemini_client=client)
    return orchestrator.orchestrate(location, severity_stats, user_type)


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 EcoRevive Orchestrator Demo")
    print("=" * 60)
    
    # Demo data
    location = (40.05, -121.20)  # Dixie Fire
    severity_stats = {
        "mean_severity": 0.72,
        "high_severity_ratio": 0.45,
        "moderate_severity_ratio": 0.35,
        "low_severity_ratio": 0.15,
        "unburned_ratio": 0.05
    }
    
    try:
        # Test professional user flow
        print("\n--- Professional User Scenario ---")
        result = orchestrate_analysis(location, severity_stats, "professional")
        print(f"\nFunction calls made: {len(result['function_calls'])}")
        for call in result["function_calls"]:
            print(f"  - {call['name']}: {call['status']}")
        
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("Set GOOGLE_API_KEY to run the demo.")
