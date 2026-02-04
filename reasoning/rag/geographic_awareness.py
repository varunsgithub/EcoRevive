"""
Geographic Awareness Module
============================
Detects region from coordinates and determines knowledge level.
CRITICAL: Prevents the model from applying California species knowledge
to non-California locations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RegionInfo:
    """Information about a geographic region and its knowledge level."""
    region_name: str
    knowledge_level: str  # 'detailed', 'moderate', 'limited', 'minimal'
    biome_type: str
    fire_regime: str
    species_recommendations_permitted: bool
    disclosure_required: bool
    disclosure_text: str
    available_data: list


# Region boundaries for California (detailed knowledge)
CALIFORNIA_BOUNDS = {
    'lat_min': 32.5,
    'lat_max': 42.0,
    'lon_min': -124.5,
    'lon_max': -114.0
}

# Other regions with moderate knowledge
REGION_BOUNDS = {
    'western_usa': {
        'lat_min': 31.0, 'lat_max': 49.0,
        'lon_min': -125.0, 'lon_max': -102.0
    },
    'mediterranean_basin': {
        'lat_min': 30.0, 'lat_max': 47.0,
        'lon_min': -10.0, 'lon_max': 40.0
    },
    'southeastern_australia': {
        'lat_min': -44.0, 'lat_max': -28.0,
        'lon_min': 140.0, 'lon_max': 154.0
    },
}


def is_in_california(lat: float, lon: float) -> bool:
    """Check if coordinates are within California."""
    return (
        CALIFORNIA_BOUNDS['lat_min'] <= lat <= CALIFORNIA_BOUNDS['lat_max'] and
        CALIFORNIA_BOUNDS['lon_min'] <= lon <= CALIFORNIA_BOUNDS['lon_max']
    )


def get_region_from_coords(lat: float, lon: float) -> str:
    """Determine region from coordinates."""
    
    # Check California first (most detailed)
    if is_in_california(lat, lon):
        return 'california'
    
    # Check other known regions
    for region_name, bounds in REGION_BOUNDS.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
            bounds['lon_min'] <= lon <= bounds['lon_max']):
            return region_name
    
    # Infer from latitude zones
    abs_lat = abs(lat)
    if abs_lat < 23:
        return 'tropical'
    elif abs_lat < 35:
        return 'subtropical'
    elif abs_lat < 55:
        return 'temperate'
    elif abs_lat < 70:
        return 'boreal'
    else:
        return 'polar'


def infer_biome(lat: float, lon: float) -> Tuple[str, str]:
    """
    Infer biome type from coordinates.
    Returns (biome_type, fire_regime).
    """
    abs_lat = abs(lat)
    
    # Tropical zone
    if abs_lat < 23:
        # Very rough inference - would need precipitation data for accuracy
        return ('tropical_savanna', 'FIRE_DEPENDENT')
    
    # Subtropical
    elif abs_lat < 35:
        # West coast subtropical = mediterranean
        if (lon < -100 and lat > 0) or (lon > 100 and lat < 0):  # California-like or SW Australia
            return ('mediterranean_shrubland', 'FIRE_ADAPTED')
        else:
            return ('subtropical_forest', 'VARIABLE')
    
    # Temperate
    elif abs_lat < 55:
        if -125 < lon < -100 and lat > 0:  # Western North America mountains
            return ('temperate_conifer_forest', 'HISTORICALLY_FIRE_MAINTAINED')
        elif 140 < lon < 155 and lat < 0:  # SE Australia
            return ('eucalyptus_forest', 'EXTREMELY_FIRE_ADAPTED')
        else:
            return ('temperate_forest', 'VARIABLE')
    
    # Boreal
    elif abs_lat < 70:
        return ('boreal_forest', 'STAND_REPLACEMENT')
    
    else:
        return ('arctic_tundra', 'NOT_FIRE_ADAPTED')


def get_region_info(lat: float, lon: float) -> RegionInfo:
    """
    Get complete region information for coordinates.
    
    This is the main function to call from server.py.
    """
    region = get_region_from_coords(lat, lon)
    biome, fire_regime = infer_biome(lat, lon)
    
    if region == 'california':
        return RegionInfo(
            region_name='California',
            knowledge_level='detailed',
            biome_type=biome,
            fire_regime=fire_regime,
            species_recommendations_permitted=True,
            disclosure_required=False,
            disclosure_text='',
            available_data=[
                'species_catalog', 'ecoregion_classification',
                'fire_history', 'restoration_protocols', 'legal_framework'
            ]
        )
    
    elif region == 'western_usa':
        return RegionInfo(
            region_name='Western USA',
            knowledge_level='moderate',
            biome_type=biome,
            fire_regime=fire_regime,
            species_recommendations_permitted=False,
            disclosure_required=True,
            disclosure_text='Detailed species data not available for this region. Biome-level ecological reasoning applied. California species catalog does NOT apply here.',
            available_data=['fire_ecology_general', 'biome_classification', 'federal_land_types']
        )
    
    elif region == 'mediterranean_basin':
        return RegionInfo(
            region_name='Mediterranean Basin',
            knowledge_level='moderate',
            biome_type='mediterranean_shrubland',
            fire_regime='FIRE_ADAPTED',
            species_recommendations_permitted=False,
            disclosure_required=True,
            disclosure_text='Mediterranean shrubland fire ecology applied. Region-specific species recommendations require local expertise. California species catalog does NOT apply.',
            available_data=['fire_ecology_general', 'biome_classification', 'mediterranean_ecology']
        )
    
    elif region == 'southeastern_australia':
        return RegionInfo(
            region_name='Southeastern Australia',
            knowledge_level='moderate',
            biome_type='eucalyptus_forest',
            fire_regime='EXTREMELY_FIRE_ADAPTED',
            species_recommendations_permitted=False,
            disclosure_required=True,
            disclosure_text='Eucalyptus forest fire ecology applied. Australian native species data not in knowledge base. Epicormic resprouting and high-intensity fire dynamics considered.',
            available_data=['eucalyptus_fire_ecology', 'biome_classification', 'bushfire_dynamics']
        )
    
    else:
        # Limited or minimal knowledge
        return RegionInfo(
            region_name=region.replace('_', ' ').title(),
            knowledge_level='limited',
            biome_type=biome,
            fire_regime=fire_regime,
            species_recommendations_permitted=False,
            disclosure_required=True,
            disclosure_text=f'⚠️ LIMITED REGIONAL DATA: This assessment uses global biome-level ecological reasoning only ({biome}). No regional species data available. Site-specific guidance requires consultation with local ecological professionals. The California species catalog does NOT apply to this location.',
            available_data=['biome_classification_only']
        )


def format_geographic_context(region_info: RegionInfo) -> str:
    """
    Format geographic awareness information for system prompt injection.
    """
    lines = [
        "# GEOGRAPHIC AWARENESS",
        f"Region: {region_info.region_name}",
        f"Knowledge Level: {region_info.knowledge_level.upper()}",
        f"Biome Type: {region_info.biome_type}",
        f"Fire Regime: {region_info.fire_regime}",
        ""
    ]
    
    if region_info.disclosure_required:
        lines.append("## ⚠️ CRITICAL KNOWLEDGE LIMITATION")
        lines.append(region_info.disclosure_text)
        lines.append("")
    
    if not region_info.species_recommendations_permitted:
        lines.append("## SPECIES RECOMMENDATION RULES")
        lines.append("- DO NOT recommend specific species (no species catalog for this region)")
        lines.append("- Use biome-level guidance only (e.g., 'fire-adapted shrubs', 'resprouting hardwoods')")
        lines.append("- ALWAYS recommend: 'Consult local ecological expertise for species selection'")
        lines.append("- The California native species catalog does NOT apply here")
        lines.append("")
    
    lines.append(f"Available Data: {', '.join(region_info.available_data)}")
    
    return "\n".join(lines)


def get_biome_fire_ecology_summary(biome_type: str) -> str:
    """
    Get fire ecology summary for a biome type.
    Used when regional data is unavailable.
    """
    summaries = {
        'mediterranean_shrubland': """
MEDITERRANEAN SHRUBLAND FIRE ECOLOGY:
- Fire is NATURAL and NECESSARY in this ecosystem
- Plants are highly fire-adapted (lignotubers, fire-stimulated germination)
- Typical burn severity: HIGH (shrubland fires burn hot)
- Natural fire return interval: 30-100 years
- Recovery: Excellent natural regeneration expected (5-15 years)
- Main concern: Invasive grass invasion post-fire
""",
        'temperate_conifer_forest': """
TEMPERATE CONIFER FOREST FIRE ECOLOGY:
- Historically maintained by FREQUENT LOW-SEVERITY fire
- Fire suppression has increased severity of fires when they occur
- Thick-barked conifers (pine, fir) survive low-intensity fire
- Recovery for conifers: SLOW (decades for canopy closure)
- May need intervention if seed source distant (>500m)
- Type conversion to shrubland is a risk after high-severity fire
""",
        'eucalyptus_forest': """
EUCALYPTUS FOREST FIRE ECOLOGY:
- EXTREMELY fire-adapted - evolved with intense fire
- Epicormic resprouting: trees resprout from trunk within months
- Highest fire intensity globally (oil-rich foliage)
- Recovery: Eucalypts regenerate vigorously; understory varies
- Natural recovery usually excellent without intervention
""",
        'tropical_savanna': """
TROPICAL SAVANNA FIRE ECOLOGY:
- Fire is ESSENTIAL for maintaining grass-tree balance
- Without fire: woody encroachment and loss of grassland
- Fire return interval: 1-5 years (very frequent)
- Recovery: FAST - grasses regrow within weeks
- Fire suppression is the problem, not fire itself
""",
        'boreal_forest': """
BOREAL FOREST FIRE ECOLOGY:
- Stand-replacement fire regime (high severity is natural)
- Serotinous cones release seeds after fire
- Creates mosaic of age classes across landscape
- Full recovery: 50-200 years (slow-growing species)
- Intervention usually not needed - natural regeneration from cone seeds
""",
        'tropical_rainforest': """
TROPICAL RAINFOREST FIRE ECOLOGY:
- NOT FIRE-ADAPTED - fire indicates ecosystem degradation
- Any fire is damaging, not natural
- Recovery: Very slow, may not return to rainforest
- Intervention urgency: HIGH
- Often requires active restoration
"""
    }
    
    return summaries.get(biome_type, f"""
FIRE ECOLOGY ({biome_type.replace('_', ' ').title()}):
- Limited specific data for this biome type
- Apply general fire ecology principles
- Recommend consultation with local experts
""")
