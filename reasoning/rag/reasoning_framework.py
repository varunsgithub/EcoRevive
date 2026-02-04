"""
EcoRevive RAG Reasoning Framework v2.0
=======================================
Principal Architect Reviewed - Production Grade

REVIEW SUMMARY:
- Tightened area-type eligibility with complete coverage
- Added explicit HARD_PROHIBITED risks per area type
- Strengthened evidence anchoring with confidence gates
- Added reasoning checkpoints and decision trees
- Removed permissive fallbacks
- Added temporal validity constraints

Design Philosophy:
- RAGs are COGNITIVE CONSTRAINTS, not content generators
- Silence is ALWAYS preferable to speculation
- Every claim must trace to a JSON field
- Proportionality is mandatory, not optional
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field


# =============================================================================
# AREA TYPE CLASSIFICATION (REFINED)
# =============================================================================

class AreaType(Enum):
    """Land use classification for reasoning scope control."""
    URBAN_DENSE = "urban_dense"              # CBD, high-rise
    URBAN_RESIDENTIAL = "urban_residential"  # Neighborhoods
    SUBURBAN = "suburban"                    # Mixed residential
    RURAL_RESIDENTIAL = "rural_residential"  # Sparse housing, WUI
    AGRICULTURAL = "agricultural"            # Farmland
    COMMERCIAL = "commercial"                # Retail, office
    INDUSTRIAL = "industrial"                # Manufacturing, warehouses
    FOREST_NATURAL = "forest_natural"        # Undeveloped forest
    GRASSLAND = "grassland"                  # Open range, savanna
    SHRUBLAND = "shrubland"                  # Chaparral, scrub
    WETLAND = "wetland"                      # Riparian, marsh
    MIXED_USE = "mixed_use"                  # Combined zones
    UNDEVELOPED = "undeveloped"              # Raw land


@dataclass
class AreaContext:
    """
    Context package for area-qualified reasoning.
    
    This is the PRIMARY INPUT to all reasoning RAGs.
    All fields must be derived from JSON evidence.
    """
    area_type: AreaType
    has_vegetation: bool
    has_structures: bool
    has_water_bodies: bool
    population_density: str  # "none", "low", "medium", "high"
    infrastructure_present: List[str] = field(default_factory=list)
    confidence_level: float = 0.5
    data_age_days: int = 0  # Days since imagery capture


# =============================================================================
# RAG 1: AREA-TYPE RISK ELIGIBILITY FILTER (REFINED)
# =============================================================================

# ELIGIBLE risks per area type (whitelist - ONLY these may be mentioned)
AREA_ELIGIBLE_RISKS: Dict[AreaType, List[str]] = {
    AreaType.URBAN_DENSE: [
        "structural_damage",
        "air_quality",
        "utility_disruption",
        "traffic_flow",
        "emergency_access",
        "public_gathering_areas",
    ],
    
    AreaType.URBAN_RESIDENTIAL: [
        "structural_damage",
        "air_quality",
        "utility_disruption",
        "yard_vegetation",
        "public_safety",
        "neighborhood_evacuation",
    ],
    
    AreaType.SUBURBAN: [
        "structural_damage",
        "air_quality",
        "utility_disruption",
        "landscape_vegetation",
        "defensible_space",
        "evacuation_routes",
        "irrigation_systems",
    ],
    
    AreaType.RURAL_RESIDENTIAL: [
        "structural_damage",
        "vegetation_proximity",
        "access_road_condition",
        "water_availability",
        "defensible_space",
        "evacuation_routes",
        "propane_tank_exposure",
    ],
    
    AreaType.FOREST_NATURAL: [
        "canopy_damage",
        "understory_loss",
        "soil_erosion",
        "watershed_impact",
        "wildlife_habitat",
        "regeneration_potential",
        "standing_dead_trees",  # Widowmakers - ONLY FOREST AREAS
        "fuel_load",
        "snag_hazard",
        "root_damage",
    ],
    
    AreaType.SHRUBLAND: [
        "shrub_recovery",
        "soil_erosion",
        "invasive_species_risk",
        "watershed_impact",
        "wildlife_corridor",
        "fire_regime_alteration",
    ],
    
    AreaType.GRASSLAND: [
        "grass_recovery",
        "soil_erosion",
        "grazing_impact",
        "wind_erosion",
        "invasive_species_risk",
    ],
    
    AreaType.AGRICULTURAL: [
        "crop_damage",
        "soil_health",
        "irrigation_infrastructure",
        "livestock_safety",
        "windbreak_loss",
        "pollinator_impact",
    ],
    
    AreaType.INDUSTRIAL: [
        "structural_damage",
        "hazardous_materials",
        "equipment_damage",
        "supply_chain",
        "worker_safety",
        "containment_breach",
    ],
    
    AreaType.WETLAND: [
        "hydrology_alteration",
        "riparian_damage",
        "water_quality",
        "habitat_loss",
        "sediment_transport",
        "fish_passage",
    ],
    
    AreaType.COMMERCIAL: [
        "structural_damage",
        "business_interruption",
        "parking_infrastructure",
        "signage_damage",
        "customer_access",
    ],
    
    AreaType.MIXED_USE: [
        "structural_damage",
        "air_quality",
        "utility_disruption",
        "defensible_space",
        "evacuation_routes",
    ],
    
    AreaType.UNDEVELOPED: [
        "vegetation_damage",
        "soil_erosion",
        "watershed_impact",
        "wildlife_habitat",
        "natural_regeneration",
    ],
}

# HARD PROHIBITED risks per area type (blacklist - NEVER mention these)
AREA_PROHIBITED_RISKS: Dict[AreaType, List[str]] = {
    AreaType.URBAN_DENSE: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "forest_regeneration", "timber_salvage", "wildlife_corridor",
        "grazing_impact", "crop_damage", "fish_passage",
    ],
    
    AreaType.URBAN_RESIDENTIAL: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "timber_salvage", "forest_regeneration", "fuel_load",
        "grazing_impact", "crop_damage", "hazardous_materials",
    ],
    
    AreaType.SUBURBAN: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "timber_salvage", "forest_regeneration",
        "hazardous_materials", "containment_breach",
        "grazing_impact", "fish_passage",
    ],
    
    AreaType.RURAL_RESIDENTIAL: [
        # WUI areas may mention some forest hazards IF vegetation present
        "timber_salvage", "containment_breach",
        "fish_passage", "crop_damage",
    ],
    
    AreaType.FOREST_NATURAL: [
        "structural_damage", "business_interruption",
        "hazardous_materials", "crop_damage", "irrigation_infrastructure",
    ],
    
    AreaType.AGRICULTURAL: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "forest_regeneration", "timber_salvage",
        "hazardous_materials", "containment_breach",
    ],
    
    AreaType.INDUSTRIAL: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "forest_regeneration", "timber_salvage", "grazing_impact",
        "wildlife_corridor", "fish_passage",
    ],
    
    AreaType.WETLAND: [
        "widowmakers", "structural_damage", "hazardous_materials",
        "crop_damage", "grazing_impact", "business_interruption",
    ],
    
    AreaType.COMMERCIAL: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "forest_regeneration", "timber_salvage",
        "wildlife_corridor", "crop_damage", "grazing_impact",
    ],
    
    AreaType.SHRUBLAND: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "structural_damage", "hazardous_materials", "crop_damage",
    ],
    
    AreaType.GRASSLAND: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "canopy_damage", "forest_regeneration", "timber_salvage",
        "structural_damage", "hazardous_materials",
    ],
    
    AreaType.MIXED_USE: [
        "widowmakers", "standing_dead_trees", "snag_hazard",
        "timber_salvage", "fish_passage",
    ],
    
    AreaType.UNDEVELOPED: [
        "structural_damage", "business_interruption",
        "hazardous_materials", "crop_damage",
    ],
}


def is_risk_permitted(risk: str, area_type: AreaType) -> Tuple[bool, str]:
    """
    Determine if a risk topic is permitted for an area type.
    
    Returns:
        Tuple of (is_permitted, reason)
    """
    prohibited = AREA_PROHIBITED_RISKS.get(area_type, [])
    eligible = AREA_ELIGIBLE_RISKS.get(area_type, [])
    
    # Check hard prohibition first
    if risk.lower() in [p.lower() for p in prohibited]:
        return False, f"HARD PROHIBITED for {area_type.value}"
    
    # Check eligibility
    if risk.lower() in [e.lower() for e in eligible]:
        return True, f"Eligible for {area_type.value}"
    
    # Default: not permitted (conservative)
    return False, f"Not in eligibility list for {area_type.value}"


# =============================================================================
# RAG 2: EVIDENCE-TO-HAZARD MAPPING RULE (REFINED)
# =============================================================================

class EvidenceAnchorRAG:
    """
    RAG: Evidence-to-Hazard Mapping Rule
    =====================================
    
    REFINED PRINCIPLE: Dual-gate system
    1. JSON field must exist AND have sufficient value
    2. Topic must pass area-type eligibility
    
    No evidence → SILENCE (not even "may" statements)
    """
    
    # Map JSON fields to permissible risk topics
    # Format: field_name -> (topics, minimum_value_to_mention)
    EVIDENCE_RISK_MAPPING: Dict[str, Tuple[List[str], float]] = {
        "burn_severity_mean": (
            ["fire_damage", "vegetation_loss", "soil_damage"],
            0.10  # Minimum 10% to even mention
        ),
        "burn_severity_high_ratio": (
            ["severe_damage_zones", "recovery_timeline", "regeneration_potential"],
            0.05  # Minimum 5% high severity
        ),
        "vegetation_cover_percent": (
            ["canopy_loss", "habitat_impact", "vegetation_damage"],
            0.15  # Minimum 15% vegetation to discuss
        ),
        "urban_percentage": (
            ["structural_exposure", "infrastructure_risk", "defensible_space"],
            0.10  # Minimum 10% urban
        ),
        "slope_degrees": (
            ["erosion_risk", "access_difficulty"],
            15.0  # Minimum 15 degrees to mention slope issues
        ),
        "water_body_present": (
            ["watershed_impact", "water_quality", "riparian_damage"],
            0.0  # Boolean - if True, can discuss
        ),
        "structures_detected": (
            ["property_damage", "defensible_space", "structural_damage"],
            0.0  # Boolean - if True, can discuss
        ),
        "road_network_present": (
            ["evacuation_routes", "access", "emergency_access"],
            0.0  # Boolean
        ),
        "power_infrastructure": (
            ["utility_disruption"],
            0.0  # Boolean
        ),
        "carbon_stock_estimate": (
            ["carbon_loss", "sequestration_potential"],
            100.0  # Minimum 100 tons to discuss carbon
        ),
    }
    
    @classmethod
    def get_permitted_topics(
        cls,
        layer2_json: Dict[str, Any],
        area_type: AreaType
    ) -> List[str]:
        """
        Given JSON fields and area type, return permitted topics.
        
        Applies DUAL-GATE:
        1. Evidence gate (field exists with sufficient value)
        2. Area eligibility gate
        """
        permitted = set()
        
        for field, (topics, min_value) in cls.EVIDENCE_RISK_MAPPING.items():
            value = layer2_json.get(field)
            
            if value is None:
                continue  # No evidence - skip entirely
            
            # Check value threshold
            if isinstance(value, bool):
                if not value:
                    continue
            elif isinstance(value, (int, float)):
                if value < min_value:
                    continue
            
            # Evidence gate passed - now check area eligibility
            for topic in topics:
                is_ok, _ = is_risk_permitted(topic, area_type)
                if is_ok:
                    permitted.add(topic)
        
        return list(permitted)
    
    @classmethod
    def get_evidence_citation(cls, topic: str, layer2_json: Dict[str, Any]) -> Optional[str]:
        """
        Get the JSON field citation for a given topic.
        
        Returns None if topic has no evidence support.
        """
        for field, (topics, _) in cls.EVIDENCE_RISK_MAPPING.items():
            if topic in topics and field in layer2_json:
                value = layer2_json[field]
                if isinstance(value, float):
                    return f"({field}: {value:.1%})"
                elif isinstance(value, bool):
                    return f"({field}: {'Yes' if value else 'No'})"
                else:
                    return f"({field}: {value})"
        return None


# =============================================================================
# RAG 3: PROFESSIONAL PLAUSIBILITY FILTER (REFINED)
# =============================================================================

PROFESSIONAL_PLAUSIBILITY_RULES = """
RAG: Professional Plausibility Filter v2.0
===========================================

PURPOSE: Enforce the "licensed professional" standard for all statements.

DECISION CHECKPOINT - Before ANY risk statement, verify:

    ┌─────────────────────────────────────────────────────┐
    │  PLAUSIBILITY GATE                                   │
    │                                                       │
    │  Q1: Does JSON contain evidence for this claim?       │
    │      NO  → REJECT (do not mention)                   │
    │      YES → Continue                                   │
    │                                                       │
    │  Q2: Is this risk ELIGIBLE for this area type?       │
    │      NO  → REJECT (hard block)                       │
    │      YES → Continue                                   │
    │                                                       │
    │  Q3: Is evidence value above mention threshold?       │
    │      NO  → REJECT (below materiality)                │
    │      YES → Continue                                   │
    │                                                       │
    │  Q4: Would a PE/licensed professional mention this?  │
    │      NO  → REJECT (not professionally relevant)      │
    │      YES → ALLOW (with citation)                     │
    └─────────────────────────────────────────────────────┘

EXPLICIT REJECTIONS (HARD BLOCKS):

1. GENERIC SAFETY LISTS
   - "Standard wildfire hazards include..."
   - "Common post-fire risks are..."
   - "Typical concerns would be..."
   ↳ BLOCK: No site-specific value

2. KNOWLEDGE-BASED HAZARDS
   - Risks from training data, not from JSON
   - "Widowmakers" without canopy_damage > 40% AND forest area type
   - Industrial risks without industrial area type
   ↳ BLOCK: Evidence-free speculation

3. WORST-CASE FRAMING
   - "In extreme scenarios..."
   - "If conditions worsen..."
   - "Could potentially lead to..."
   ↳ BLOCK: Unless JSON shows values > 0.70

4. CONFIDENCE ESCALATION
   - Treating 0.2 confidence as concern
   - "While uncertain, there may be..."
   ↳ BLOCK: Low confidence requires silence or explicit uncertainty

5. TEMPORAL SPECULATION
   - Future projections without baseline
   - "Will likely result in..."
   ↳ BLOCK: Use "based on current data" framing only
"""


# =============================================================================
# RAG 4: MATERIALITY & PROPORTIONALITY CONTROLLER (REFINED)
# =============================================================================

@dataclass
class SeverityThresholds:
    """
    Thresholds for scaling response severity to evidence.
    
    REFINED: Added SILENCE threshold and tightened boundaries.
    """
    # Below this: ABSOLUTE SILENCE on the topic
    SILENCE_THRESHOLD: float = 0.10
    
    # 0.10-0.20: May acknowledge IF directly asked, with heavy qualification
    MINIMAL_THRESHOLD: float = 0.20
    
    # 0.20-0.40: Minor/limited language only
    MINOR_THRESHOLD: float = 0.40
    
    # 0.40-0.60: Standard professional language
    MODERATE_THRESHOLD: float = 0.60
    
    # 0.60-0.80: Notable/significant language
    SIGNIFICANT_THRESHOLD: float = 0.80
    
    # Above 0.80: Critical/urgent language permitted
    CRITICAL_THRESHOLD: float = 0.80


class MaterialityRAG:
    """
    RAG: Materiality & Proportionality Controller v2.0
    ===================================================
    
    REFINED PRINCIPLE: 
    - Evidence strength DIRECTLY maps to language strength
    - No inflation permitted
    - Silence is the default for low evidence
    """
    
    SEVERITY_LANGUAGE_MAP = {
        # level: (permitted_adjectives, permitted_verbs, framing_rule)
        "silence": (
            [],
            [],
            "DO NOT MENTION - below materiality threshold"
        ),
        "minimal": (
            ["slight", "trace", "negligible"],
            ["may show"],
            "Mention ONLY if directly asked; heavy qualification required"
        ),
        "minor": (
            ["minor", "limited", "localized"],
            ["indicates", "suggests"],
            "Brief acknowledgment with qualifying language"
        ),
        "moderate": (
            ["moderate", "measurable", "observable"],
            ["shows", "demonstrates"],
            "Standard professional language; cite evidence"
        ),
        "significant": (
            ["substantial", "considerable", "elevated"],
            ["exhibits", "presents"],
            "Emphasize appropriately; recommend action"
        ),
        "critical": (
            ["severe", "extensive", "critical"],
            ["requires immediate", "demands"],
            "Urgent framing justified; action imperative"
        ),
    }
    
    @staticmethod
    def get_severity_level(value: float) -> str:
        """Map a 0-1 value to severity level with refined thresholds."""
        t = SeverityThresholds()
        
        if value < t.SILENCE_THRESHOLD:
            return "silence"
        elif value < t.MINIMAL_THRESHOLD:
            return "minimal"
        elif value < t.MINOR_THRESHOLD:
            return "minor"
        elif value < t.MODERATE_THRESHOLD:
            return "moderate"
        elif value < t.SIGNIFICANT_THRESHOLD:
            return "significant"
        else:
            return "critical"
    
    @staticmethod
    def get_language_constraints(severity_level: str) -> Dict[str, Any]:
        """Return language constraints for a severity level."""
        if severity_level not in MaterialityRAG.SEVERITY_LANGUAGE_MAP:
            return {
                "adjectives": [],
                "verbs": [],
                "framing_guidance": "DO NOT MENTION",
            }
        
        adjectives, verbs, framing = MaterialityRAG.SEVERITY_LANGUAGE_MAP[severity_level]
        return {
            "adjectives": adjectives,
            "verbs": verbs,
            "framing_guidance": framing,
            "is_mentionable": severity_level != "silence",
        }


# =============================================================================
# RAG 5: CARBON REASONING SCOPE CONTROLLER (REFINED)
# =============================================================================

CARBON_REASONING_RAG = """
RAG: Carbon Reasoning Scope Controller v2.0
=============================================

PURPOSE: Constrain carbon claims to site-specific, quantified statements.

ENTRY GATE:
Before ANY carbon statement, verify:
- JSON contains "carbon_stock_estimate" OR "area_km2" with vegetation data
- Value is >= 100 tons OR area >= 0.5 km² with forest/vegetation cover
- If NEITHER present → NO carbon statements permitted

PERMITTED CARBON TOPICS (with required evidence):

1. SITE-LEVEL CARBON STOCK
   Required: carbon_stock_estimate field
   Format: "Estimated carbon stock of X tons (site data)"
   
2. SEQUESTRATION POTENTIAL
   Required: vegetation_cover_percent > 20% AND area_km2
   Format: "Potential sequestration of X tons/year based on Y% vegetation cover"

3. CARBON LOSS FROM FIRE
   Required: burn_severity_mean > 0.3 AND (carbon_stock_estimate OR vegetation_cover)
   Format: "Estimated carbon release of X tons from burned biomass"

HARD PROHIBITIONS:

❌ "Contributes to climate change" - narrative drift
❌ "Carbon footprint" without quantification
❌ "Significant carbon impact" without numbers
❌ "Climate implications" - scope creep
❌ Carbon offset recommendations
❌ Comparisons to other sites without data
❌ Future projections beyond 5 years
❌ Policy recommendations

REQUIRED FORMAT FOR ALL CARBON STATEMENTS:

[Quantity] + [Unit] + [Uncertainty] + [Citation]

Example: "Estimated 15,000 ± 3,000 tons CO₂ (carbon_stock_estimate: 0.73)"

Statements without this format → REJECT
"""


# =============================================================================
# RAG 6: ABSENCE-OF-EVIDENCE NEUTRALITY (REFINED)
# =============================================================================

NEUTRALITY_RAG = """
RAG: Absence-of-Evidence Neutrality Rule v2.0
==============================================

PURPOSE: Enforce silence over speculation for missing data.

CORE PRINCIPLE:

    ┌─────────────────────────────────────────────────────┐
    │  ABSENCE OF EVIDENCE ≠ EVIDENCE OF ANYTHING        │
    │                                                     │
    │  Missing data → SILENCE                            │
    │  Low confidence → EXPLICIT UNCERTAINTY             │
    │  Outdated data → TEMPORAL CAVEAT                   │
    └─────────────────────────────────────────────────────┘

DECISION TREE FOR MISSING DATA:

    Is field present in JSON?
    ├── NO → SILENCE on related topics
    │         (do not say "unable to determine" unless asked)
    └── YES → Is value null/NaN?
              ├── YES → SILENCE
              └── NO → Is confidence >= 0.3?
                        ├── NO → Use heavy uncertainty language
                        └── YES → Is data < 30 days old?
                                  ├── NO → Add temporal caveat
                                  └── YES → Normal processing

HARD PROHIBITED PHRASES (ABSOLUTE BLOCK):

❌ "There could be..."
❌ "It's possible that..."
❌ "In similar areas..."
❌ "Typically, this would..."
❌ "Based on general knowledge..."
❌ "Research suggests..."  (without RAG citation)
❌ "Studies have shown..." (without RAG citation)
❌ "One might expect..."
❌ "It's reasonable to assume..."

PERMITTED PHRASES (use sparingly):

✓ "[Silence - just don't mention it]"
✓ "Data not available" (only if user specifically asks)
✓ "Cannot assess from available imagery"
✓ "Requires field verification"

CONFIDENCE-BASED LANGUAGE GATES:

Confidence < 0.3:
→ "Preliminary data suggests..." with explicit "(confidence: low)"
→ OR complete silence

Confidence 0.3-0.6:
→ Standard professional language with "(moderate confidence)"

Confidence > 0.6:
→ Direct statements permitted
"""


# =============================================================================
# RAG 7: TEMPORAL VALIDITY CONSTRAINT (NEW)
# =============================================================================

TEMPORAL_VALIDITY_RAG = """
RAG: Temporal Validity Constraint
==================================

PURPOSE: Prevent stale data from generating false confidence.

DATA AGE GATES:

0-7 days: 
→ Full confidence permitted
→ No temporal caveats required

8-30 days:
→ Standard statements with note: "Based on imagery from [date]"
→ Dynamic conditions (e.g., active fires) require explicit caveat

31-90 days:
→ All statements require: "Note: Data is [X] days old"
→ Dynamic indicators (weather, fire progression) → silenceCHECK

> 90 days:
→ Prefix all statements with temporal warning
→ Recommend updated assessment
→ Do NOT make claims about current conditions

HARD PROHIBITION:

   Present-tense claims with data > 30 days old
   Wrong: "The site shows high burn severity"
   Right: "As of [date], the site showed high burn severity"
"""


# =============================================================================
# COMPOSITE REASONING CHAIN (REFINED)
# =============================================================================

def build_reasoning_chain(
    area_context: AreaContext,
    layer2_json: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the complete RAG reasoning chain for a given context.
    
    REFINED: Stricter filtering, explicit prohibitions, no fallbacks.
    """
    
    area_type = area_context.area_type
    
    # Step 1: Get area-eligible and prohibited risks
    eligible_risks = AREA_ELIGIBLE_RISKS.get(area_type, [])
    prohibited_risks = AREA_PROHIBITED_RISKS.get(area_type, [])
    
    # Step 2: Evidence-anchored permitted topics (with dual-gate)
    permitted_topics = EvidenceAnchorRAG.get_permitted_topics(layer2_json, area_type)
    
    # Step 3: Determine severity constraints with SILENCE enforcement
    severity_constraints = {}
    mention_allowed = {}
    
    for field, value in layer2_json.items():
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            severity = MaterialityRAG.get_severity_level(value)
            constraints = MaterialityRAG.get_language_constraints(severity)
            severity_constraints[field] = constraints
            mention_allowed[field] = constraints.get("is_mentionable", False)
    
    # Step 4: Check temporal validity
    data_age = area_context.data_age_days
    if data_age > 90:
        temporal_warning = "STALE DATA WARNING: Data is >90 days old. Recommend fresh assessment."
    elif data_age > 30:
        temporal_warning = f"Data age caveat required: {data_age} days old."
    else:
        temporal_warning = None
    
    # Step 5: Build constraint package
    return {
        "area_type": area_type.value,
        "eligible_risk_categories": eligible_risks,
        "prohibited_risk_categories": prohibited_risks,
        "permitted_topics": permitted_topics,
        "severity_constraints": severity_constraints,
        "mention_allowed": mention_allowed,
        "confidence_level": area_context.confidence_level,
        "temporal_warning": temporal_warning,
        "data_age_days": data_age,
    }


def format_reasoning_prompt(constraint_package: Dict[str, Any]) -> str:
    """
    Format the constraint package as a system prompt injection.
    
    REFINED: More explicit constraints, prohibition emphasis.
    """
    
    prohibited_str = ', '.join(constraint_package.get('prohibited_risk_categories', [])[:10])
    permitted_str = ', '.join(constraint_package.get('permitted_topics', []))
    eligible_str = ', '.join(constraint_package.get('eligible_risk_categories', [])[:8])
    
    temporal_block = ""
    if constraint_package.get('temporal_warning'):
        temporal_block = f"\n {constraint_package['temporal_warning']}\n"
    
    confidence = constraint_package.get('confidence_level', 0.5)
    confidence_instruction = ""
    if confidence < 0.3:
        confidence_instruction = "\n LOW CONFIDENCE DATA: All statements require uncertainty qualifiers.\n"
    
    return f"""
# COGNITIVE REASONING CONSTRAINTS (MANDATORY)

You are operating under professional reasoning constraints.
Violations will produce unacceptable output.
{temporal_block}{confidence_instruction}
## Area Context: {constraint_package['area_type']}

## ELIGIBLE Risk Categories (whitelist)
You may ONLY discuss risks in these categories IF they have evidence:
{eligible_str}

## HARD PROHIBITED Topics (blacklist)
You MUST NEVER mention these regardless of any other consideration:
{prohibited_str}

## Evidence-Backed Permitted Topics
These specific topics passed both evidence AND area gates:
{permitted_str}

## Severity Language Rules (STRICT)
Match language strength EXACTLY to evidence strength:
- Values < 0.10: SILENCE - do not mention
- Values 0.10-0.20: Only if asked, with "negligible" language
- Values 0.20-0.40: "minor", "limited" only
- Values 0.40-0.60: Standard professional language
- Values 0.60-0.80: "notable", "significant" permitted
- Values > 0.80: "critical", "urgent" permitted

## Operating Principles (NON-NEGOTIABLE)
1. NO generic hazard lists from training knowledge
2. NO mention of prohibited topics regardless of JSON
3. NO worst-case speculation without values > 0.70
4. NO confidence escalation (low values stay low)
5. SILENCE over speculation for missing data
6. CITE JSON field for every factual claim
7. NO present-tense claims for data > 30 days old

Data confidence: {constraint_package['confidence_level']:.0%}
Data age: {constraint_package.get('data_age_days', 0)} days
"""


# =============================================================================
# INTEGRATION FUNCTION (REFINED)
# =============================================================================

def _safe_get_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safe extraction of float values dealing with potential dicts or None."""
    val = data.get(key, default)
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    # If it's a dict (e.g. from complex JSON), try to extract 'value' or just default
    if isinstance(val, dict):
        return float(val.get('value', default))
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def apply_reasoning_guardrails(
    area_type: str,
    layer2_json: Dict[str, Any],
    data_age_days: int = 0
) -> str:
    """
    Main integration function for applying RAG guardrails.
    
    REFINED: No permissive fallbacks, stricter type handling.
    """
    # Parse area type - STRICT matching
    area_enum = None
    for at in AreaType:
        if at.value == area_type.lower().replace(" ", "_"):
            area_enum = at
            break
    
    
    # If no match, use most conservative option (not mixed-use)
    if area_enum is None:
        # Log warning - this should be investigated
        print(f"   [GUARDRAILS WARNING] Unknown area type: {area_type}, defaulting to UNDEVELOPED")
        area_enum = AreaType.UNDEVELOPED
    
    # Sanitize inputs to ensure all numeric fields are actual floats, not dicts
    # This prevents complex objects from bypassing threshold checks or causing storage errors
    sanitized_json = layer2_json.copy()
    numeric_fields = [
        "burn_severity_mean", "burn_severity_high_ratio", 
        "vegetation_cover_percent", "urban_percentage", "slope_degrees",
        "carbon_stock_estimate", "overall_confidence"
    ]
    for field in numeric_fields:
        if field in sanitized_json:
            sanitized_json[field] = _safe_get_float(sanitized_json, field)
            
    # Build area context from JSON evidence with SAFE EXTRACTION
    area_context = AreaContext(
        area_type=area_enum,
        has_vegetation=sanitized_json.get("vegetation_cover_percent", 0) > 15,
        has_structures=bool(sanitized_json.get("structures_detected", False)),
        has_water_bodies=bool(sanitized_json.get("water_body_present", False)),
        population_density=_infer_population_density(sanitized_json),
        infrastructure_present=sanitized_json.get("infrastructure", []),
        confidence_level=sanitized_json.get("overall_confidence", 0.5),
        data_age_days=data_age_days,
    )
    
    # Build constraint package
    constraints = build_reasoning_chain(area_context, sanitized_json)
    
    # Format as prompt
    return format_reasoning_prompt(constraints)


def _infer_population_density(layer2_json: Dict[str, Any]) -> str:
    """Infer population density from urban percentage."""
    urban = _safe_get_float(layer2_json, "urban_percentage", 0)
    if urban > 60:
        return "high"
    elif urban > 30:
        return "medium"
    elif urban > 10:
        return "low"
    else:
        return "none"
