"""
EcoRevive Constraints Module
=============================
Multi-layered safety constraints to prevent illegal or impossible
restoration recommendations.

Layer 1: Hard-coded rules (pre-LLM filtering)
Layer 2: LLM system prompts with guardrails

This module ensures that:
- Non-native species are never recommended
- Protected areas receive appropriate treatment
- Legal requirements are always considered
- Indigenous land rights are respected
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ProtectedStatus(Enum):
    """Land protection status."""
    NONE = "none"
    NATIONAL_PARK = "national_park"
    WILDERNESS = "wilderness"
    NATIONAL_MONUMENT = "national_monument"
    STATE_PARK = "state_park"
    WILDLIFE_REFUGE = "wildlife_refuge"
    CRITICAL_HABITAT = "critical_habitat"
    PRIVATE_CONSERVATION = "private_conservation"


class SoilType(Enum):
    """Soil classifications affecting restoration options."""
    SANDY = "sandy"
    LOAM = "loam"
    CLAY = "clay"
    ROCKY = "rocky"
    SERPENTINE = "serpentine"
    VOLCANIC = "volcanic"
    WETLAND = "wetland"


@dataclass
class SiteConditions:
    """
    Site conditions that affect what restoration actions are legal/possible.
    
    This is used for Layer 1 hard-coded rule filtering.
    """
    # Location
    latitude: float = 0.0
    longitude: float = 0.0
    elevation_m: float = 0.0
    
    # Protection status
    protected_status: ProtectedStatus = ProtectedStatus.NONE
    land_owner: str = "unknown"  # "federal", "state", "private", "tribal"
    managing_agency: Optional[str] = None
    
    # Physical conditions
    slope_degrees: float = 0.0
    aspect: Optional[str] = None  # N, S, E, W, NE, NW, SE, SW
    soil_type: SoilType = SoilType.LOAM
    
    # Ecological conditions
    burn_severity: float = 0.0  # 0-1
    existing_vegetation: List[str] = field(default_factory=list)
    invasive_species_present: List[str] = field(default_factory=list)
    
    # Hydrology
    near_water_body: bool = False
    in_floodplain: bool = False
    wetland_present: bool = False
    
    # Legal
    has_restoration_permit: bool = False
    tribal_consultation_completed: bool = False


@dataclass
class RestorationAction:
    """A proposed restoration action."""
    name: str
    description: str
    species: Optional[str] = None
    method: Optional[str] = None
    allowed: bool = True
    warnings: List[str] = field(default_factory=list)
    required_permits: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None


class ConstraintChecker:
    """
    Layer 1: Hard-coded rules for pre-LLM filtering.
    
    Applies ecological and legal constraints before any LLM reasoning.
    """
    
    # Non-native species that should NEVER be recommended
    PROHIBITED_SPECIES = {
        # Invasive trees
        "Eucalyptus", "Eucalyptus globulus", "Blue Gum",
        "Ailanthus altissima", "Tree of Heaven",
        "Tamarix", "Salt Cedar",
        "Washingtonia", "Palm",  # Wrong climate
        
        # Invasive shrubs
        "Cytisus scoparius", "Scotch Broom",
        "Genista monspessulana", "French Broom",
        "Spartium junceum", "Spanish Broom",
        "Ulex europaeus", "Gorse",
        
        # Invasive grasses
        "Arundo donax", "Giant Reed",
        "Pennisetum setaceum", "Fountain Grass",
    }
    
    # Actions not allowed in certain protected areas
    PROTECTED_AREA_RESTRICTIONS = {
        ProtectedStatus.WILDERNESS: {
            "allowed": ["natural_regeneration", "invasive_removal_manual", "monitoring"],
            "blocked": ["planting", "machinery", "herbicide", "road_construction"],
            "reason": "Wilderness areas require minimal intervention per Wilderness Act"
        },
        ProtectedStatus.NATIONAL_PARK: {
            "allowed": ["natural_regeneration", "native_planting_approved", "invasive_removal", "monitoring"],
            "blocked": ["non_native_species", "machinery_heavy", "commercial_activity"],
            "reason": "National Parks require NPS approval for active restoration"
        },
        ProtectedStatus.CRITICAL_HABITAT: {
            "allowed": ["habitat_enhancement", "invasive_removal", "native_planting"],
            "blocked": ["actions_affecting_listed_species"],
            "reason": "Critical Habitat requires ESA Section 7 consultation"
        }
    }
    
    def __init__(self):
        """Initialize the constraint checker."""
        pass
    
    def check_site_constraints(
        self,
        site: SiteConditions,
        proposed_actions: List[RestorationAction]
    ) -> Tuple[List[RestorationAction], List[str]]:
        """
        Apply all hard-coded constraints to proposed actions.
        
        Args:
            site: Site conditions
            proposed_actions: List of proposed restoration actions
            
        Returns:
            Tuple of (filtered_actions, global_warnings)
        """
        filtered_actions = []
        global_warnings = []
        
        for action in proposed_actions:
            action = self._check_species_constraint(action)
            action = self._check_protected_area_constraint(action, site)
            action = self._check_slope_constraint(action, site)
            action = self._check_soil_constraint(action, site)
            action = self._check_hydrology_constraint(action, site)
            action = self._check_permit_constraint(action, site)
            action = self._check_tribal_constraint(action, site)
            
            filtered_actions.append(action)
        
        # Add global warnings based on site conditions
        if site.protected_status != ProtectedStatus.NONE:
            global_warnings.append(
                f"[WARNING] Site is in {site.protected_status.value.replace('_', ' ').title()}. "
                f"Special restrictions apply."
            )
        
        if site.slope_degrees > 30:
            global_warnings.append(
                f"[WARNING] Steep slope ({site.slope_degrees}°). Erosion control measures required."
            )
        
        if site.land_owner == "tribal":
            global_warnings.append(
                "[WARNING] Tribal land. Free, Prior, and Informed Consent (FPIC) required."
            )
        
        if site.burn_severity > 0.7:
            global_warnings.append(
                "[WARNING] High burn severity. Natural regeneration may be limited. "
                "Check for surviving seed sources."
            )
        
        return filtered_actions, global_warnings
    
    def _check_species_constraint(self, action: RestorationAction) -> RestorationAction:
        """Block non-native/invasive species."""
        if action.species and action.species in self.PROHIBITED_SPECIES:
            action.allowed = False
            action.blocked_reason = (
                f"BLOCKED: {action.species} is non-native or invasive. "
                f"Only California native species may be recommended."
            )
        return action
    
    def _check_protected_area_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Apply protected area restrictions."""
        if site.protected_status in self.PROTECTED_AREA_RESTRICTIONS:
            restrictions = self.PROTECTED_AREA_RESTRICTIONS[site.protected_status]
            
            if action.method in restrictions.get("blocked", []):
                action.allowed = False
                action.blocked_reason = (
                    f"BLOCKED: {action.method} not allowed in "
                    f"{site.protected_status.value}. {restrictions['reason']}"
                )
        return action
    
    def _check_slope_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Add warnings for steep slopes."""
        if site.slope_degrees > 30:
            if action.method == "deep_plowing":
                action.allowed = False
                action.blocked_reason = (
                    "BLOCKED: Deep plowing on slopes >30° causes severe erosion risk."
                )
            elif action.method in ["planting", "seeding"]:
                action.warnings.append(
                    f"Slope is {site.slope_degrees}°. Terracing or contour planting required."
                )
                action.warnings.append(
                    "Consider erosion control measures (mulching, wattles, check dams)."
                )
        return action
    
    def _check_soil_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Apply soil-based constraints."""
        if site.soil_type == SoilType.ROCKY:
            if action.method == "deep_plowing":
                action.allowed = False
                action.blocked_reason = "BLOCKED: Deep plowing not possible in rocky soil."
            action.warnings.append("Rocky soil limits root development. Choose adapted species.")
        
        if site.soil_type == SoilType.SERPENTINE:
            action.warnings.append(
                "Serpentine soil is toxic to most plants. "
                "Use specialized serpentine-adapted species only."
            )
        
        return action
    
    def _check_hydrology_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Apply hydrology-based constraints."""
        if site.near_water_body:
            action.warnings.append(
                "Near water body. California Clean Water Act restrictions apply. "
                "Maintain minimum 30m buffer."
            )
            if action.method == "herbicide":
                action.allowed = False
                action.blocked_reason = (
                    "BLOCKED: Herbicide use restricted near water bodies. "
                    "Use manual removal methods."
                )
        
        if site.in_floodplain:
            action.warnings.append(
                "In floodplain. Plant flood-tolerant species. "
                "Avoid permanent structures."
            )
        
        return action
    
    def _check_permit_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Check permit requirements."""
        if site.land_owner == "federal" and not site.has_restoration_permit:
            action.required_permits.append("USFS Special Use Permit or Volunteer Agreement")
        
        if site.land_owner == "state" and not site.has_restoration_permit:
            action.required_permits.append("California State Parks Authorization")
        
        if site.protected_status == ProtectedStatus.CRITICAL_HABITAT:
            action.required_permits.append("ESA Section 7 Consultation (USFWS)")
        
        return action
    
    def _check_tribal_constraint(
        self,
        action: RestorationAction,
        site: SiteConditions
    ) -> RestorationAction:
        """Ensure tribal consultation is respected."""
        if site.land_owner == "tribal" and not site.tribal_consultation_completed:
            action.warnings.append(
                "IMPORTANT: This is tribal land. Free, Prior, and Informed Consent (FPIC) "
                "from the relevant tribe is required before any restoration activities."
            )
        return action
    
    def filter_allowed_actions(
        self,
        actions: List[RestorationAction]
    ) -> List[RestorationAction]:
        """Return only allowed actions."""
        return [a for a in actions if a.allowed]
    
    def get_blocked_actions(
        self,
        actions: List[RestorationAction]
    ) -> List[RestorationAction]:
        """Return blocked actions with reasons."""
        return [a for a in actions if not a.allowed]


# ============================================================
# LAYER 2: LLM SYSTEM PROMPTS WITH GUARDRAILS
# ============================================================

RESTORATION_SYSTEM_PROMPT = """You are an expert ecological restoration advisor for California ecosystems. You provide science-based, legally compliant restoration recommendations.

# CRITICAL CONSTRAINTS (You MUST follow these)

## You MUST NOT recommend:
1. **Non-native species** - Only California native species are acceptable
2. **Illegal actions** - No activities that violate federal, state, or local laws
3. **Actions in protected areas without justification** - National Parks, Wilderness Areas, and Critical Habitat have special restrictions
4. **Methods that harm existing ecosystems** - Preservation of existing native vegetation is priority
5. **Ignoring indigenous land rights** - Tribal consultation is required for ancestral lands
6. **Commercially-motivated recommendations** - Focus on ecological outcomes, not commercial interests

## You MUST always:
1. **Cite relevant laws or policies** when making recommendations
2. **Disclose uncertainty** - If you're not sure, say so
3. **Consider climate change** - Recommend species adapted to future conditions
4. **Prioritize natural regeneration** when conditions allow
5. **Recommend permits** when required for the land ownership type
6. **Respect cultural resources** - Archaeological and cultural sites must be protected

## Legal frameworks to reference:
- **National Environmental Policy Act (NEPA)** - Federal projects
- **California Environmental Quality Act (CEQA)** - State projects
- **Endangered Species Act (ESA)** - Listed species and critical habitat
- **Clean Water Act (CWA)** - Wetlands and water bodies
- **Wilderness Act** - Wilderness areas
- **National Historic Preservation Act** - Cultural resources

## If asked to do something prohibited:
- Politely explain WHY it's not allowed
- Suggest a legal/ecological alternative
- Cite the relevant law or ecological principle

## Response format:
- Be specific and actionable
- Include species scientific names
- Provide planting densities when relevant
- Note required permits
- Include monitoring recommendations
"""

SPECIES_RECOMMENDATION_PROMPT = """You are recommending native plant species for restoration in California. 

# CRITICAL RULES:
1. ONLY recommend California native species
2. Match species to site conditions (elevation, soil, precipitation)
3. Consider fire adaptation for fire-prone areas
4. Prioritize local genotypes when available
5. Include both early successional and climax species
6. Consider wildlife value

# NEVER recommend:
- Eucalyptus (any species) - invasive
- Palm trees - wrong ecosystem
- Scotch Broom, French Broom - highly invasive
- Tree of Heaven - invasive
- Any species not native to California

# Always include:
- Scientific name
- Common name
- Fire response (resprouter vs seeder)
- Planting priority (1-3)
- Typical survival rate
- Source considerations (local genotype preferred)
"""

LEGAL_COMPLIANCE_PROMPT = """You are providing legal and regulatory guidance for ecosystem restoration in California.

# Key regulations to consider:

## International Frameworks (for context):
- UN Convention on Biological Diversity - 30% restoration by 2030
- UN Decade on Ecosystem Restoration - 2021-2030
- Paris Agreement - Nature-based climate solutions

## Federal:
- NEPA (National Environmental Policy Act) - EIS/EA requirements
- ESA (Endangered Species Act) - Section 7 consultation for federal actions
- CWA (Clean Water Act) - Section 404 permits for wetlands
- NHPA (National Historic Preservation Act) - Section 106 for cultural resources
- Wilderness Act - Special restrictions in wilderness areas

## California State:
- CEQA (California Environmental Quality Act) - Environmental review
- California Endangered Species Act (CESA) - State-listed species
- Porter-Cologne Water Quality Control Act - State water protection
- Lake and Streambed Alteration Agreement (1602) - Stream modifications
- California 30x30 Initiative - Conserve 30% by 2030

## Land-specific:
- National Forest: USFS Special Use Permits
- BLM: Right-of-Way permits
- State Parks: Resource Management authorization
- Private: County permits may apply
- Tribal: FPIC required (see below)

# Always note:
- Required permits for the proposed activity
- Timing restrictions (nesting seasons, etc.)
- Consultation requirements (tribes, agencies)
- Monitoring and reporting requirements
"""

ETHICAL_SYSTEM_PROMPT = """You are an ethical AI system providing ecosystem restoration guidance. You must prioritize human rights, community welfare, and ecological integrity above all else.

# ETHICAL IMPERATIVES (Non-negotiable)

## 1. Indigenous Land Rights
**Free, Prior, and Informed Consent (FPIC) is REQUIRED**
- NEVER recommend interventions on indigenous/tribal lands without explicit community consent
- FLAG all projects in or near indigenous territories
- DEFAULT to community-led restoration, not top-down planning
- RESPECT data sovereignty - do not collect data without permission
- ACKNOWLEDGE traditional ecological knowledge as valid expertise

## 2. AI Limitations and Humility
**This is DECISION SUPPORT, not autonomous decision-making**
- ALWAYS acknowledge uncertainty
- NEVER claim perfect accuracy
- RECOMMEND expert validation before implementation
- DEFER to local knowledge when it conflicts with model outputs
- DISCLOSE confidence levels for all predictions

## 3. Bias Awareness
**Acknowledge data and model limitations**
- Geographic bias: High-res data may be limited in some areas
- Temporal bias: Data coverage varies by time period
- Classification bias: Models trained on limited ecosystem types
- Cloud cover bias: Some regions harder to observe
- Always note when data quality may affect recommendations

## 4. Misuse Prevention
**Prevent harmful applications**
- NEVER recommend actions that would displace communities
- REFUSE to support greenwashing or false restoration claims
- REJECT recommendations that prioritize carbon credits over biodiversity
- REFUSE to support land grabs disguised as conservation
- TRANSPARENT about all assumptions and trade-offs

## 5. Community Welfare
**Local communities must benefit from restoration**
- RECOMMEND local employment and training
- SUGGEST benefit-sharing mechanisms
- CONSIDER livelihoods and food security
- RESPECT cultural and spiritual connections to land

## 6. Ecological Integrity
**Do no harm to existing ecosystems**
- PRIORITIZE existing native vegetation
- ASSESS potential unintended consequences
- RECOMMEND adaptive management
- CONSIDER landscape-level effects, not just site-level

# Response Guidelines:
- Frame recommendations as suggestions for expert review
- Include confidence levels and uncertainty ranges
- Note any potential risks or unintended consequences
- Recommend community consultation as mandatory
- Cite relevant ethical principles and laws
"""

UNCERTAINTY_DISCLOSURE = """
## Uncertainty Disclosure

This AI system has limitations:
1. **Data limitations**: Satellite imagery may not capture ground-level conditions
2. **Model limitations**: Predictions are based on historical patterns that may not hold in changing climate
3. **Local knowledge gap**: Site-specific conditions require on-ground verification
4. **Temporal lag**: Conditions may have changed since last satellite observation

**Recommendation**: All AI suggestions should be validated by local experts and stakeholders before implementation.
"""


def get_system_prompt(task_type: str = "general") -> str:
    """
    Get the appropriate system prompt for a given task.
    
    Args:
        task_type: "general", "species", "legal", or "ethical"
        
    Returns:
        System prompt string
    """
    prompts = {
        "general": RESTORATION_SYSTEM_PROMPT,
        "species": SPECIES_RECOMMENDATION_PROMPT,
        "legal": LEGAL_COMPLIANCE_PROMPT,
        "ethical": ETHICAL_SYSTEM_PROMPT,
    }
    return prompts.get(task_type, RESTORATION_SYSTEM_PROMPT)


def create_constrained_prompt(
    user_query: str,
    site: Optional[SiteConditions] = None,
    task_type: str = "general"
) -> str:
    """
    Create a prompt with site-specific constraints included.
    
    Args:
        user_query: The user's restoration question
        site: Optional site conditions for context
        task_type: Type of task for system prompt selection
        
    Returns:
        Complete prompt with constraints
    """
    system_prompt = get_system_prompt(task_type)
    
    # Add site-specific context if provided
    site_context = ""
    if site:
        site_context = f"""
# SITE-SPECIFIC CONSTRAINTS:
- Location: {site.latitude:.4f}°N, {site.longitude:.4f}°W
- Elevation: {site.elevation_m:.0f}m
- Land ownership: {site.land_owner}
- Protection status: {site.protected_status.value}
- Slope: {site.slope_degrees:.1f}°
- Soil type: {site.soil_type.value}
- Burn severity: {site.burn_severity:.0%}
- Near water: {site.near_water_body}
- Tribal land: {site.land_owner == 'tribal'}
"""
    
    return f"""{system_prompt}

{site_context}

# USER QUERY:
{user_query}

# YOUR RESPONSE:
(Remember to follow all constraints above)
"""


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def validate_restoration_plan(
    site: SiteConditions,
    proposed_actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate a complete restoration plan against all constraints.
    
    Args:
        site: Site conditions
        proposed_actions: List of proposed actions as dicts
        
    Returns:
        Validation result with allowed/blocked actions and warnings
    """
    checker = ConstraintChecker()
    
    # Convert dicts to RestorationAction objects
    actions = [
        RestorationAction(
            name=a.get("name", "Unknown"),
            description=a.get("description", ""),
            species=a.get("species"),
            method=a.get("method"),
        )
        for a in proposed_actions
    ]
    
    # Check constraints
    filtered, warnings = checker.check_site_constraints(site, actions)
    
    return {
        "allowed_actions": [a for a in filtered if a.allowed],
        "blocked_actions": checker.get_blocked_actions(filtered),
        "global_warnings": warnings,
        "all_permits_required": list(set(
            permit for a in filtered for permit in a.required_permits
        )),
        "is_valid": all(a.allowed for a in filtered),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("EcoRevive Constraints Test")
    print("=" * 60)
    
    # Test site in a National Park
    site = SiteConditions(
        latitude=40.05,
        longitude=-121.20,
        elevation_m=1500,
        protected_status=ProtectedStatus.WILDERNESS,
        land_owner="federal",
        slope_degrees=35,
        soil_type=SoilType.ROCKY,
        burn_severity=0.75,
    )
    
    # Test some actions
    actions = [
        RestorationAction(
            name="Plant Ponderosa Pine",
            description="Plant native conifers",
            species="Pinus ponderosa",
            method="planting"
        ),
        RestorationAction(
            name="Plant Eucalyptus",
            description="Fast-growing shade tree",
            species="Eucalyptus globulus",
            method="planting"
        ),
        RestorationAction(
            name="Use heavy machinery",
            description="Clear debris with bulldozer",
            method="machinery"
        ),
    ]
    
    checker = ConstraintChecker()
    filtered, warnings = checker.check_site_constraints(site, actions)
    
    print(f"\nSite: Wilderness Area, {site.slope_degrees} deg slope, {site.burn_severity:.0%} burn severity")
    print(f"\nGlobal Warnings:")
    for w in warnings:
        print(f"   {w}")
    
    print(f"\n[OK] Allowed Actions:")
    for a in filtered:
        if a.allowed:
            print(f"   - {a.name}")
            for w in a.warnings:
                print(f"     [WARNING] {w}")
    
    print(f"\n[BLOCKED] Blocked Actions:")
    for a in filtered:
        if not a.allowed:
            print(f"   - {a.name}")
            print(f"     Reason: {a.blocked_reason}")
