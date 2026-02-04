"""
EcoRevive Prompt Augmentation Rules v2.0
=========================================
Principal Architect Reviewed - Production Grade

REVIEW FINDINGS (ADDRESSED):
1. Citation requirements were too loose - now mandatory format
2. Uncertainty language gates were inconsistent - now threshold-based
3. Action-type rules lacked prohibition enforcement - now explicit
4. Evidence block missing validation checkpoints - now structured

Design Principle: These rules shape prompt structure, not content.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# =============================================================================
# PROMPT STRUCTURE RULES (REFINED)
# =============================================================================

PROMPT_STRUCTURE_RULES = """
PROMPT AUGMENTATION STRUCTURE v2.0
===================================

Every LLM prompt MUST follow this exact structure:

1. ROLE DEFINITION (mandatory - sets professional identity)
   ├── Professional scope
   ├── Ethical constraints
   └── Output constraints

2. REASONING CONSTRAINTS (mandatory - from reasoning_framework.py)
   ├── Area-type eligible risks (whitelist)
   ├── Area-type prohibited risks (blacklist)
   ├── Evidence-permitted topics
   └── Severity language rules

3. EVIDENCE BLOCK (mandatory - structured JSON summary)
   ├── Location with proper N/S, E/W formatting
   ├── Quantified metrics with units
   ├── Confidence indicators
   └── Data freshness timestamp

4. RAG CONTEXT (conditional - if retrieval performed)
   ├── Source citations
   ├── Relevance scores
   └── Document types

5. USER QUERY (mandatory)
   └── Original request with action type

6. OUTPUT CONSTRAINTS (mandatory)
   ├── Format requirements
   ├── Citation format
   └── Prohibited phrases
"""


# =============================================================================
# EVIDENCE INJECTION RULES (REFINED)
# =============================================================================

@dataclass
class EvidenceValidation:
    """Validation results for evidence block."""
    is_valid: bool
    missing_required: List[str]
    warnings: List[str]


def validate_evidence(layer2_json: Dict[str, Any]) -> EvidenceValidation:
    """
    Validate that evidence block meets minimum requirements.
    
    Required fields for ANY output:
    - location (lat/lon)
    - area
    - confidence indicator
    """
    required = ["location", "area_km2"]
    recommended = ["burn_severity", "overall_confidence"]
    
    missing = [f for f in required if f not in layer2_json]
    warnings = [f"Recommended field missing: {f}" for f in recommended if f not in layer2_json]
    
    return EvidenceValidation(
        is_valid=len(missing) == 0,
        missing_required=missing,
        warnings=warnings,
    )


def format_evidence_block(layer2_json: Dict[str, Any]) -> str:
    """
    Format Layer 2 JSON as evidence block for prompt injection.
    
    REFINED: Mandatory format, validation checkpoint, explicit structure.
    """
    
    lines = [
        "# AUTHORITATIVE SITE DATA (JSON-derived)",
        "════════════════════════════════════════",
        "",
        "⚠️ These values are your ONLY evidence source.",
        "⚠️ You may NOT make claims that contradict this data.",
        "⚠️ You may NOT make claims without citing a field below.",
        "",
    ]
    
    # Location (with proper formatting)
    location = layer2_json.get("location", {})
    lat = location.get("lat", layer2_json.get("center_lat", 0))
    lon = location.get("lon", layer2_json.get("center_lon", 0))
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    lines.append(f"**Location**: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")
    
    # Area
    area = layer2_json.get("area_km2", layer2_json.get("area", 0))
    if area:
        lines.append(f"**Area**: {area:.2f} km² ({area * 100:.1f} hectares)")
    
    # Severity metrics
    severity = layer2_json.get("burn_severity", layer2_json.get("severity_stats", {}))
    if severity:
        lines.append("")
        lines.append("**Burn Severity Metrics**:")
        
        mean = severity.get("mean", severity.get("mean_severity", 0))
        if isinstance(mean, (int, float)):
            lines.append(f"- mean_severity: {mean:.1%}")
        
        high = severity.get("high_ratio", severity.get("high_severity_ratio", 0))
        if isinstance(high, (int, float)):
            lines.append(f"- high_severity_ratio: {high:.1%}")
        
        mod = severity.get("moderate_ratio", severity.get("moderate_severity_ratio", 0))
        if isinstance(mod, (int, float)):
            lines.append(f"- moderate_severity_ratio: {mod:.1%}")
    
    # Land use (from Layer 3)
    land_use = layer2_json.get("land_use", {})
    if land_use:
        lines.append("")
        lines.append("**Land Use Classification**:")
        lines.append(f"- type: {land_use.get('type', 'unknown')}")
        urban = land_use.get("urban_percent", 0)
        if isinstance(urban, (int, float)):
            lines.append(f"- urban_percent: {urban:.0f}%")
    
    # Confidence
    confidence = layer2_json.get("overall_confidence", 
                                 layer2_json.get("confidence", 0.5))
    lines.append("")
    lines.append(f"**Data Confidence**: {confidence:.0%}")
    
    # Data age if available
    data_age = layer2_json.get("data_age_days", 0)
    if data_age > 0:
        lines.append(f"**Data Age**: {data_age} days")
    
    lines.append("")
    lines.append("════════════════════════════════════════")
    
    return "\n".join(lines)


# =============================================================================
# CITATION REQUIREMENT RULES (REFINED)
# =============================================================================

CITATION_RULES = """
CITATION REQUIREMENTS v2.0 (MANDATORY)
=======================================

Every factual claim MUST have a citation. No exceptions.

## CITATION FORMAT

1. JSON-derived claims:
   Format: "statement (field_name: value)"
   Example: "The site shows moderate burn severity (mean_severity: 47%)"

2. RAG-retrieved claims:
   Format: "statement [source type]"
   Example: "Native toyon is fire-adapted [ecology_rag: species_catalog]"

3. Inference claims (allowed only for moderate+ evidence):
   Format: "Given [evidence], [inference]"
   Example: "Given 65% high-severity burn, natural regeneration will be slow"

## PROHIBITED (ABSOLUTE BLOCK)

❌ Unsourced factual claims
❌ "Studies show..." without [source] tag
❌ "Research indicates..." without citation
❌ "Generally..." or "Typically..." claims
❌ Claims contradicting JSON values
❌ Inferences from < 0.3 evidence values

## VALIDATION CHECKPOINT

Before every factual statement, verify:
1. Is there a JSON field or RAG source supporting this?
2. Is the citation format correct?
3. Does the claim align with (not contradict) the evidence?

If ANY check fails → DO NOT MAKE THE STATEMENT
"""


# =============================================================================
# UNCERTAINTY LANGUAGE RULES (REFINED)
# =============================================================================

UNCERTAINTY_LANGUAGE_RULES = """
UNCERTAINTY DISCLOSURE RULES v2.0
==================================

Language certainty MUST match data confidence.

## CONFIDENCE GATES (STRICT)

Confidence ≥ 0.8:
├── Permitted: "shows", "indicates", "demonstrates", "has"
├── Example: "The analysis shows high burn severity"
└── No qualifiers required

Confidence 0.6 - 0.8:
├── Permitted: "suggests", "appears to", "is likely"
├── Required: Cite confidence level in section
└── Example: "The data suggests moderate vegetation loss"

Confidence 0.4 - 0.6:
├── Required: "may", "could", "potentially"
├── Required: "(moderate confidence)" tag
└── Example: "There may be some erosion risk (moderate confidence)"

Confidence 0.2 - 0.4:
├── Required: "preliminary data suggests", "uncertain evidence indicates"
├── Required: "(low confidence)" tag
└── Example: "Preliminary data suggests possible impact (low confidence)"

Confidence < 0.2:
├── SILENCE preferred
├── If must mention: "Insufficient data; field verification required"
└── NO substantive claims permitted

## HARD PROHIBITED PHRASES (ANY CONFIDENCE)

❌ "Definitely"
❌ "Certainly" 
❌ "Without doubt"
❌ "Absolutely"
❌ "Always"
❌ "Never"
❌ "Guaranteed"
❌ "Obvious" / "Clearly" (unless confidence > 0.9)
"""


# =============================================================================
# ACTION-TYPE SPECIFIC RULES (REFINED)
# =============================================================================

ACTION_TYPE_RULES = {
    "species": {
        "focus": [
            "native species suitability",
            "fire adaptation characteristics", 
            "ecological niche requirements",
            "planting zone compatibility",
        ],
        "prohibit": [
            "aesthetic preferences",
            "non-native alternatives",
            "landscaping design",
            "species not in RAG catalog",
        ],
        "citation_priority": "ecology_rag",
        "confidence_floor": 0.4,  # Minimum confidence to give species recs
        "prohibited_phrases": [
            "consider planting" (without RAG source),
            "popular choices",
            "commonly used",
        ],
    },
    
    "ownership": {
        "focus": [
            "land jurisdiction classification",
            "permit requirements by land type",
            "relevant agency contacts",
            "access restrictions",
        ],
        "prohibit": [
            "legal advice",
            "ownership dispute guidance",
            "property valuation",
            "liability assessment",
        ],
        "citation_priority": "legal_rag",
        "confidence_floor": 0.6,  # Higher bar for legal info
        "prohibited_phrases": [
            "you should contact a lawyer",
            "legal requirements include",
            "you are required to",
        ],
    },
    
    "safety": {
        "focus": [
            "site-specific hazards from JSON evidence",
            "access considerations from terrain data",
            "temporal factors from data age",
        ],
        "prohibit": [
            "generic safety checklists",
            "encyclopedic hazard lists",
            "worst-case scenarios without evidence",
            "hazards outside area-type eligibility",
        ],
        "citation_priority": "json_evidence",
        "confidence_floor": 0.3,
        "prohibited_phrases": [
            "standard safety precautions include",
            "common hazards are",
            "always watch out for",
            "widowmakers" (unless forest area with canopy_damage > 0.4),
        ],
    },
    
    "monitoring": {
        "focus": [
            "observable recovery indicators",
            "measurement methods appropriate to scale",
            "temporal comparison points",
        ],
        "prohibit": [
            "research-grade protocols",
            "academic methodologies",
            "expensive instrumentation",
        ],
        "citation_priority": "ecological_context",
        "confidence_floor": 0.3,
        "prohibited_phrases": [
            "scientific monitoring requires",
            "research protocols suggest",
        ],
    },
    
    "biophysical": {
        "focus": [
            "soil conditions from evidence",
            "hydrology from water_body data",
            "terrain from slope data",
        ],
        "prohibit": [
            "subsurface speculation",
            "geological assumptions",
            "unobserved soil chemistry",
        ],
        "citation_priority": "json_evidence",
        "confidence_floor": 0.4,
        "prohibited_phrases": [
            "the underlying geology",
            "soil composition likely includes",
        ],
    },
    
    "hope": {
        "focus": [
            "evidence-based recovery potential",
            "documented success factors",
            "positive indicators from JSON",
        ],
        "prohibit": [
            "false promises",
            "guaranteed outcomes",
            "timeline certainties",
            "emotional manipulation",
        ],
        "citation_priority": "ecological_context",
        "confidence_floor": 0.3,
        "prohibited_phrases": [
            "will definitely recover",
            "guaranteed to succeed",
            "no doubt that",
        ],
    },
    
    "carbon": {
        "focus": [
            "quantified site-level carbon",
            "sequestration estimates with units",
            "emission factors from vegetation data",
        ],
        "prohibit": [
            "climate change narratives",
            "global carbon framing",
            "offset recommendations",
            "policy suggestions",
        ],
        "citation_priority": "json_evidence",
        "confidence_floor": 0.5,  # Higher bar for carbon claims
        "prohibited_phrases": [
            "carbon footprint",
            "contributes to climate change",
            "significant emissions",
        ],
    },
}


def get_action_specific_rules(action_type: str) -> str:
    """
    Get action-type specific prompt rules.
    
    REFINED: Explicit prohibitions, confidence floors, phrase blocks.
    """
    rules = ACTION_TYPE_RULES.get(action_type, ACTION_TYPE_RULES["safety"])
    
    focus_str = '\n'.join([f"  • {f}" for f in rules['focus']])
    prohibit_str = '\n'.join([f"  ❌ {p}" for p in rules['prohibit']])
    
    confidence_instruction = ""
    if rules.get('confidence_floor', 0) > 0.3:
        confidence_instruction = f"\n⚠️ Minimum confidence required: {rules['confidence_floor']:.0%}"
    
    return f"""
## ACTION-SPECIFIC CONSTRAINTS: {action_type.upper()}

**Focus your response on**:
{focus_str}

**Do NOT discuss**:
{prohibit_str}

**Primary citation source**: {rules['citation_priority']}
{confidence_instruction}
"""


# =============================================================================
# COMPLETE PROMPT BUILDER (REFINED)
# =============================================================================

def build_rag_augmented_prompt(
    user_query: str,
    action_type: str,
    layer2_json: Dict[str, Any],
    rag_context: str,
    reasoning_constraints: str,
    user_type: str = "personal"
) -> str:
    """
    Build a complete RAG-augmented prompt with all guardrails.
    
    REFINED: Validation checkpoint, stricter structure, explicit constraints.
    """
    
    # Validation checkpoint
    validation = validate_evidence(layer2_json)
    validation_note = ""
    if not validation.is_valid:
        validation_note = f"\n⚠️ Evidence validation warning: Missing {', '.join(validation.missing_required)}\n"
    
    # Role definition (tightened)
    role = """You are a Senior Restoration Ecologist operating as a RAG-Constrained Reasoning Engine.

IDENTITY:
- You provide EVIDENCE-BASED assessments only
- You are bound by the constraints below
- You do NOT have general knowledge outside provided context
- You prefer SILENCE over speculation

CONSTRAINTS ARE MANDATORY, NOT ADVISORY."""

    # Evidence block
    evidence = format_evidence_block(layer2_json)
    
    # Action-specific rules
    action_rules = get_action_specific_rules(action_type)
    
    # User type framing
    if user_type == "professional":
        audience = """## AUDIENCE: PROFESSIONAL
- Use technical terminology
- Include quantified data with units
- Provide error bounds where appropriate
- Cite all sources explicitly"""
    else:
        audience = """## AUDIENCE: GENERAL PUBLIC
- Use accessible language
- Explain technical concepts briefly
- Still maintain evidence grounding
- Do NOT oversimplify to the point of inaccuracy"""
    
    # Assemble prompt
    return f"""{role}
{validation_note}
{reasoning_constraints}

{evidence}

{rag_context}

{action_rules}

{CITATION_RULES}

{UNCERTAINTY_LANGUAGE_RULES}

{audience}

---

# USER QUERY:
{user_query}

---

# YOUR RESPONSE:
(Apply ALL constraints above. Cite sources. Match language to evidence. Silence over speculation.)
"""
