"""
EcoRevive Reasoning Engine (Layer 2)
====================================
Gemini-powered restoration intelligence.

This package provides AI-driven ecosystem analysis for post-fire restoration,
showcasing multiple Gemini API features:

- Multimodal analysis (images + text)
- Structured JSON output
- Function calling for agentic behavior
- Grounding with Google Search
- Imagen 3 for visualization
- RAG with text-embedding-004

Modules:
    gemini_client: Unified Gemini API client
    gemini_ecosystem: Ecosystem classification and species recommendations
    gemini_orchestrator: Function calling coordinator
    gemini_hope: Recovery forecasting and hope visualization
    constraints: Safety guardrails and legal compliance

Example:
    >>> from reasoning import classify_ecosystem, orchestrate_analysis
    >>> result = classify_ecosystem((40.05, -121.20), severity_map)
    >>> full_report = orchestrate_analysis(location, stats, "professional")
"""

from .gemini_client import EcoReviveGemini, create_client
from .gemini_ecosystem import (
    EcosystemClassifier,
    classify_ecosystem,
    severity_map_to_image,
    compute_severity_stats,
    CALIFORNIA_ECOREGIONS
)
from .gemini_orchestrator import (
    RestorationOrchestrator,
    orchestrate_analysis,
    RESTORATION_TOOLS
)
from .gemini_hope import (
    HopeVisualizer,
    forecast_recovery,
    generate_hope_visualization,
    RECOVERY_PROFILES
)
from .constraints import (
    ConstraintChecker,
    SiteConditions,
    ProtectedStatus,
    SoilType,
    RestorationAction,
    validate_restoration_plan,
    get_system_prompt,
    create_constrained_prompt,
    RESTORATION_SYSTEM_PROMPT,
    SPECIES_RECOMMENDATION_PROMPT,
    LEGAL_COMPLIANCE_PROMPT,
    ETHICAL_SYSTEM_PROMPT,
    UNCERTAINTY_DISCLOSURE,
)

__version__ = "0.1.0"
__author__ = "EcoRevive Team"

__all__ = [
    # Client
    "EcoReviveGemini",
    "create_client",
    
    # Ecosystem
    "EcosystemClassifier", 
    "classify_ecosystem",
    "severity_map_to_image",
    "compute_severity_stats",
    "CALIFORNIA_ECOREGIONS",
    
    # Orchestrator
    "RestorationOrchestrator",
    "orchestrate_analysis",
    "RESTORATION_TOOLS",
    
    # Hope
    "HopeVisualizer",
    "forecast_recovery",
    "generate_hope_visualization",
    "RECOVERY_PROFILES",
    
    # Constraints (Safety Guardrails)
    "ConstraintChecker",
    "SiteConditions",
    "ProtectedStatus",
    "SoilType",
    "RestorationAction",
    "validate_restoration_plan",
    "get_system_prompt",
    "create_constrained_prompt",
    "RESTORATION_SYSTEM_PROMPT",
    "SPECIES_RECOMMENDATION_PROMPT",
    "LEGAL_COMPLIANCE_PROMPT",
    "ETHICAL_SYSTEM_PROMPT",
    "UNCERTAINTY_DISCLOSURE",
]

