"""
EcoRevive Reasoning Engine (Layers 2 & 3)
==========================================
Gemini-powered restoration intelligence with TRUE MULTIMODAL analysis.

This package provides AI-driven ecosystem analysis for post-fire restoration,
showcasing multiple Gemini API features:

- TRUE MULTIMODAL analysis (RGB + severity overlay + structured context)
- Spatial pattern reasoning (fragmentation, edges, gradients)
- Structured JSON output with machine-readable signals
- Land use classification and contextual warnings (Layer 3)
- Function calling for agentic behavior
- Grounding with Google Search
- Imagen 3 for visualization
- RAG with text-embedding-004

Modules:
    gemini_client: Unified Gemini API client
    gemini_multimodal: TRUE multimodal U-Net → Gemini integration
    gemini_ecosystem: Ecosystem classification and species recommendations
    gemini_orchestrator: Function calling coordinator
    gemini_hope: Recovery forecasting and hope visualization
    constraints: Safety guardrails and legal compliance
    layer2_output: Structured Layer 2 output schema
    layer3_context: Urban detection and contextual warnings (NEW)

Example:
    >>> from reasoning import classify_ecosystem_multimodal, orchestrate_analysis
    >>> result = classify_ecosystem_multimodal((40.05, -121.20), rgb_tile, severity_map)
    >>> full_report = orchestrate_analysis(location, stats, "professional")
    >>>
    >>> # Layer 3: Check land use and get warnings
    >>> from reasoning import create_layer3_response
    >>> context = create_layer3_response(client, rgb, severity, location)
    >>> if context['land_use']['is_urban']:
    ...     print("Warning:", context['land_use']['caution_message'])
"""

from .gemini_client import EcoReviveGemini, create_client
from .gemini_multimodal import (
    MultimodalAnalyzer,
    create_image_pack,
    build_gemini_context,
    compute_severity_statistics,
    compute_spatial_metrics,
    validate_gemini_output,
    should_trigger_human_review,
    analyze_with_multimodal,
)
from .gemini_ecosystem import (
    EcosystemClassifier,
    classify_ecosystem,
    classify_ecosystem_multimodal,
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
from .layer2_output import (
    Layer2Output,
    LocationContext,
    SiteCharacteristics,
    EcosystemInfo,
    SiteMetrics,
    ZonePrimitive,
    HazardAnnotation,
    RiskGrid,
    run_layer2_analysis,
    create_layer2_response,
)
from .layer3_context import (
    Layer3Output,
    LandUseContext,
    LandUseType,
    CautionLevel,
    run_layer3_analysis,
    create_layer3_response,
)
from .carbon_calculator import (
    CarbonCalculatorOutput,
    CarbonAnalysisPersonal,
    CarbonAnalysisProfessional,
    calculate_carbon,
    create_carbon_response,
)
from .pdf_export import (
    generate_pdf,
    build_personal_report,
    build_professional_report,
)

__version__ = "0.1.0"
__author__ = "EcoRevive Team"

__all__ = [
    # Client
    "EcoReviveGemini",
    "create_client",
    
    # Multimodal (NEW - TRUE multimodal integration)
    "MultimodalAnalyzer",
    "create_image_pack",
    "build_gemini_context",
    "compute_severity_statistics",
    "compute_spatial_metrics",
    "validate_gemini_output",
    "should_trigger_human_review",
    "analyze_with_multimodal",
    
    # Ecosystem
    "EcosystemClassifier", 
    "classify_ecosystem",
    "classify_ecosystem_multimodal",  # RECOMMENDED
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

    # Layer 2 Output
    "Layer2Output",
    "LocationContext",
    "SiteCharacteristics",
    "EcosystemInfo",
    "SiteMetrics",
    "ZonePrimitive",
    "HazardAnnotation",
    "RiskGrid",
    "run_layer2_analysis",
    "create_layer2_response",

    # Layer 3 Context (Urban Detection & Cautions)
    "Layer3Output",
    "LandUseContext",
    "LandUseType",
    "CautionLevel",
    "run_layer3_analysis",
    "create_layer3_response",

    # Carbon Calculator
    "CarbonCalculatorOutput",
    "CarbonAnalysisPersonal",
    "CarbonAnalysisProfessional",
    "calculate_carbon",
    "create_carbon_response",

    # PDF Export
    "generate_pdf",
    "build_personal_report",
    "build_professional_report",
]

