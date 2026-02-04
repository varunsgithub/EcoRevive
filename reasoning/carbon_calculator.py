"""
EcoRevive Carbon Calculator Module
==================================
Calculates carbon sequestration potential for restoration sites.

Provides two output modes:
- Personal: Fun equivalencies and emotional impact metrics
- Professional: Scientific estimates with uncertainty and protocol references

Carbon Calculation Methodology:
- Based on IPCC Tier 2 guidelines for forest carbon accounting
- Uses California-specific carbon accumulation rates from peer-reviewed literature
- Accounts for burn severity impact on baseline carbon stock

References:
- IPCC 2006 Guidelines for National GHG Inventories
- California Air Resources Board Forest Protocol
- Hurteau & North 2009 - Carbon recovery in Sierra Nevada forests
- USFS Forest Inventory and Analysis data

DISCLAIMER: These are estimates for planning purposes only.
Actual carbon sequestration varies based on site conditions,
species selection, survival rates, and climate factors.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CARBON CONSTANTS & RATES
# =============================================================================

# Carbon accumulation rates by ecosystem type (tC/ha/year)
# Source: USFS FIA data, California-specific values
CARBON_RATES = {
    "mixed_conifer": 2.8,      # Sierra Nevada mixed conifer
    "oak_woodland": 1.5,       # Blue oak, valley oak woodlands
    "chaparral": 0.8,          # Shrubland/chaparral
    "grassland": 0.3,          # Annual/perennial grasslands
    "riparian": 3.2,           # Riparian forest (high productivity)
    "redwood": 4.5,            # Coastal redwood (highest)
    "pinyon_juniper": 0.6,     # High desert woodland
    "default": 2.0,            # Conservative default
}

# Baseline carbon stock by ecosystem (tC/ha) - pre-fire
BASELINE_CARBON = {
    "mixed_conifer": 180,
    "oak_woodland": 90,
    "chaparral": 35,
    "grassland": 15,
    "riparian": 200,
    "redwood": 350,
    "pinyon_juniper": 45,
    "default": 100,
}

# Carbon loss by burn severity (fraction of baseline lost)
SEVERITY_CARBON_LOSS = {
    "high": 0.65,      # 65% of carbon lost in high severity
    "moderate": 0.35,  # 35% lost in moderate
    "low": 0.15,       # 15% lost in low severity
    "unburned": 0.0,   # No loss
}

# Time horizons for projections (years)
TIME_HORIZONS = [5, 10, 20, 30, 50]

# Conversion factors
TC_TO_TCO2E = 3.67  # 1 ton Carbon = 3.67 tons CO2 equivalent

# Fun equivalencies for personal users
# Source: EPA Greenhouse Gas Equivalencies Calculator
EQUIVALENCIES = {
    "cars_per_year": 4.6,           # tCO2e per car per year
    "homes_per_year": 7.5,          # tCO2e per home electricity per year
    "gasoline_gallons": 0.00889,    # tCO2e per gallon
    "flights_nyc_la": 0.9,          # tCO2e per round trip
    "smartphones_charged": 0.000008, # tCO2e per smartphone charge
    "trees_seedling_10yr": 0.06,    # tCO2e per seedling over 10 years
    "acres_forest_1yr": 0.84,       # tCO2e per acre per year
}

# Uncertainty ranges (coefficient of variation)
UNCERTAINTY = {
    "carbon_rate": 0.25,      # ±25% on accumulation rates
    "baseline": 0.30,         # ±30% on baseline carbon
    "severity_impact": 0.20,  # ±20% on severity loss estimates
    "combined": 0.35,         # Combined uncertainty ~35%
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CarbonEquivalencies:
    """Fun equivalencies for personal users."""
    cars_off_road_for_year: float
    homes_electricity_year: float
    round_trip_flights_nyc_la: float
    gallons_gasoline: float
    tree_seedlings_grown_10yr: int
    smartphone_charges: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CarbonProjection:
    """Carbon sequestration projection for a time horizon."""
    years: int
    cumulative_tco2e: float
    cumulative_tc: float
    annual_rate_tco2e: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CarbonAnalysisPersonal:
    """Simplified carbon analysis for personal users."""
    # Simple headline numbers
    total_co2_capture_20yr: float  # tons CO2 over 20 years
    annual_co2_capture: float      # tons CO2 per year at maturity
    trees_equivalent: int          # Number of mature trees equivalent

    # Fun equivalencies
    equivalencies: CarbonEquivalencies

    # Emotional impact statements
    impact_statements: List[str]

    # Simple confidence indicator
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['equivalencies'] = self.equivalencies.to_dict()
        return result


@dataclass
class CarbonAnalysisProfessional:
    """Detailed carbon analysis for professional users."""
    # Core metrics
    area_hectares: float
    baseline_carbon_tc: float           # Pre-fire carbon stock
    carbon_lost_tc: float               # Carbon lost to fire
    current_carbon_tc: float            # Current carbon stock

    # Sequestration potential
    annual_sequestration_tc: float      # tC/ha/year rate
    annual_sequestration_tco2e: float   # tCO2e/year total

    # Projections
    projections: List[CarbonProjection]

    # Protocol compatibility
    protocols: Dict[str, bool]  # Which protocols this qualifies for

    # Uncertainty quantification
    uncertainty_pct: float
    confidence_interval_low: float   # 95% CI lower bound (20yr)
    confidence_interval_high: float  # 95% CI upper bound (20yr)

    # Methodology notes
    methodology: str
    limitations: List[str]
    data_sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['projections'] = [p.to_dict() for p in self.projections]
        return result


@dataclass
class CarbonCalculatorOutput:
    """Complete carbon calculator output."""
    personal: CarbonAnalysisPersonal
    professional: CarbonAnalysisProfessional
    ecosystem_type: str
    calculation_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personal": self.personal.to_dict(),
            "professional": self.professional.to_dict(),
            "ecosystem_type": self.ecosystem_type,
            "calculation_timestamp": self.calculation_timestamp,
        }


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def estimate_ecosystem_type(
    location: Tuple[float, float],
    elevation_m: Optional[float] = None,
    land_use_type: Optional[str] = None
) -> str:
    """
    Estimate ecosystem type from location and context.

    This is a simplified estimation. In production, would use
    actual vegetation maps (CALVEG, LANDFIRE, etc.)
    """
    lat, lon = location

    # California-specific heuristics
    if land_use_type:
        if land_use_type in ["forest"]:
            return "mixed_conifer"
        elif land_use_type in ["shrubland"]:
            return "chaparral"
        elif land_use_type in ["grassland"]:
            return "grassland"
        elif land_use_type in ["wetland"]:
            return "riparian"

    # Location-based estimation for California
    if -124 < lon < -122 and 38 < lat < 42:
        # North coast - likely redwood/mixed conifer
        return "redwood" if lon < -123 else "mixed_conifer"
    elif -122 < lon < -118 and 36 < lat < 41:
        # Sierra Nevada
        return "mixed_conifer"
    elif -122 < lon < -118 and lat < 36:
        # Southern California
        return "chaparral"
    elif -122 < lon < -119 and lat < 38:
        # Central Valley
        return "oak_woodland"

    return "default"


def calculate_carbon_metrics(
    area_hectares: float,
    severity_stats: Dict[str, float],
    ecosystem_type: str = "default",
    time_horizon_years: int = 20
) -> Tuple[float, float, float, float]:
    """
    Calculate core carbon metrics.

    Returns:
        (baseline_tc, carbon_lost_tc, annual_rate_tc, total_potential_tco2e)
    """
    # Get ecosystem-specific rates
    carbon_rate = CARBON_RATES.get(ecosystem_type, CARBON_RATES["default"])
    baseline_per_ha = BASELINE_CARBON.get(ecosystem_type, BASELINE_CARBON["default"])

    # Calculate baseline carbon stock
    baseline_tc = baseline_per_ha * area_hectares

    # Calculate carbon lost based on severity distribution
    high_pct = severity_stats.get("high_severity_ratio", 0) or severity_stats.get("high_severity_pct", 0) / 100
    mod_pct = severity_stats.get("moderate_severity_ratio", 0) or severity_stats.get("moderate_severity_pct", 0) / 100
    low_pct = severity_stats.get("low_severity_ratio", 0) or severity_stats.get("low_severity_pct", 0) / 100

    # Weighted carbon loss
    weighted_loss = (
        high_pct * SEVERITY_CARBON_LOSS["high"] +
        mod_pct * SEVERITY_CARBON_LOSS["moderate"] +
        low_pct * SEVERITY_CARBON_LOSS["low"]
    )

    carbon_lost_tc = baseline_tc * weighted_loss

    # Annual sequestration rate (full area)
    annual_rate_tc = carbon_rate * area_hectares

    # Total potential over time horizon (simplified linear model)
    # In reality, this follows a sigmoid growth curve
    total_potential_tc = annual_rate_tc * time_horizon_years
    total_potential_tco2e = total_potential_tc * TC_TO_TCO2E

    return baseline_tc, carbon_lost_tc, annual_rate_tc, total_potential_tco2e


def generate_projections(
    annual_rate_tc: float,
    time_horizons: List[int] = None
) -> List[CarbonProjection]:
    """Generate carbon projections for multiple time horizons."""
    if time_horizons is None:
        time_horizons = TIME_HORIZONS

    projections = []
    for years in time_horizons:
        cumulative_tc = annual_rate_tc * years
        cumulative_tco2e = cumulative_tc * TC_TO_TCO2E
        annual_tco2e = annual_rate_tc * TC_TO_TCO2E

        projections.append(CarbonProjection(
            years=years,
            cumulative_tco2e=round(cumulative_tco2e, 1),
            cumulative_tc=round(cumulative_tc, 1),
            annual_rate_tco2e=round(annual_tco2e, 2)
        ))

    return projections


def calculate_equivalencies(total_tco2e_20yr: float) -> CarbonEquivalencies:
    """Calculate fun equivalencies for personal users."""
    return CarbonEquivalencies(
        cars_off_road_for_year=round(total_tco2e_20yr / EQUIVALENCIES["cars_per_year"], 1),
        homes_electricity_year=round(total_tco2e_20yr / EQUIVALENCIES["homes_per_year"], 1),
        round_trip_flights_nyc_la=round(total_tco2e_20yr / EQUIVALENCIES["flights_nyc_la"], 0),
        gallons_gasoline=round(total_tco2e_20yr / EQUIVALENCIES["gasoline_gallons"], 0),
        tree_seedlings_grown_10yr=int(total_tco2e_20yr / EQUIVALENCIES["trees_seedling_10yr"]),
        smartphone_charges=int(total_tco2e_20yr / EQUIVALENCIES["smartphones_charged"])
    )


def generate_impact_statements(
    total_tco2e: float,
    area_hectares: float,
    equivalencies: CarbonEquivalencies
) -> List[str]:
    """Generate emotional impact statements for personal users."""
    statements = []

    # Lead with the most impactful number
    if equivalencies.cars_off_road_for_year >= 10:
        statements.append(
            f"Restoring this area could capture the same CO2 as taking "
            f"{int(equivalencies.cars_off_road_for_year)} cars off the road for a year!"
        )

    # Tree equivalent
    statements.append(
        f"That's like growing {equivalencies.tree_seedlings_grown_10yr:,} tree seedlings for 10 years."
    )

    # Flight equivalent if significant
    if equivalencies.round_trip_flights_nyc_la >= 5:
        statements.append(
            f"It offsets {int(equivalencies.round_trip_flights_nyc_la)} round-trip flights from NYC to LA."
        )

    # Fun smartphone stat
    if equivalencies.smartphone_charges > 1000000:
        millions = equivalencies.smartphone_charges / 1000000
        statements.append(
            f"Enough clean energy to charge {millions:.1f} million smartphones!"
        )

    # Hopeful closing statement
    statements.append(
        "Every restored hectare is a step toward healing our planet."
    )

    return statements


def assess_protocol_compatibility(
    area_hectares: float,
    total_tco2e_20yr: float
) -> Dict[str, bool]:
    """Assess compatibility with carbon credit protocols."""
    # Ensure we're working with native Python floats to avoid numpy bool issues
    area = float(area_hectares)
    total = float(total_tco2e_20yr)
    return {
        "verra_vcs_eligible": bool(area >= 10 and total >= 100),
        "gold_standard_eligible": bool(area >= 5 and total >= 50),
        "plan_vivo_eligible": True,  # Community-scale projects
        "california_arb_eligible": bool(area >= 40),  # 100 acres minimum
        "american_carbon_registry": bool(area >= 20),
    }


# =============================================================================
# MAIN CALCULATION FUNCTION
# =============================================================================

def calculate_carbon(
    area_hectares: float,
    severity_stats: Dict[str, float],
    location: Tuple[float, float],
    land_use_type: Optional[str] = None,
    ecosystem_type: Optional[str] = None
) -> CarbonCalculatorOutput:
    """
    Main carbon calculation function.

    Args:
        area_hectares: Site area in hectares
        severity_stats: Burn severity statistics from Layer 1
        location: (latitude, longitude) tuple
        land_use_type: Optional land use from Layer 3
        ecosystem_type: Optional override for ecosystem type

    Returns:
        CarbonCalculatorOutput with personal and professional analyses
    """
    from datetime import datetime, timezone

    # Determine ecosystem type
    if ecosystem_type is None:
        ecosystem_type = estimate_ecosystem_type(location, land_use_type=land_use_type)

    # Calculate core metrics
    baseline_tc, carbon_lost_tc, annual_rate_tc, total_20yr_tco2e = calculate_carbon_metrics(
        area_hectares=area_hectares,
        severity_stats=severity_stats,
        ecosystem_type=ecosystem_type,
        time_horizon_years=20
    )

    current_tc = baseline_tc - carbon_lost_tc
    annual_tco2e = annual_rate_tc * TC_TO_TCO2E

    # Generate projections
    projections = generate_projections(annual_rate_tc)

    # Calculate equivalencies
    equivalencies = calculate_equivalencies(total_20yr_tco2e)

    # Generate impact statements
    impact_statements = generate_impact_statements(
        total_20yr_tco2e, area_hectares, equivalencies
    )

    # Assess protocol compatibility
    protocols = assess_protocol_compatibility(area_hectares, total_20yr_tco2e)

    # Calculate uncertainty
    uncertainty_pct = UNCERTAINTY["combined"] * 100
    ci_low = total_20yr_tco2e * (1 - UNCERTAINTY["combined"] * 1.96)
    ci_high = total_20yr_tco2e * (1 + UNCERTAINTY["combined"] * 1.96)

    # Determine confidence level
    if area_hectares > 50 and ecosystem_type != "default":
        confidence = "high"
    elif area_hectares > 10:
        confidence = "medium"
    else:
        confidence = "low"

    # Build personal analysis
    personal = CarbonAnalysisPersonal(
        total_co2_capture_20yr=round(total_20yr_tco2e, 0),
        annual_co2_capture=round(annual_tco2e, 1),
        trees_equivalent=equivalencies.tree_seedlings_grown_10yr,
        equivalencies=equivalencies,
        impact_statements=impact_statements,
        confidence=confidence
    )

    # Build professional analysis
    professional = CarbonAnalysisProfessional(
        area_hectares=round(area_hectares, 2),
        baseline_carbon_tc=round(baseline_tc, 1),
        carbon_lost_tc=round(carbon_lost_tc, 1),
        current_carbon_tc=round(current_tc, 1),
        annual_sequestration_tc=round(annual_rate_tc, 2),
        annual_sequestration_tco2e=round(annual_tco2e, 2),
        projections=projections,
        protocols=protocols,
        uncertainty_pct=round(uncertainty_pct, 0),
        confidence_interval_low=round(ci_low, 0),
        confidence_interval_high=round(ci_high, 0),
        methodology="IPCC Tier 2 with California-specific coefficients",
        limitations=[
            "Estimates assume successful restoration with >70% survival rate",
            "Does not account for climate change impacts on growth rates",
            "Actual sequestration depends on species selection and site conditions",
            "Fire risk and potential re-release not factored into projections",
        ],
        data_sources=[
            "USFS Forest Inventory and Analysis (FIA)",
            "California Air Resources Board Forest Protocol",
            "IPCC 2006 Guidelines for National GHG Inventories",
            "Hurteau & North 2009 - Sierra Nevada carbon dynamics",
        ]
    )

    return CarbonCalculatorOutput(
        personal=personal,
        professional=professional,
        ecosystem_type=ecosystem_type,
        calculation_timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    )


def create_carbon_response(
    area_hectares: float,
    severity_stats: Dict[str, float],
    location: Tuple[float, float],
    user_type: str = "personal",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function that returns carbon analysis as a dict.

    Args:
        area_hectares: Site area in hectares
        severity_stats: Burn severity statistics
        location: (lat, lon) tuple
        user_type: "personal" or "professional"

    Returns:
        Dict with carbon analysis (filtered by user type)
    """
    output = calculate_carbon(
        area_hectares=area_hectares,
        severity_stats=severity_stats,
        location=location,
        **kwargs
    )

    result = output.to_dict()

    # Add user-type specific top-level summary
    if user_type == "personal":
        result["summary"] = {
            "headline": f"{int(output.personal.total_co2_capture_20yr):,} tons CO2",
            "subheadline": f"captured over 20 years of restoration",
            "key_equivalency": output.personal.impact_statements[0] if output.personal.impact_statements else "",
        }
    else:
        result["summary"] = {
            "headline": f"{output.professional.annual_sequestration_tco2e:.1f} tCO2e/year",
            "subheadline": f"sequestration potential ({output.ecosystem_type} ecosystem)",
            "protocols_eligible": sum(1 for v in output.professional.protocols.values() if v),
        }

    return result
