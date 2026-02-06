"""
Professional Report Generator for EcoRevive
=============================================
Comprehensive 5-10 page technical document for grants and stakeholders.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle

from .styles import get_styles, ECOREVIVE_GREEN, BORDER_COLOR
from .utils import (
    create_side_by_side_images, format_coordinates, calculate_area_km2, truncate_text
)

# Design constants
PRIMARY_DARK = HexColor("#0f1c18")
ACCENT_GREEN = HexColor("#00d4aa")
CARD_BG = HexColor("#f4f7f6")
TABLE_HEADER_BG = HexColor("#1a2e28")
TABLE_ALT_ROW = HexColor("#f8fafa")
TEXT_DARK = HexColor("#0f1c18")
TEXT_GRAY = HexColor("#5a6b66")
TEXT_LIGHT = HexColor("#8a9a95")
SEVERITY_HIGH = HexColor("#e63946")
SEVERITY_MOD = HexColor("#f4a261")
SEVERITY_LOW = HexColor("#2a9d8f")

PAGE_WIDTH = 6.0 * inch  # Reduced for better margins


def build_professional_report(
    satellite_image: str,
    severity_image: str,
    severity_stats: dict,
    bbox: dict,
    layer2_output: dict = None,
    layer3_context: dict = None,
    carbon_analysis: dict = None,
    location_name: str = None,
    analysis_id: str = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> list:
    """Build the story elements for a professional grant-ready PDF."""
    styles = get_styles()
    story = []

    area_km2 = calculate_area_km2(bbox)
    area_hectares = area_km2 * 100
    coordinates = format_coordinates(bbox)
    timestamp = datetime.now().strftime("%B %d, %Y")
    time_full = datetime.now().strftime("%B %d, %Y at %H:%M UTC")

    if not location_name:
        location_name = "Site Analysis"

    # ==================== COVER PAGE ====================
    story.extend(_build_cover_page(location_name, coordinates, area_km2, time_full, analysis_id))
    story.append(PageBreak())

    # ==================== TABLE OF CONTENTS ====================
    story.extend(_build_table_of_contents())
    story.append(Spacer(1, 0.3 * inch))

    # ==================== EXECUTIVE SUMMARY ====================
    story.extend(_build_executive_summary(severity_stats, carbon_analysis, area_km2, layer3_context))
    story.append(PageBreak())

    # ==================== SITE INFORMATION ====================
    story.extend(_build_site_section(bbox, coordinates, area_km2, area_hectares, layer2_output, layer3_context))
    story.append(Spacer(1, 0.4 * inch))

    # ==================== BURN SEVERITY ANALYSIS ====================
    story.extend(_build_severity_section(satellite_image, severity_image, severity_stats, area_km2))
    story.append(PageBreak())

    # ==================== BIOPHYSICAL CHARACTERIZATION ====================
    if layer2_output or layer3_context:
        story.extend(_build_biophysical_section(layer2_output, layer3_context))
        story.append(Spacer(1, 0.4 * inch))

    # ==================== ECOLOGICAL ASSESSMENT ====================
    story.extend(_build_ecological_section(severity_stats, layer2_output, layer3_context))
    story.append(PageBreak())

    # ==================== CARBON ACCOUNTING ====================
    if carbon_analysis and carbon_analysis.get('professional'):
        story.extend(_build_carbon_section(carbon_analysis['professional'], area_hectares))
        story.append(PageBreak())

    # ==================== RISK ASSESSMENT ====================
    story.extend(_build_risk_section(layer2_output, severity_stats))
    story.append(Spacer(1, 0.4 * inch))

    # ==================== RECOMMENDATIONS ====================
    story.extend(_build_recommendations_section(severity_stats, layer3_context, carbon_analysis))
    story.append(PageBreak())

    # ==================== METHODOLOGY ====================
    story.extend(_build_methodology_section(layer2_output, layer3_context))
    
    # ==================== CONSULTATION LOG APPENDIX ====================
    if chat_history and len(chat_history) > 0:
        story.append(PageBreak())
        story.extend(_build_consultation_log_appendix(chat_history))

    return story


def _build_cover_page(location_name, coordinates, area_km2, timestamp, analysis_id):
    """Build professional cover page."""
    story = []

    story.append(Spacer(1, 1.5 * inch))

    # EcoRevive branding
    story.append(Paragraph(
        '<font size="16" color="#00d4aa"><b>ECOREVIVE</b></font>',
        _style_left()
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        '<font size="10" color="#8a9a95">Wildfire Recovery Intelligence Platform</font>',
        _style_left()
    ))
    story.append(Spacer(1, 0.4 * inch))

    # Document type
    story.append(Paragraph(
        '<font size="11" color="#8a9a95">PROFESSIONAL BURN SEVERITY ANALYSIS REPORT</font>',
        _style_left()
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Location name - large title
    story.append(Paragraph(
        f'<font size="28" color="#0f1c18"><b>{truncate_text(location_name, 45)}</b></font>',
        _style_left()
    ))
    story.append(Spacer(1, 0.6 * inch))

    # Details table with consistent alignment
    details = [
        ["COORDINATES", coordinates],
        ["ANALYSIS AREA", f"{area_km2:.3f} km² ({area_km2 * 100:.1f} hectares)"],
        ["ANALYSIS DATE", timestamp],
    ]

    for label, value in details:
        row = Table(
            [[
                Paragraph(f'<font size="9" color="#8a9a95">{label}</font>', _style_left()),
                Paragraph(f'<font size="10" color="#0f1c18">{value}</font>', _style_left()),
            ]],
            colWidths=[1.4 * inch, 5.1 * inch]
        )
        row.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(row)

    story.append(Spacer(1, 1.2 * inch))

    # Disclaimer box
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
    story.append(Spacer(1, 0.12 * inch))

    disclaimer = '''
    <font size="8" color="#8a9a95">
    <b>DISCLAIMER:</b> This report is generated using satellite imagery and AI-powered analysis.
    Results are indicative and should be verified with ground-truth data before use in
    official documentation, grant applications, or regulatory submissions. Carbon estimates
    follow IPCC Tier 2 methodology with stated uncertainties. EcoRevive makes no guarantees
    regarding the accuracy of automated analysis. For critical decisions, consult with
    qualified environmental professionals.
    </font>
    '''
    story.append(Paragraph(disclaimer, _style_left_wrapped()))

    return story


def _build_table_of_contents():
    """Build table of contents."""
    story = []

    story.append(_section_header("Table of Contents"))
    story.append(Spacer(1, 0.2 * inch))

    toc_items = [
        ("1. Executive Summary", "Key findings and recommendations"),
        ("2. Site Information", "Location, boundaries, and context"),
        ("3. Burn Severity Analysis", "Satellite imagery and severity classification"),
        ("4. Biophysical Characterization", "Land use, soil, and terrain assessment"),
        ("5. Ecological Assessment", "Vegetation impacts and recovery potential"),
        ("6. Carbon Accounting", "Sequestration potential and protocol eligibility"),
        ("7. Risk Assessment", "Hazards and restoration challenges"),
        ("8. Recommendations", "Prioritized restoration actions"),
        ("9. Data Sources & Methodology", "Technical approach and limitations"),
    ]

    for title, desc in toc_items:
        item = Paragraph(
            f'<font size="10" color="#0f1c18"><b>{title}</b></font><br/>'
            f'<font size="9" color="#8a9a95">{desc}</font>',
            _style_left()
        )
        story.append(item)
        story.append(Spacer(1, 0.08 * inch))

    return story


def _build_executive_summary(severity_stats, carbon_analysis, area_km2, layer3_context):
    """Build comprehensive executive summary."""
    story = []

    story.append(_section_header("1. Executive Summary"))

    # Safely extract severity values with type coercion
    high_sev = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100
    mod_sev = float(severity_stats.get('moderate_severity_ratio', 0) or 0) * 100
    mean_sev = float(severity_stats.get('mean_severity', 0) or 0) * 100
    high_area = high_sev * float(area_km2) / 100

    # Key metrics grid
    metrics = [
        (f"{mean_sev:.0f}%", "Mean Severity"),
        (f"{high_sev:.0f}%", "High Severity"),
        (f"{high_area:.2f}", "km² Critical Area"),
    ]

    if carbon_analysis and carbon_analysis.get('professional'):
        annual = float(carbon_analysis['professional'].get('annual_sequestration_tco2e', 0) or 0)
        metrics.append((f"{annual:.0f}", "tCO₂e/yr Potential"))

    cells = [_metric_box(val, label) for val, label in metrics]
    metrics_table = Table([cells], colWidths=[PAGE_WIDTH / len(cells)] * len(cells))
    metrics_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.25 * inch))

    # Key findings
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Key Findings</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    findings = [
        f"<b>Burn Extent:</b> High severity burn covers {high_sev:.1f}% of the {area_km2:.2f} km² analysis area ({high_area:.3f} km²).",
        f"<b>Severity Distribution:</b> Moderate severity affects an additional {mod_sev:.1f}% of the site, with remaining areas showing low severity or unburned conditions.",
    ]

    if layer3_context:
        land_use = layer3_context.get('land_use', {}).get('land_use_type', 'unknown')
        findings.append(f"<b>Land Classification:</b> Primary land use identified as {land_use.title()}.")

    if carbon_analysis and carbon_analysis.get('professional'):
        annual = float(carbon_analysis['professional'].get('annual_sequestration_tco2e', 0) or 0)
        total_20yr = annual * 20
        findings.append(f"<b>Carbon Potential:</b> Restoration could sequester approximately {annual:.0f} tCO₂e annually, totaling {total_20yr:,.0f} tCO₂e over 20 years.")

    bullet_items = [
        ListItem(Paragraph(f'<font size="10" color="#5a6b66">{f}</font>', _style_left_wrapped()), leftIndent=16)
        for f in findings
    ]
    story.append(ListFlowable(bullet_items, bulletType='bullet', bulletColor=ACCENT_GREEN))
    story.append(Spacer(1, 0.2 * inch))

    # Priority recommendations
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Priority Recommendations</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    recs = _get_priority_recommendations(severity_stats, layer3_context, carbon_analysis)
    rec_items = [
        ListItem(Paragraph(f'<font size="10" color="#5a6b66">{r}</font>', _style_left_wrapped()), leftIndent=16)
        for r in recs[:4]
    ]
    story.append(ListFlowable(rec_items, bulletType='1', bulletColor=ACCENT_GREEN))

    return story


def _build_site_section(bbox, coordinates, area_km2, area_hectares, layer2_output, layer3_context):
    """Build detailed site information section."""
    story = []

    story.append(_section_header("2. Site Information"))

    # Location details table
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Geographic Boundaries</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    data = [
        ["Parameter", "Value", "Notes"],
        ["Center Coordinates", coordinates, "WGS84 Datum"],
        ["North Boundary", f"{bbox.get('north', 0):.6f}°", "Decimal degrees"],
        ["South Boundary", f"{bbox.get('south', 0):.6f}°", ""],
        ["East Boundary", f"{bbox.get('east', 0):.6f}°", ""],
        ["West Boundary", f"{bbox.get('west', 0):.6f}°", ""],
        ["Total Area", f"{area_km2:.4f} km²", f"= {area_hectares:.2f} hectares"],
        ["Approximate Dimensions", f"{_calc_dimensions(bbox)}", "E-W × N-S"],
    ]

    if layer2_output:
        loc = layer2_output.get('location', {})
        if loc.get('state'):
            data.append(["State/Region", loc.get('state'), ""])
        if loc.get('country'):
            data.append(["Country", loc.get('country'), ""])

    table = _data_table(data)
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    # Land use context
    if layer3_context and layer3_context.get('land_use'):
        story.append(Paragraph('<font size="11" color="#0f1c18"><b>Land Use Context</b></font>', _style_left()))
        story.append(Spacer(1, 0.1 * inch))

        land_use = layer3_context.get('land_use', {})
        land_data = [
            ["Attribute", "Value"],
            ["Primary Land Use", land_use.get('land_use_type', 'Unknown').title()],
            ["Urban Percentage", f"{land_use.get('urban_percentage', 0):.1f}%"],
            ["Classification Confidence", land_use.get('gemini_confidence', 'N/A')],
        ]

        if land_use.get('land_use_description'):
            story.append(Paragraph(
                f'<font size="9" color="#5a6b66">{land_use.get("land_use_description")}</font>',
                _style_left_wrapped()
            ))
            story.append(Spacer(1, 0.1 * inch))

        story.append(_data_table(land_data))

    return story


def _build_severity_section(satellite_image, severity_image, severity_stats, area_km2):
    """Build comprehensive burn severity section."""
    story = []

    story.append(_section_header("3. Burn Severity Analysis"))

    # Images
    if satellite_image and severity_image:
        story.append(Paragraph('<font size="11" color="#0f1c18"><b>Satellite Imagery Analysis</b></font>', _style_left()))
        story.append(Spacer(1, 0.1 * inch))

        images_table = create_side_by_side_images(
            satellite_image, severity_image,
            total_width=PAGE_WIDTH,
            spacing=0.2 * inch
        )
        story.append(images_table)

        captions = Table(
            [[
                Paragraph('<font size="8" color="#8a9a95">Figure 1a: Sentinel-2 False Color Composite (B5-B4-B3)</font>', _style_center()),
                Paragraph('<font size="8" color="#8a9a95">Figure 1b: U-Net Burn Severity Prediction</font>', _style_center())
            ]],
            colWidths=[PAGE_WIDTH / 2, PAGE_WIDTH / 2]
        )
        story.append(captions)
        story.append(Spacer(1, 0.25 * inch))

    # Severity classification table
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Severity Classification</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    # Safely extract severity values with type coercion
    high = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100
    mod = float(severity_stats.get('moderate_severity_ratio', 0) or 0) * 100
    low = float(severity_stats.get('low_severity_ratio', 0) or 0) * 100
    mean = float(severity_stats.get('mean_severity', 0) or 0) * 100

    data = [
        ["Severity Class", "Threshold", "Coverage (%)", "Area (km²)", "Ecological Impact"],
        ["High Severity", "> 75%", f"{high:.1f}%", f"{high * area_km2 / 100:.3f}", "Stand-replacing fire; complete canopy mortality"],
        ["Moderate Severity", "25-75%", f"{mod:.1f}%", f"{mod * area_km2 / 100:.3f}", "Mixed mortality; understory affected"],
        ["Low/Unburned", "< 25%", f"{low:.1f}%", f"{low * area_km2 / 100:.3f}", "Surface fire or unburned; canopy intact"],
    ]

    table = _data_table(data)
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    # Additional statistics
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Statistical Summary</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    stats_data = [
        ["Metric", "Value", "Interpretation"],
        ["Mean Severity", f"{mean:.1f}%", _interpret_severity(mean)],
        ["Maximum Severity", f"{float(severity_stats.get('max_severity', 0) or 0) * 100:.1f}%", "Hottest detected burn signature"],
        ["Minimum Severity", f"{float(severity_stats.get('min_severity', 0) or 0) * 100:.1f}%", "Least affected areas"],
        ["Burned Ratio (>50%)", f"{float(severity_stats.get('burned_ratio', 0) or 0) * 100:.1f}%", "Proportion with significant burn"],
    ]

    story.append(_data_table(stats_data))

    return story


def _build_biophysical_section(layer2_output, layer3_context):
    """Build biophysical characterization section."""
    story = []

    story.append(_section_header("4. Biophysical Characterization"))

    if layer2_output:
        chars = layer2_output.get('characteristics', {})
        if chars:
            story.append(Paragraph('<font size="11" color="#0f1c18"><b>Site Characteristics</b></font>', _style_left()))
            story.append(Spacer(1, 0.1 * inch))

            data = [["Characteristic", "Value", "Restoration Implications"]]

            if chars.get('soil_type'):
                data.append(["Soil Type", chars['soil_type'], _soil_implications(chars['soil_type'])])
            if chars.get('terrain'):
                data.append(["Terrain", chars['terrain'], "Affects equipment access and erosion risk"])
            if chars.get('slope_percent'):
                slope = chars['slope_percent']
                data.append(["Slope", f"{slope}%", "High" if slope > 30 else "Moderate" if slope > 15 else "Low" + " erosion risk"])
            if chars.get('aspect'):
                data.append(["Aspect", chars['aspect'], _aspect_implications(chars['aspect'])])
            if chars.get('elevation_m'):
                data.append(["Elevation", f"{chars['elevation_m']} m", "Affects species selection and climate"])

            if len(data) > 1:
                story.append(_data_table(data))
                story.append(Spacer(1, 0.2 * inch))

    # Ecosystem classification
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Ecosystem Classification</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    eco_type = "Mixed Conifer Forest"  # Default
    if layer3_context and layer3_context.get('land_use'):
        land_type = layer3_context['land_use'].get('land_use_type', '')
        if 'forest' in land_type.lower():
            eco_type = "Mixed Conifer Forest"
        elif 'shrub' in land_type.lower() or 'chaparral' in land_type.lower():
            eco_type = "Chaparral Shrubland"
        elif 'grass' in land_type.lower():
            eco_type = "Grassland"
        elif 'oak' in land_type.lower():
            eco_type = "Oak Woodland"

    story.append(Paragraph(
        f'<font size="10" color="#5a6b66">Based on location and land cover analysis, this site is classified as '
        f'<b>{eco_type}</b>. This classification informs species selection and carbon accounting parameters.</font>',
        _style_left_wrapped()
    ))

    return story


def _build_ecological_section(severity_stats, layer2_output, layer3_context):
    """Build ecological assessment section."""
    story = []

    story.append(_section_header("5. Ecological Assessment"))

    high = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100

    # Vegetation impact
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Vegetation Impact Assessment</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    if high > 50:
        impact_text = '''
        <font size="10" color="#5a6b66">
        The analysis indicates <b>severe vegetation loss</b> across a majority of the site. High-severity burn
        typically results in complete canopy mortality and significant soil damage. Natural regeneration may be
        limited by lack of viable seed sources. Active reforestation will likely be required for timely recovery.
        </font>
        '''
    elif high > 25:
        impact_text = '''
        <font size="10" color="#5a6b66">
        The analysis indicates <b>moderate vegetation impact</b> with significant patches of high-severity burn.
        Mixed conditions suggest potential for both natural regeneration and active restoration. Priority should be
        given to high-severity areas while monitoring natural recovery in moderate zones.
        </font>
        '''
    else:
        impact_text = '''
        <font size="10" color="#5a6b66">
        The analysis indicates <b>relatively limited high-severity impact</b>. The presence of intact seed sources
        and surviving vegetation suggests good potential for natural regeneration. Monitoring and targeted erosion
        control may be sufficient, with active planting focused on specific problem areas.
        </font>
        '''

    story.append(Paragraph(impact_text, _style_left_wrapped()))
    story.append(Spacer(1, 0.2 * inch))

    # Recovery potential
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Recovery Potential Assessment</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    recovery_data = [
        ["Factor", "Assessment", "Notes"],
        ["Natural Regeneration Potential", "High" if high < 30 else "Moderate" if high < 60 else "Low", "Based on seed source availability"],
        ["Erosion Risk", "High" if high > 40 else "Moderate" if high > 20 else "Low", "Requires stabilization if high"],
        ["Intervention Urgency", "Critical" if high > 60 else "High" if high > 40 else "Moderate", "Timing affects success"],
        ["Expected Recovery Time", f"{_estimate_recovery_years(high)} years", "To 80% pre-fire vegetation"],
    ]

    story.append(_data_table(recovery_data))

    return story


def _build_carbon_section(pro_carbon, area_hectares):
    """Build comprehensive carbon accounting section."""
    story = []

    story.append(_section_header("6. Carbon Accounting"))

    # Ensure area_hectares is a float
    area_hectares = float(area_hectares or 0)

    # Methodology note
    methodology = pro_carbon.get('methodology', 'IPCC Tier 2')
    story.append(Paragraph(
        f'<font size="10" color="#00d4aa"><b>Methodology:</b> {methodology}</font>',
        _style_left()
    ))
    story.append(Spacer(1, 0.15 * inch))

    # Carbon stock table
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Carbon Stock Assessment</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    data = [
        ["Metric", "Value", "Unit", "Notes"],
        ["Site Area", f"{area_hectares:.2f}", "hectares", "Analysis boundary"],
        ["Pre-Fire Baseline", f"{pro_carbon.get('baseline_carbon_tc', 0):.1f}", "tC", "Estimated from regional averages"],
        ["Carbon Lost to Fire", f"{pro_carbon.get('carbon_lost_tc', 0):.1f}", "tC", "Based on severity-weighted loss"],
        ["Current Carbon Stock", f"{pro_carbon.get('current_carbon_tc', 0):.1f}", "tC", "Baseline minus loss"],
        ["Annual Sequestration Rate", f"{pro_carbon.get('annual_sequestration_tc', 0):.2f}", "tC/yr", "Ecosystem-specific rate"],
        ["Annual CO₂ Equivalent", f"{pro_carbon.get('annual_sequestration_tco2e', 0):.1f}", "tCO₂e/yr", "× 3.67 conversion factor"],
    ]
    story.append(_data_table(data))
    story.append(Spacer(1, 0.25 * inch))

    # Projections
    projections = pro_carbon.get('projections', [])
    if projections:
        story.append(Paragraph('<font size="11" color="#0f1c18"><b>Sequestration Projections</b></font>', _style_left()))
        story.append(Spacer(1, 0.1 * inch))

        proj_data = [["Time Horizon", "Cumulative tC", "Cumulative tCO₂e", "Annual Rate"]]
        for proj in projections:
            proj_data.append([
                f"{proj.get('years', 0)} years",
                f"{proj.get('cumulative_tc', 0):,.1f}",
                f"{proj.get('cumulative_tco2e', 0):,.0f}",
                f"{proj.get('annual_rate_tco2e', 0):.1f} tCO₂e/yr"
            ])
        story.append(_data_table(proj_data))
        story.append(Spacer(1, 0.25 * inch))

    # Protocol eligibility
    protocols = pro_carbon.get('protocols', {})
    if protocols:
        story.append(Paragraph('<font size="11" color="#0f1c18"><b>Carbon Credit Protocol Eligibility</b></font>', _style_left()))
        story.append(Spacer(1, 0.1 * inch))

        prot_data = [["Protocol", "Eligibility", "Minimum Requirements"]]
        protocol_reqs = {
            "verra_vcs_eligible": ("Verra VCS", "≥10 ha, ≥100 tCO₂e"),
            "gold_standard_eligible": ("Gold Standard", "≥5 ha, ≥50 tCO₂e"),
            "plan_vivo_eligible": ("Plan Vivo", "Community-scale, no minimum"),
            "california_arb_eligible": ("California ARB", "≥40 ha (100 acres)"),
            "american_carbon_registry": ("American Carbon Registry", "≥20 ha"),
        }

        for key, (name, req) in protocol_reqs.items():
            eligible = protocols.get(key, False)
            status = "✓ ELIGIBLE" if eligible else "✗ Not Eligible"
            prot_data.append([name, status, req])

        story.append(_data_table(prot_data))
        story.append(Spacer(1, 0.25 * inch))

    # Uncertainty
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Uncertainty Quantification</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    uncertainty_pct = pro_carbon.get('uncertainty_pct', 35)
    ci_low = pro_carbon.get('confidence_interval_low', 0)
    ci_high = pro_carbon.get('confidence_interval_high', 0)

    # Get 20-year projection
    proj_20yr = next((p for p in projections if p.get('years') == 20), {})
    central = proj_20yr.get('cumulative_tco2e', 0)

    unc_text = f'''
    <font size="10" color="#5a6b66">
    <b>Combined Uncertainty:</b> ±{uncertainty_pct:.0f}%<br/><br/>
    <b>95% Confidence Interval (20-year):</b> {ci_low:,.0f} – {ci_high:,.0f} tCO₂e<br/><br/>
    <b>Sources of Uncertainty:</b><br/>
    • Carbon accumulation rates: ±25% (ecosystem variability)<br/>
    • Baseline carbon stock: ±30% (regional estimates)<br/>
    • Severity impact assessment: ±20% (model prediction error)<br/><br/>
    <i>Note: Actual sequestration depends on restoration success, species survival rates, and climate factors.</i>
    </font>
    '''
    story.append(Paragraph(unc_text, _style_left_wrapped()))

    return story


def _build_risk_section(layer2_output, severity_stats):
    """Build risk assessment section."""
    story = []

    story.append(_section_header("7. Risk Assessment"))

    high = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100

    # Hazard identification
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Identified Hazards</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    hazards = []
    if layer2_output and layer2_output.get('hazards'):
        hazards = layer2_output['hazards'][:6]

    if not hazards:
        # Generate default hazards based on severity
        if high > 40:
            hazards = [
                {"type": "Standing Dead Trees", "severity": "high", "description": "Widowmaker risk in high-severity zones"},
                {"type": "Soil Instability", "severity": "moderate", "description": "Erosion and debris flow risk on slopes"},
                {"type": "Ash Pits", "severity": "moderate", "description": "Hidden root holes where trees burned underground"},
            ]
        elif high > 20:
            hazards = [
                {"type": "Scattered Hazard Trees", "severity": "moderate", "description": "Some standing dead trees present"},
                {"type": "Erosion Risk", "severity": "moderate", "description": "Exposed soil susceptible to rain events"},
            ]
        else:
            hazards = [
                {"type": "Low Hazard Conditions", "severity": "low", "description": "Standard outdoor safety precautions apply"},
            ]

    haz_data = [["Hazard Type", "Severity", "Description/Mitigation"]]
    for hazard in hazards:
        haz_data.append([
            hazard.get('type', 'Unknown'),
            hazard.get('severity', 'unknown').title(),
            hazard.get('description', 'N/A')[:80]
        ])

    story.append(_data_table(haz_data))
    story.append(Spacer(1, 0.2 * inch))

    # Overall risk level
    if high > 50:
        risk_level = "HIGH"
        risk_color = "#e63946"
    elif high > 25:
        risk_level = "MODERATE"
        risk_color = "#f4a261"
    else:
        risk_level = "LOW"
        risk_color = "#2a9d8f"

    story.append(Paragraph(
        f'<font size="11" color="#0f1c18"><b>Overall Site Risk Level:</b></font> '
        f'<font size="11" color="{risk_color}"><b>{risk_level}</b></font>',
        _style_left()
    ))

    return story


def _build_recommendations_section(severity_stats, layer3_context, carbon_analysis):
    """Build prioritized recommendations section."""
    story = []

    story.append(_section_header("8. Recommendations"))

    high = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100

    # Immediate actions
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Immediate Actions (0-6 months)</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    immediate = []
    if high > 40:
        immediate = [
            "Install erosion control measures (straw wattles, log check dams) before rainy season",
            "Conduct hazard tree assessment and removal in areas of public access",
            "Establish photo monitoring points for recovery tracking",
            "Secure site access and post safety warnings",
        ]
    else:
        immediate = [
            "Establish baseline monitoring transects",
            "Document natural regeneration patterns",
            "Identify priority areas for intervention",
        ]

    for action in immediate:
        story.append(Paragraph(f'<font size="10" color="#5a6b66">• {action}</font>', _style_left()))
    story.append(Spacer(1, 0.2 * inch))

    # Short-term actions
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Short-Term Actions (6-24 months)</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    short_term = [
        "Develop species palette with local native plant nurseries",
        "Plan and implement reforestation in high-severity areas",
        "Establish wildlife corridors connecting unburned patches",
        "Monitor for invasive species establishment and implement control",
    ]

    for action in short_term:
        story.append(Paragraph(f'<font size="10" color="#5a6b66">• {action}</font>', _style_left()))
    story.append(Spacer(1, 0.2 * inch))

    # Long-term actions
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Long-Term Actions (2-10 years)</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    long_term = [
        "Continue vegetation monitoring using satellite imagery (annual NDVI assessment)",
        "Implement adaptive management based on survival rates and growth",
        "Pursue carbon credit verification if eligible",
        "Document lessons learned for future restoration projects",
    ]

    for action in long_term:
        story.append(Paragraph(f'<font size="10" color="#5a6b66">• {action}</font>', _style_left()))

    return story


def _build_methodology_section(layer2_output, layer3_context):
    """Build methodology and data sources section."""
    story = []

    story.append(_section_header("9. Data Sources & Methodology"))

    # Data sources
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Data Sources</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    processing = layer2_output.get('processing', {}) if layer2_output else {}
    imagery_date = processing.get('imagery_date', 'specified date range')
    # Safely extract confidence - might be dict or number
    raw_confidence = layer2_output.get('confidence', 0.85) if layer2_output else 0.85
    confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.85

    sources = [
        f"<b>Satellite Imagery:</b> Sentinel-2 MSI Level-2A (10m resolution), acquired {imagery_date}",
        f"<b>Burn Severity Model:</b> U-Net convolutional neural network trained on California wildfires ({confidence*100:.0f}% confidence)",
        "<b>Processing Platform:</b> Google Earth Engine for imagery acquisition and preprocessing",
        "<b>AI Analysis:</b> Google Gemini multimodal model for pattern recognition and context analysis",
        "<b>Carbon Accounting:</b> IPCC 2006 Guidelines with 2019 Refinement, California-specific coefficients",
    ]

    for src in sources:
        story.append(Paragraph(f'<font size="10" color="#5a6b66">{src}</font>', _style_left_wrapped()))
        story.append(Spacer(1, 0.06 * inch))
    story.append(Spacer(1, 0.15 * inch))

    # Methodology
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Analysis Pipeline</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    steps = [
        ("1. Satellite Acquisition", f"Sentinel-2 imagery retrieved via Earth Engine API for {imagery_date}"),
        ("2. Preprocessing", "Atmospheric correction, cloud masking, band normalization to training statistics"),
        ("3. Burn Severity Prediction", f"U-Net model inference with {confidence*100:.0f}% confidence threshold"),
        ("4. Spatial Analysis", "Zone extraction, severity classification, area calculations"),
        ("5. AI Enhancement", "Gemini multimodal analysis for pattern recognition and anomaly detection"),
        ("6. Carbon Accounting", "IPCC Tier 2 methodology with Monte Carlo uncertainty propagation"),
    ]

    for step, desc in steps:
        story.append(Paragraph(
            f'<font size="10" color="#0f1c18"><b>{step}:</b></font> '
            f'<font size="10" color="#5a6b66">{desc}</font>',
            _style_left_wrapped()
        ))
        story.append(Spacer(1, 0.04 * inch))
    story.append(Spacer(1, 0.15 * inch))

    # Limitations
    story.append(Paragraph('<font size="11" color="#0f1c18"><b>Limitations & Caveats</b></font>', _style_left()))
    story.append(Spacer(1, 0.1 * inch))

    limitations = [
        "Model predictions based on satellite imagery may differ from ground-truth conditions",
        "Carbon estimates require field verification for carbon credit applications",
        "Pre-fire baseline estimated from regional averages, not site-specific measurements",
        "Cloud cover and atmospheric conditions may affect imagery quality",
        "Fire perimeter and timing assumptions based on available data",
    ]

    for lim in limitations:
        story.append(Paragraph(f'<font size="10" color="#5a6b66">• {lim}</font>', _style_left_wrapped()))
        story.append(Spacer(1, 0.04 * inch))

    return story


# ==================== HELPER FUNCTIONS ====================

def _section_header(text):
    """Create a section header with green underline."""
    header = Paragraph(
        f'<font size="14" color="#0f1c18"><b>{text}</b></font>',
        _style_left()
    )
    line = HRFlowable(width="25%", thickness=2, color=ACCENT_GREEN, spaceBefore=4, spaceAfter=12, hAlign='LEFT')
    wrapper = Table([[header], [line]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return wrapper


def _metric_box(value, label):
    """Create a metric display box."""
    data = [
        [Paragraph(f'<font size="20" color="#0f1c18"><b>{value}</b></font>', _style_center())],
        [Paragraph(f'<font size="8" color="#8a9a95">{label}</font>', _style_center())],
    ]
    box = Table(data, colWidths=[1.5 * inch])
    box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    return box


def _data_table(data):
    """Create a styled data table with proper alignment."""
    # Calculate column widths based on content
    num_cols = len(data[0]) if data else 0
    # Reduce width slightly to prevent overflow
    available_width = PAGE_WIDTH - 0.2 * inch
    col_width = available_width / num_cols if num_cols > 0 else available_width

    table = Table(data, colWidths=[col_width] * num_cols, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), TEXT_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TABLE_ALT_ROW]),
    ]))
    return table


def _calc_dimensions(bbox):
    """Calculate approximate E-W x N-S dimensions."""
    lat_mid = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
    km_per_deg_lon = 111.32 * 0.9  # Approximate
    km_per_deg_lat = 110.574

    width_km = abs(bbox.get('east', 0) - bbox.get('west', 0)) * km_per_deg_lon
    height_km = abs(bbox.get('north', 0) - bbox.get('south', 0)) * km_per_deg_lat

    return f"{width_km:.1f} × {height_km:.1f} km"


def _interpret_severity(mean):
    """Interpret mean severity value."""
    if mean > 60:
        return "Severe impact; active restoration required"
    elif mean > 35:
        return "Moderate impact; mixed intervention needed"
    else:
        return "Lower impact; monitor natural recovery"


def _estimate_recovery_years(high_severity_pct):
    """Estimate years to recovery based on severity."""
    if high_severity_pct > 60:
        return "20-30"
    elif high_severity_pct > 40:
        return "15-20"
    elif high_severity_pct > 20:
        return "10-15"
    else:
        return "5-10"


def _soil_implications(soil_type):
    """Get restoration implications for soil type."""
    implications = {
        "clay": "May need drainage; good water retention",
        "loam": "Ideal for planting; good nutrient availability",
        "sand": "Fast drainage; may need amendments",
        "rocky": "Limited root space; erosion-resistant",
    }
    return implications.get(soil_type.lower(), "Assess site-specific conditions")


def _aspect_implications(aspect):
    """Get implications for slope aspect."""
    if aspect in ["N", "NE", "NW"]:
        return "Cooler, moister microclimate; shade-tolerant species"
    elif aspect in ["S", "SE", "SW"]:
        return "Warmer, drier conditions; drought-tolerant species"
    return "Mixed exposure conditions"


def _get_priority_recommendations(severity_stats, layer3_context, carbon_analysis):
    """Generate priority recommendations based on analysis."""
    high = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100
    recs = []

    if high > 40:
        recs.append("Implement immediate erosion control measures before the next rainy season.")
        recs.append("Conduct hazard tree assessment and plan removal in high-traffic areas.")
    else:
        recs.append("Establish monitoring transects to track natural regeneration progress.")

    recs.append("Develop a native species planting plan tailored to site conditions.")

    if carbon_analysis and carbon_analysis.get('professional', {}).get('protocols', {}):
        eligible_count = sum(1 for v in carbon_analysis['professional']['protocols'].values() if v)
        if eligible_count > 0:
            recs.append(f"Pursue carbon credit verification (eligible for {eligible_count} protocols).")

    recs.append("Engage local stakeholders and potential volunteer groups for restoration activities.")

    return recs


def _build_consultation_log_appendix(chat_history: List[Dict[str, Any]]) -> list:
    """Build appendix section for AI consultation log."""
    story = []
    
    story.append(_section_header("Appendix A: AI Consultation Log"))
    
    # Introduction
    intro_text = '''
    <font size="10" color="#5a6b66">
    The following is a record of questions and responses during the AI-assisted analysis session.
    This log is provided for transparency and to document the analytical process. All AI responses
    should be verified independently before use in decision-making.
    </font>
    '''
    story.append(Paragraph(intro_text, _style_left_wrapped()))
    story.append(Spacer(1, 0.15 * inch))
    
    # Limit messages
    messages = chat_history[-20:] if len(chat_history) > 20 else chat_history
    
    if len(chat_history) > 20:
        story.append(Paragraph(
            f'<font size="9" color="#8a9a95"><i>(Showing last 20 of {len(chat_history)} interactions)</i></font>',
            _style_left()
        ))
        story.append(Spacer(1, 0.1 * inch))
    
    # Build consultation table
    data = [["Time", "Role", "Content"]]
    
    for msg in messages:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', '')
        
        # Format timestamp
        time_str = "—"
        if timestamp:
            try:
                from datetime import datetime as dt
                ts = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = ts.strftime("%H:%M")
            except:
                pass
        
        # Truncate long content for table display
        if len(content) > 300:
            content = content[:300] + "..."
        
        # Role styling
        if role == 'User':
            role_display = Paragraph('<font size="9" color="#006494"><b>User</b></font>', _style_left())
        else:
            role_display = Paragraph('<font size="9" color="#00856a"><b>AI</b></font>', _style_left())
        
        content_para = Paragraph(f'<font size="9" color="#5a6b66">{content}</font>', _style_left_wrapped())
        time_para = Paragraph(f'<font size="8" color="#8a9a95">{time_str}</font>', _style_center())
        
        data.append([time_para, role_display, content_para])
    
    table = Table(data, colWidths=[0.5 * inch, 0.5 * inch, 5.0 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TABLE_ALT_ROW]),
    ]))
    
    story.append(table)
    
    return story


# Style helper functions
def _style_left():
    return ParagraphStyle(name='Left', alignment=TA_LEFT)


def _style_left_wrapped():
    return ParagraphStyle(name='LeftWrap', alignment=TA_LEFT, leading=14)


def _style_center():
    return ParagraphStyle(name='Center', alignment=TA_CENTER)


def _style_right():
    return ParagraphStyle(name='Right', alignment=TA_RIGHT)
