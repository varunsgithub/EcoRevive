"""
Personal Report Generator for EcoRevive
=========================================
Modern, card-based 2-3 page Impact Card design with detailed sections.
"""

from datetime import datetime
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle

from .styles import get_styles, ECOREVIVE_GREEN, BORDER_COLOR
from .utils import (
    base64_to_pil, pil_to_reportlab_image, create_side_by_side_images,
    format_coordinates, calculate_area_km2, truncate_text
)

# Design constants
PRIMARY_DARK = HexColor("#0f1c18")
PRIMARY_GREEN = HexColor("#00d4aa")
CARD_BG = HexColor("#f4f7f6")
WHITE = HexColor("#ffffff")
GRAY_TEXT = HexColor("#5a6b66")
LIGHT_GRAY = HexColor("#8a9a95")
SEVERITY_HIGH = HexColor("#e63946")
SEVERITY_MOD = HexColor("#f4a261")
SEVERITY_LOW = HexColor("#2a9d8f")

PAGE_WIDTH = 6.0 * inch  # Reduced for better margins


def build_personal_report(
    satellite_image: str,
    severity_image: str,
    severity_stats: dict,
    bbox: dict,
    carbon_analysis: dict = None,
    layer3_context: dict = None,
    location_name: str = None,
    analysis_id: str = None,
) -> list:
    """
    Build the story elements for a personal Impact Card PDF.
    Enhanced with more detail and better alignment.
    """
    styles = get_styles()
    story = []

    # Calculate values
    area_km2 = calculate_area_km2(bbox)
    area_hectares = area_km2 * 100
    coordinates = format_coordinates(bbox)
    timestamp = datetime.now().strftime("%B %d, %Y")
    time_short = datetime.now().strftime("%m/%d/%Y")

    if not location_name:
        location_name = "Burn Analysis Site"

    # ==================== PAGE 1: HEADER & OVERVIEW ====================

    # Dark Header Banner
    story.append(_create_header_banner(location_name, coordinates, area_km2, timestamp, analysis_id))
    story.append(Spacer(1, 0.3 * inch))

    # Executive Summary Box
    story.append(_create_summary_box(severity_stats, area_km2, carbon_analysis))
    story.append(Spacer(1, 0.3 * inch))

    # Satellite Imagery Section
    if satellite_image and severity_image:
        story.append(_section_label("SATELLITE ANALYSIS"))
        story.append(Spacer(1, 0.1 * inch))

        images_table = create_side_by_side_images(
            satellite_image, severity_image,
            total_width=PAGE_WIDTH,
            spacing=0.15 * inch
        )
        story.append(images_table)

        # Image captions
        captions = Table(
            [[
                Paragraph('<font size="8" color="#8a9a95">Sentinel-2 Satellite View</font>', _style_center()),
                Paragraph('<font size="8" color="#8a9a95">AI Burn Severity Map</font>', _style_center())
            ]],
            colWidths=[PAGE_WIDTH / 2, PAGE_WIDTH / 2]
        )
        captions.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(captions)
        story.append(Spacer(1, 0.25 * inch))

    # Severity Breakdown with Visual Bar
    story.append(_section_label("BURN SEVERITY BREAKDOWN"))
    story.append(Spacer(1, 0.12 * inch))

    # Safely extract severity values with type coercion
    high_sev = float(severity_stats.get('high_severity_ratio', 0) or 0) * 100
    mod_sev = float(severity_stats.get('moderate_severity_ratio', 0) or 0) * 100
    low_sev = float(severity_stats.get('low_severity_ratio', 0) or 0) * 100
    mean_sev = float(severity_stats.get('mean_severity', 0) or 0) * 100

    story.append(_create_severity_bar(high_sev, mod_sev, low_sev))
    story.append(Spacer(1, 0.15 * inch))

    # Severity details table
    story.append(_create_severity_details(high_sev, mod_sev, low_sev, mean_sev, area_km2))
    story.append(Spacer(1, 0.3 * inch))

    # ==================== CARBON IMPACT SECTION ====================
    if carbon_analysis and carbon_analysis.get('personal'):
        story.append(_section_label("RESTORATION IMPACT POTENTIAL"))
        story.append(Spacer(1, 0.12 * inch))

        personal_carbon = carbon_analysis['personal']
        total_co2 = personal_carbon.get('total_co2_capture_20yr', 0)
        annual_co2 = personal_carbon.get('annual_co2_capture', 0)
        equivalencies = personal_carbon.get('equivalencies', {})

        story.append(_create_carbon_impact_card(total_co2, annual_co2, equivalencies))
        story.append(Spacer(1, 0.15 * inch))

        # Impact statements
        impact_statements = personal_carbon.get('impact_statements', [])
        if impact_statements:
            story.append(_create_impact_statements(impact_statements))
        story.append(Spacer(1, 0.3 * inch))

    # ==================== PAGE 2: WHAT YOU CAN DO ====================
    story.append(PageBreak())

    # Land Use Context (if available)
    if layer3_context and layer3_context.get('land_use'):
        story.append(_section_label("SITE CONTEXT"))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_create_land_use_section(layer3_context))
        story.append(Spacer(1, 0.25 * inch))

    # Action Items
    story.append(_section_label("HOW YOU CAN HELP"))
    story.append(Spacer(1, 0.12 * inch))

    actions = _get_detailed_recommendations(layer3_context, severity_stats)
    story.append(_create_action_grid(actions))
    story.append(Spacer(1, 0.25 * inch))

    # Safety Notice
    story.append(_create_safety_notice(severity_stats))
    story.append(Spacer(1, 0.25 * inch))

    # Timeline Expectations
    story.append(_section_label("RECOVERY TIMELINE"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_create_timeline_section())
    story.append(Spacer(1, 0.3 * inch))

    # Footer
    story.append(_create_footer(timestamp, analysis_id))

    return story


def _create_header_banner(location_name, coordinates, area_km2, timestamp, analysis_id):
    """Create the dark header banner with site info - centered alignment."""
    area_hectares = area_km2 * 100

    # Title - centered
    title_para = Paragraph(
        f'<font size="20" color="#ffffff"><b>{truncate_text(location_name, 40)}</b></font>',
        _style_center()
    )

    # Metadata - centered
    meta_text = f'{coordinates}  |  {area_km2:.2f} km² ({area_hectares:.0f} ha)  |  {timestamp}'
    meta_para = Paragraph(
        f'<font size="9" color="#7a9a8a">{meta_text}</font>',
        _style_center()
    )

    inner_data = [
        [title_para],
        [meta_para],
    ]

    inner_table = Table(inner_data, colWidths=[PAGE_WIDTH - 0.3 * inch])
    inner_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    wrapper = Table([[inner_table]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_DARK),
        ('TOPPADDING', (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
    ]))

    return wrapper


def _create_summary_box(severity_stats, area_km2, carbon_analysis):
    """Create a quick summary box at the top."""
    mean_sev = severity_stats.get('mean_severity', 0) * 100
    high_sev = severity_stats.get('high_severity_ratio', 0) * 100

    # Determine severity level text
    if mean_sev > 60:
        sev_text = "Severe Impact"
        sev_color = "#e63946"
    elif mean_sev > 35:
        sev_text = "Moderate Impact"
        sev_color = "#f4a261"
    else:
        sev_text = "Low-Moderate Impact"
        sev_color = "#2a9d8f"

    co2_text = ""
    if carbon_analysis and carbon_analysis.get('personal'):
        co2 = carbon_analysis['personal'].get('total_co2_capture_20yr', 0)
        co2_text = f'{int(co2):,} tons CO₂ capture potential over 20 years'

    summary_content = f'''
    <font size="11" color="#0f1c18"><b>Quick Summary:</b></font><br/>
    <font size="10" color="#5a6b66">
    This {area_km2:.1f} km² site shows <font color="{sev_color}"><b>{sev_text}</b></font>
    with {mean_sev:.0f}% average burn severity and {high_sev:.0f}% high-severity areas.
    {f"<br/><br/><b>Carbon Impact:</b> {co2_text}" if co2_text else ""}
    </font>
    '''

    summary_para = Paragraph(summary_content, _style_left_wrapped())

    wrapper = Table([[summary_para]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))

    return wrapper


def _section_label(text):
    """Create a small section label - centered for professional look."""
    return Paragraph(
        f'<font size="10" color="#5a6b66"><b>{text}</b></font>',
        _style_center()
    )


def _create_severity_bar(high, mod, low):
    """Create a horizontal stacked bar showing severity percentages."""
    bar_height = 0.32 * inch

    # Ensure minimum width for visibility
    high_width = max(0.02, high / 100) * PAGE_WIDTH
    mod_width = max(0.02, mod / 100) * PAGE_WIDTH
    low_width = max(0.02, low / 100) * PAGE_WIDTH

    bar_data = [[
        Paragraph(f'<font size="10" color="#ffffff"><b>{high:.0f}%</b></font>', _style_center()) if high > 10 else '',
        Paragraph(f'<font size="10" color="#ffffff"><b>{mod:.0f}%</b></font>', _style_center()) if mod > 10 else '',
        Paragraph(f'<font size="10" color="#ffffff"><b>{low:.0f}%</b></font>', _style_center()) if low > 10 else '',
    ]]

    bar = Table(bar_data, colWidths=[high_width, mod_width, low_width], rowHeights=[bar_height])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), SEVERITY_HIGH),
        ('BACKGROUND', (1, 0), (1, 0), SEVERITY_MOD),
        ('BACKGROUND', (2, 0), (2, 0), SEVERITY_LOW),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))

    # Legend
    legend_data = [[
        Paragraph('<font size="8" color="#e63946">■</font> <font size="8" color="#5a6b66">High Severity</font>', _style_left()),
        Paragraph('<font size="8" color="#f4a261">■</font> <font size="8" color="#5a6b66">Moderate</font>', _style_center()),
        Paragraph('<font size="8" color="#2a9d8f">■</font> <font size="8" color="#5a6b66">Low/Unburned</font>', _style_right()),
    ]]
    legend = Table(legend_data, colWidths=[PAGE_WIDTH / 3] * 3)
    legend.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 8)]))

    return Table([[bar], [legend]], colWidths=[PAGE_WIDTH])


def _create_severity_details(high, mod, low, mean, area_km2):
    """Create detailed severity breakdown table."""
    high_area = high * area_km2 / 100
    mod_area = mod * area_km2 / 100
    low_area = low * area_km2 / 100

    data = [
        ['Category', '%', 'Area', 'What This Means'],
        ['High Severity', f'{high:.1f}%', f'{high_area:.2f}km²', 'Complete vegetation loss'],
        ['Moderate', f'{mod:.1f}%', f'{mod_area:.2f}km²', 'Partial damage, recovery likely'],
        ['Low/Unburned', f'{low:.1f}%', f'{low_area:.2f}km²', 'Minimal impact'],
    ]

    table = Table(data, colWidths=[1.2 * inch, 0.7 * inch, 0.8 * inch, 3.0 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), GRAY_TEXT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, CARD_BG]),
    ]))

    return table


def _create_carbon_impact_card(total_co2, annual_co2, equivalencies):
    """Create the carbon impact display card."""
    cars = equivalencies.get('cars_off_road_for_year', 0)
    trees = equivalencies.get('tree_seedlings_grown_10yr', 0)
    flights = equivalencies.get('round_trip_flights_nyc_la', 0)
    homes = equivalencies.get('homes_electricity_year', 0)

    # Large number display - smaller font for better fit
    big_number = Paragraph(
        f'<font size="32" color="#00d4aa"><b>{int(total_co2):,}</b></font>',
        _style_center()
    )
    unit_label = Paragraph(
        '<font size="10" color="#5a6b66">tons CO2 captured over 20 years</font>',
        _style_center()
    )

    number_block = Table([[big_number], [unit_label]], colWidths=[PAGE_WIDTH - 0.4 * inch])
    number_block.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    # Equivalency grid (2x2) - smaller to avoid overflow
    half_width = (PAGE_WIDTH - 0.4 * inch) / 2
    equiv_data = [
        [_equiv_item("", f"{int(cars):,}", "cars off road/year"),
         _equiv_item("", f"{int(trees):,}", "seedlings grown 10yr")],
        [_equiv_item("", f"{int(flights):,}", "NYC-LA flights"),
         _equiv_item("", f"{int(homes):,}", "homes powered/year")],
    ]

    equiv_table = Table(equiv_data, colWidths=[half_width, half_width], rowHeights=[0.55 * inch] * 2)
    equiv_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))

    # Wrap everything in a card
    card_data = [[number_block], [equiv_table]]
    card = Table(card_data, colWidths=[PAGE_WIDTH - 0.2 * inch])
    card.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
    ]))

    return card


def _equiv_item(icon, value, label):
    """Create a single equivalency item."""
    icon_part = f'<font size="16">{icon}</font> ' if icon else ''
    content = Paragraph(
        f'{icon_part}<font size="18" color="#0f1c18"><b>{value}</b></font><br/>'
        f'<font size="8" color="#8a9a95">{label}</font>',
        _style_center()
    )
    return content


def _create_impact_statements(statements):
    """Create impact statements section."""
    content = '<font size="10" color="#5a6b66">'
    for i, stmt in enumerate(statements[:4]):
        bullet = "*" if i == len(statements) - 1 else "-"
        content += f'{bullet} {stmt}<br/>'
    content += '</font>'

    para = Paragraph(content, _style_left_wrapped())
    wrapper = Table([[para]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#e8f5f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    return wrapper


def _create_land_use_section(layer3_context):
    """Create land use context section."""
    land_use = layer3_context.get('land_use', {})
    land_type = land_use.get('land_use_type', 'Unknown').title()
    description = land_use.get('land_use_description', '')

    content = f'''
    <font size="11" color="#0f1c18"><b>Land Classification:</b> {land_type}</font><br/>
    <font size="9" color="#5a6b66">{description[:200] + "..." if len(description) > 200 else description}</font>
    '''

    para = Paragraph(content, _style_left_wrapped())
    wrapper = Table([[para]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    return wrapper


def _create_action_grid(actions):
    """Create a detailed action items grid."""
    cells = []
    for i, (title, desc) in enumerate(actions[:6]):
        cell = _action_cell(i + 1, title, desc)
        cells.append(cell)

    # Pad to even number
    while len(cells) % 2 != 0:
        cells.append('')

    # Create rows of 2
    rows = []
    for i in range(0, len(cells), 2):
        rows.append([cells[i], cells[i + 1] if i + 1 < len(cells) else ''])

    grid = Table(rows, colWidths=[PAGE_WIDTH / 2] * 2)
    grid.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    return grid


def _action_cell(number, title, desc):
    """Create a single action cell with number badge."""
    badge = Paragraph(
        f'<font size="10" color="#00d4aa"><b>{number}</b></font>',
        _style_center()
    )
    badge_table = Table([[badge]], colWidths=[0.26 * inch], rowHeights=[0.26 * inch])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [3, 3, 3, 3]),
    ]))

    title_para = Paragraph(f'<font size="10" color="#0f1c18"><b>{title}</b></font>', _style_left())
    desc_para = Paragraph(f'<font size="8" color="#5a6b66">{truncate_text(desc, 100)}</font>', _style_left_wrapped())

    text_block = Table([[title_para], [desc_para]], colWidths=[2.8 * inch])
    text_block.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))

    cell = Table([[badge_table, text_block]], colWidths=[0.35 * inch, 2.85 * inch])
    cell.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))

    return cell


def _create_safety_notice(severity_stats):
    """Create a safety notice box."""
    high_sev = severity_stats.get('high_severity_ratio', 0) * 100

    if high_sev > 40:
        level = "HIGH"
        color = "#e63946"
        message = "This site has significant high-severity burn areas. Before visiting, be aware of hazards like standing dead trees (widowmakers), unstable slopes, and ash pits. Always go with a buddy and inform someone of your plans."
    elif high_sev > 20:
        level = "MODERATE"
        color = "#f4a261"
        message = "Exercise caution when visiting this site. Some areas may have standing dead trees and loose soil. Wear sturdy boots and bring plenty of water."
    else:
        level = "LOW"
        color = "#2a9d8f"
        message = "This site appears relatively safe for visits, but always be aware of your surroundings. Standard outdoor safety precautions apply."

    content = f'''
    <font size="10" color="{color}"><b>SAFETY LEVEL: {level}</b></font><br/><br/>
    <font size="9" color="#5a6b66">{message}</font>
    '''

    para = Paragraph(content, _style_left_wrapped())
    wrapper = Table([[para]], colWidths=[PAGE_WIDTH])
    wrapper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#fff8f0") if level == "HIGH" else CARD_BG),
        ('BOX', (0, 0), (-1, -1), 1, HexColor(color)),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    return wrapper


def _create_timeline_section():
    """Create recovery timeline expectations."""
    timeline_data = [
        ['Timeframe', 'What to Expect'],
        ['Year 1-2', 'Ground cover begins returning. Pioneer grasses and shrubs establish. Erosion control critical.'],
        ['Year 3-5', 'Shrub layer develops. Tree seedlings visible. Wildlife begins returning. 40-60% ground cover.'],
        ['Year 5-10', 'Young forest structure emerges. Canopy begins closing. Watershed function improving.'],
        ['Year 10-20', 'Maturing forest. Mixed species structure. Carbon sequestration accelerates.'],
    ]

    table = Table(timeline_data, colWidths=[1.0 * inch, PAGE_WIDTH - 1.2 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), GRAY_TEXT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, CARD_BG]),
    ]))

    return table


def _create_footer(timestamp, analysis_id):
    """Create the report footer."""
    footer_text = Paragraph(
        f'<font size="7" color="#8a9a95">'
        f'Generated by EcoRevive on {timestamp}<br/>'
        f'Data sources: Sentinel-2 satellite imagery, U-Net deep learning model, Gemini AI analysis<br/>'
        f'This report is for informational purposes. Verify with ground-truth data before making decisions.'
        f'</font>',
        _style_center()
    )

    line = HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=0, spaceAfter=8)
    return Table([[line], [footer_text]], colWidths=[PAGE_WIDTH])


def _get_detailed_recommendations(layer3_context: dict, severity_stats: dict) -> list:
    """Get detailed recommendations based on context."""
    recommendations = []

    # Add context-aware recommendations
    if layer3_context and layer3_context.get('land_use', {}).get('recommendations'):
        for rec in layer3_context['land_use']['recommendations'][:3]:
            recommendations.append(("Site-Specific", rec))

    # Default recommendations
    default_recs = [
        ("Learn About Native Species", "Research which native plants thrive in your region. Focus on fire-adapted species that can handle the local climate."),
        ("Join Restoration Events", "Connect with local conservation groups, national forests, or state parks that organize volunteer planting days."),
        ("Support Organizations", "Donate to or volunteer with groups like One Tree Planted, Arbor Day Foundation, or local land trusts."),
        ("Monitor & Document", "If you visit the site, document recovery with photos. Citizen science data helps track ecosystem health."),
        ("Reduce Fire Risk", "Create defensible space around homes, support prescribed burn programs, and practice fire-safe behaviors."),
        ("Spread Awareness", "Share information about wildfire recovery and the importance of healthy forests with your community."),
    ]

    for title, desc in default_recs:
        if len(recommendations) < 6:
            recommendations.append((title, desc))

    return recommendations


# Style helper functions
def _style_left():
    return ParagraphStyle(name='Left', alignment=TA_LEFT)


def _style_left_wrapped():
    return ParagraphStyle(name='LeftWrap', alignment=TA_LEFT, leading=14)


def _style_center():
    return ParagraphStyle(name='Center', alignment=TA_CENTER)


def _style_right():
    return ParagraphStyle(name='Right', alignment=TA_RIGHT)
