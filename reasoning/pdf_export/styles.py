"""
ReportLab styles for EcoRevive PDF reports.
Clean, modern design with professional typography.
"""

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import inch

# Color Palette - Modern and professional
ECOREVIVE_GREEN = HexColor("#00d4aa")
ECOREVIVE_DARK = HexColor("#1a2e28")
ECOREVIVE_ACCENT = HexColor("#00a080")
LIGHT_BG = HexColor("#f8fafa")
BORDER_COLOR = HexColor("#e0e5e4")

# Severity Colors - Refined palette
SEVERITY_HIGH = HexColor("#dc3545")
SEVERITY_MODERATE = HexColor("#fd7e14")
SEVERITY_LOW = HexColor("#28a745")

# Text Colors
TEXT_PRIMARY = HexColor("#1a2e28")
TEXT_SECONDARY = HexColor("#4a5a56")
TEXT_MUTED = HexColor("#7a8a86")
TEXT_LIGHT = HexColor("#9aa9a5")

# Style names
HEADER_STYLE = "EcoReviveHeader"
BODY_STYLE = "EcoReviveBody"
STAT_STYLE = "EcoReviveStat"
TITLE_STYLE = "EcoReviveTitle"
SUBTITLE_STYLE = "EcoReviveSubtitle"
SECTION_HEADER_STYLE = "EcoReviveSectionHeader"
CAPTION_STYLE = "EcoReviveCaption"
TABLE_HEADER_STYLE = "EcoReviveTableHeader"
TABLE_CELL_STYLE = "EcoReviveTableCell"
METRIC_LABEL_STYLE = "EcoReviveMetricLabel"
METRIC_VALUE_STYLE = "EcoReviveMetricValue"


def get_styles():
    """
    Get all custom styles for EcoRevive PDF reports.

    Returns:
        dict: Style sheet with all custom styles
    """
    styles = getSampleStyleSheet()

    # Title style - clean and prominent
    styles.add(ParagraphStyle(
        name=TITLE_STYLE,
        parent=styles['Heading1'],
        fontSize=32,
        leading=38,
        textColor=TEXT_PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=8,
        fontName='Helvetica-Bold',
    ))

    # Subtitle style
    styles.add(ParagraphStyle(
        name=SUBTITLE_STYLE,
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=TEXT_MUTED,
        alignment=TA_LEFT,
        spaceAfter=24,
        fontName='Helvetica',
    ))

    # Section header style - with accent underline effect
    styles.add(ParagraphStyle(
        name=SECTION_HEADER_STYLE,
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=TEXT_PRIMARY,
        spaceBefore=24,
        spaceAfter=12,
        fontName='Helvetica-Bold',
        borderPadding=0,
    ))

    # Header style
    styles.add(ParagraphStyle(
        name=HEADER_STYLE,
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=TEXT_PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=12,
        fontName='Helvetica-Bold',
    ))

    # Body text style
    styles.add(ParagraphStyle(
        name=BODY_STYLE,
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        textColor=TEXT_SECONDARY,
        alignment=TA_LEFT,
        spaceAfter=8,
        fontName='Helvetica',
    ))

    # Stat style - for large numbers
    styles.add(ParagraphStyle(
        name=STAT_STYLE,
        parent=styles['Normal'],
        fontSize=42,
        leading=48,
        textColor=TEXT_PRIMARY,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))

    # Caption style
    styles.add(ParagraphStyle(
        name=CAPTION_STYLE,
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Helvetica',
    ))

    # Table header style
    styles.add(ParagraphStyle(
        name=TABLE_HEADER_STYLE,
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=TEXT_SECONDARY,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
    ))

    # Table cell style
    styles.add(ParagraphStyle(
        name=TABLE_CELL_STYLE,
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=TEXT_PRIMARY,
        alignment=TA_LEFT,
        fontName='Helvetica',
    ))

    # Metric label style
    styles.add(ParagraphStyle(
        name=METRIC_LABEL_STYLE,
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        fontName='Helvetica',
        textTransform='uppercase',
    ))

    # Metric value style
    styles.add(ParagraphStyle(
        name=METRIC_VALUE_STYLE,
        parent=styles['Normal'],
        fontSize=28,
        leading=32,
        textColor=TEXT_PRIMARY,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    ))

    return styles


def get_severity_color(severity_ratio):
    """Get color based on severity percentage."""
    if severity_ratio >= 0.75:
        return SEVERITY_HIGH
    elif severity_ratio >= 0.25:
        return SEVERITY_MODERATE
    else:
        return SEVERITY_LOW


def create_accent_line():
    """Create a colored accent line for section headers."""
    from reportlab.platypus import HRFlowable
    return HRFlowable(
        width="20%",
        thickness=3,
        color=ECOREVIVE_GREEN,
        spaceBefore=0,
        spaceAfter=16,
        hAlign='LEFT'
    )
