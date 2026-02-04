"""
Main PDF Generator for EcoRevive
==================================
Entry point for generating PDF reports.
"""

import io
import base64
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageTemplate, Frame
from reportlab.lib.units import inch

from .personal_report import build_personal_report
from .professional_report import build_professional_report
from .styles import ECOREVIVE_GREEN, TEXT_MUTED


def generate_pdf(
    report_type: str,
    satellite_image: str,
    severity_image: str,
    severity_stats: Dict[str, Any],
    bbox: Dict[str, float],
    layer2_output: Optional[Dict[str, Any]] = None,
    layer3_context: Optional[Dict[str, Any]] = None,
    carbon_analysis: Optional[Dict[str, Any]] = None,
    location_name: Optional[str] = None,
    analysis_id: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Generate a PDF report for burn severity analysis.

    Args:
        report_type: Either "personal" or "professional"
        satellite_image: Base64 encoded satellite image
        severity_image: Base64 encoded severity map
        severity_stats: Dict with severity statistics
        bbox: Bounding box dict with west, south, east, north
        layer2_output: Layer 2 analysis results (optional)
        layer3_context: Layer 3 context analysis (optional)
        carbon_analysis: Carbon analysis results (optional)
        location_name: Human-readable location name (optional)
        analysis_id: Unique analysis identifier (optional)

    Returns:
        Tuple of (pdf_bytes, metadata_dict)

    Example:
        >>> pdf_bytes, metadata = generate_pdf(
        ...     report_type="personal",
        ...     satellite_image="data:image/png;base64,...",
        ...     severity_image="data:image/png;base64,...",
        ...     severity_stats={"high_severity_ratio": 0.3, ...},
        ...     bbox={"west": -121.5, "south": 39.5, "east": -121.0, "north": 40.0},
        ... )
        >>> with open("report.pdf", "wb") as f:
        ...     f.write(pdf_bytes)
    """
    # Generate analysis ID if not provided
    if not analysis_id:
        analysis_id = f"ECO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Create PDF buffer
    buffer = io.BytesIO()

    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"EcoRevive Analysis - {location_name or analysis_id}",
        author="EcoRevive",
        subject="Burn Severity Analysis Report",
        creator="EcoRevive PDF Generator"
    )

    # Build story based on report type
    if report_type.lower() == "professional":
        story = build_professional_report(
            satellite_image=satellite_image,
            severity_image=severity_image,
            severity_stats=severity_stats,
            bbox=bbox,
            layer2_output=layer2_output,
            layer3_context=layer3_context,
            carbon_analysis=carbon_analysis,
            location_name=location_name,
            analysis_id=analysis_id,
        )
        filename = f"EcoRevive_Professional_Report_{analysis_id}.pdf"
    else:
        story = build_personal_report(
            satellite_image=satellite_image,
            severity_image=severity_image,
            severity_stats=severity_stats,
            bbox=bbox,
            carbon_analysis=carbon_analysis,
            layer3_context=layer3_context,
            location_name=location_name,
            analysis_id=analysis_id,
        )
        filename = f"EcoRevive_Impact_Card_{analysis_id}.pdf"

    # Build PDF
    doc.build(
        story,
        onFirstPage=_add_page_header_footer,
        onLaterPages=_add_page_header_footer
    )

    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Create metadata
    metadata = {
        "filename": filename,
        "report_type": report_type,
        "analysis_id": analysis_id,
        "location_name": location_name,
        "generated_at": datetime.now().isoformat(),
        "file_size_bytes": len(pdf_bytes),
        "page_count": _estimate_page_count(report_type),
    }

    return pdf_bytes, metadata


def _add_page_header_footer(canvas, doc):
    """Add header and footer to each page."""
    canvas.saveState()

    page_width, page_height = letter

    # Footer
    footer_y = 0.4 * inch

    # EcoRevive branding
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(0.75 * inch, footer_y, "EcoRevive - Wildfire Recovery Intelligence")

    # Page number
    page_num = canvas.getPageNumber()
    canvas.drawRightString(page_width - 0.75 * inch, footer_y, f"Page {page_num}")

    # Generation timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    canvas.drawCentredString(page_width / 2, footer_y, f"Generated: {timestamp}")

    # Top accent line (subtle)
    canvas.setStrokeColor(ECOREVIVE_GREEN)
    canvas.setLineWidth(2)
    canvas.line(0.75 * inch, page_height - 0.5 * inch,
                page_width - 0.75 * inch, page_height - 0.5 * inch)

    canvas.restoreState()


def _estimate_page_count(report_type: str) -> str:
    """Estimate page count for metadata."""
    if report_type.lower() == "professional":
        return "5-10"
    return "1-2"


def generate_pdf_base64(
    report_type: str,
    satellite_image: str,
    severity_image: str,
    severity_stats: Dict[str, Any],
    bbox: Dict[str, float],
    layer2_output: Optional[Dict[str, Any]] = None,
    layer3_context: Optional[Dict[str, Any]] = None,
    carbon_analysis: Optional[Dict[str, Any]] = None,
    location_name: Optional[str] = None,
    analysis_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a PDF report and return as base64 encoded string.

    Same arguments as generate_pdf().

    Returns:
        Tuple of (base64_encoded_pdf, metadata_dict)
    """
    pdf_bytes, metadata = generate_pdf(
        report_type=report_type,
        satellite_image=satellite_image,
        severity_image=severity_image,
        severity_stats=severity_stats,
        bbox=bbox,
        layer2_output=layer2_output,
        layer3_context=layer3_context,
        carbon_analysis=carbon_analysis,
        location_name=location_name,
        analysis_id=analysis_id,
    )

    # Encode to base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    return pdf_base64, metadata
