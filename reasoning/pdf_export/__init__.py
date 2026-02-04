"""
PDF Export Module for EcoRevive
================================
Generate professional PDF reports for burn severity analysis.

Two report types:
- Personal: 1-2 page "Impact Card" (shareable, emotional)
- Professional: 5-10 page grant-ready document (technical, comprehensive)
"""

from .pdf_generator import generate_pdf
from .personal_report import build_personal_report
from .professional_report import build_professional_report
from .utils import base64_to_pil, pil_to_reportlab_image, create_side_by_side_images
from .styles import (
    get_styles,
    ECOREVIVE_GREEN,
    ECOREVIVE_DARK,
    HEADER_STYLE,
    BODY_STYLE,
    STAT_STYLE,
)

__all__ = [
    "generate_pdf",
    "build_personal_report",
    "build_professional_report",
    "base64_to_pil",
    "pil_to_reportlab_image",
    "create_side_by_side_images",
    "get_styles",
    "ECOREVIVE_GREEN",
    "ECOREVIVE_DARK",
    "HEADER_STYLE",
    "BODY_STYLE",
    "STAT_STYLE",
]
