"""
Utility functions for PDF generation.
Handles image conversion and layout helpers.
"""

import base64
import io
from PIL import Image
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, Table, TableStyle
from reportlab.lib.colors import HexColor


def base64_to_pil(base64_string: str, max_size: int = 512) -> Image.Image:
    """
    Convert base64 encoded image string to PIL Image, resizing for speed.

    Args:
        base64_string: Base64 encoded image, optionally with data URI prefix
        max_size: Maximum dimension (width or height) - resizes large images for faster PDF

    Returns:
        PIL Image object (resized if larger than max_size)
    """
    # Remove data URI prefix if present
    if base64_string.startswith('data:'):
        # Format: data:image/png;base64,ACTUALBASE64DATA
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64
    image_data = base64.b64decode(base64_string)

    # Create PIL Image
    image = Image.open(io.BytesIO(image_data))

    # Resize large images for faster PDF generation
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def pil_to_reportlab_image(pil_image: Image.Image, width: float = None, height: float = None) -> RLImage:
    """
    Convert PIL Image to ReportLab Image object.

    Args:
        pil_image: PIL Image object
        width: Target width in points (optional)
        height: Target height in points (optional)

    Returns:
        ReportLab Image object
    """
    # Save PIL image to bytes buffer - use JPEG for speed (smaller files)
    buffer = io.BytesIO()

    # Convert to RGB if necessary (for PNG with alpha, or JPEG requirement)
    if pil_image.mode in ('RGBA', 'LA', 'P', 'L'):
        # Create white background
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        if pil_image.mode in ('RGBA', 'LA'):
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        elif pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')

    # Use JPEG for faster encoding/smaller size (quality 85 is good balance)
    pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)

    # Create ReportLab Image
    if width and height:
        img = RLImage(buffer, width=width, height=height)
    elif width:
        # Calculate height maintaining aspect ratio
        aspect = pil_image.height / pil_image.width
        img = RLImage(buffer, width=width, height=width * aspect)
    elif height:
        # Calculate width maintaining aspect ratio
        aspect = pil_image.width / pil_image.height
        img = RLImage(buffer, width=height * aspect, height=height)
    else:
        # Default size
        img = RLImage(buffer)

    return img


def create_side_by_side_images(
    satellite_base64: str,
    severity_base64: str,
    total_width: float = 6.5 * inch,
    spacing: float = 0.2 * inch
) -> Table:
    """
    Create a side-by-side layout for satellite and severity images.

    Args:
        satellite_base64: Base64 encoded satellite image
        severity_base64: Base64 encoded severity image
        total_width: Total width for both images
        spacing: Space between images

    Returns:
        ReportLab Table containing both images
    """
    # Calculate individual image width
    image_width = (total_width - spacing) / 2

    # Convert images
    satellite_pil = base64_to_pil(satellite_base64)
    severity_pil = base64_to_pil(severity_base64)

    satellite_rl = pil_to_reportlab_image(satellite_pil, width=image_width)
    severity_rl = pil_to_reportlab_image(severity_pil, width=image_width)

    # Create table with images
    data = [[satellite_rl, severity_rl]]

    table = Table(data, colWidths=[image_width, image_width])
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), spacing / 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), spacing / 2),
    ]))

    return table


def create_stat_box(
    value: str,
    label: str,
    color: str = "#00d4aa",
    width: float = 1.5 * inch,
    height: float = 1.0 * inch
) -> Table:
    """
    Create a styled statistics box.

    Args:
        value: The main value to display (e.g., "45%")
        label: Label below the value (e.g., "High Severity")
        color: Background color hex string
        width: Box width
        height: Box height

    Returns:
        ReportLab Table styled as a stat box
    """
    from reportlab.platypus import Paragraph
    from .styles import get_styles, STAT_STYLE

    styles = get_styles()

    # Create paragraphs
    value_para = Paragraph(f'<font size="24" color="white"><b>{value}</b></font>', styles['Normal'])
    label_para = Paragraph(f'<font size="9" color="white">{label}</font>', styles['Normal'])

    data = [[value_para], [label_para]]

    table = Table(data, colWidths=[width])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(color)),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))

    return table


def format_coordinates(bbox: dict) -> str:
    """
    Format bounding box coordinates for display.

    Args:
        bbox: Dict with west, south, east, north keys

    Returns:
        Formatted string like "39.8°N, 121.4°W"
    """
    center_lat = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
    center_lon = (bbox.get('west', 0) + bbox.get('east', 0)) / 2

    lat_dir = 'N' if center_lat >= 0 else 'S'
    lon_dir = 'W' if center_lon < 0 else 'E'

    return f"{abs(center_lat):.4f}°{lat_dir}, {abs(center_lon):.4f}°{lon_dir}"


def calculate_area_km2(bbox: dict) -> float:
    """
    Calculate area in square kilometers from bounding box.

    Args:
        bbox: Dict with west, south, east, north keys

    Returns:
        Area in km²
    """
    import math

    lat_mid = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
    km_per_deg_lon = 111.32 * math.cos(lat_mid * math.pi / 180)
    km_per_deg_lat = 110.574

    width_km = abs(bbox.get('east', 0) - bbox.get('west', 0)) * km_per_deg_lon
    height_km = abs(bbox.get('north', 0) - bbox.get('south', 0)) * km_per_deg_lat

    return width_km * height_km


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text with ellipsis if too long.

    Args:
        text: Input text
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
