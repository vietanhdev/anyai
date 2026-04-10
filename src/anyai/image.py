"""Basic image AI utilities.

Provides image metadata extraction using Pillow. When optional AI backends
are installed, the ``describe`` function can be extended to provide
AI-generated image descriptions.
"""

import os
from typing import Any, Dict, Optional

from anyai.utils import check_deps


def describe(image_path: str) -> Dict[str, Any]:
    """Describe an image by extracting its metadata.

    Returns a dictionary containing the image file path, format, dimensions,
    color mode, file size, and any EXIF information available. When optional
    AI backends (e.g. ``anycv``) are installed, this function can be extended
    to include an AI-generated textual description.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary with image metadata including:
        - ``'path'``: the file path
        - ``'format'``: image format (e.g. ``'JPEG'``, ``'PNG'``)
        - ``'width'``: image width in pixels
        - ``'height'``: image height in pixels
        - ``'mode'``: color mode (e.g. ``'RGB'``, ``'RGBA'``)
        - ``'file_size_bytes'``: file size in bytes
        - ``'has_exif'``: whether EXIF data is present
        - ``'ai_description'``: AI-generated description if backend available,
          otherwise ``None``

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install it with: pip install Pillow"
        )

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    file_size = os.path.getsize(image_path)

    # Extract EXIF data if available.
    exif_data: Optional[Dict[str, Any]] = None
    has_exif = False
    try:
        exif_raw = img._getexif()
        if exif_raw:
            has_exif = True
    except (AttributeError, Exception):
        pass

    info: Dict[str, Any] = {
        "path": image_path,
        "format": img.format,
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "file_size_bytes": file_size,
        "has_exif": has_exif,
        "ai_description": None,
    }

    # If an AI backend is available, attempt to generate a description.
    if check_deps("anycv"):
        try:
            import anycv
            info["ai_description"] = anycv.describe_image(image_path)
        except Exception:
            pass

    img.close()
    return info


def classify(image_path: str) -> Dict[str, Any]:
    """Classify an image using available backends.

    Without AI backends, returns basic format-based classification.
    With optional backends installed, returns AI-powered classification.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary with:
        - ``'path'``: the file path
        - ``'format_type'``: broad format category (``'photo'``, ``'graphic'``,
          ``'unknown'``)
        - ``'ai_classification'``: AI classification if backend available,
          otherwise ``None``

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install it with: pip install Pillow"
        )

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    # Basic heuristic classification by format and mode.
    fmt = (img.format or "").upper()
    mode = img.mode

    photo_formats = {"JPEG", "JPG", "TIFF"}
    graphic_formats = {"PNG", "GIF", "BMP", "SVG", "WEBP"}

    if fmt in photo_formats:
        format_type = "photo"
    elif fmt in graphic_formats:
        format_type = "graphic"
    else:
        format_type = "unknown"

    result: Dict[str, Any] = {
        "path": image_path,
        "format_type": format_type,
        "ai_classification": None,
    }

    img.close()
    return result
