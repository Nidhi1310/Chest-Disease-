"""Secure validation for uploaded chest X-ray images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

MIN_DIMENSION = 50
MAX_DIMENSION = 5000
ALLOWED_MODES = {"L", "RGB", "RGBA"}


def validate_image_content(filepath: str | Path) -> bool:
    """Verify that a file is a supported image with reasonable dimensions."""
    path = Path(filepath)
    if not path.exists():
        raise ValueError("Image file does not exist")

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode

        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            raise ValueError("Image too small")
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise ValueError("Image too large")
        if mode not in ALLOWED_MODES:
            raise ValueError("Unsupported image mode")

        return True
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image content") from exc
