"""
DRScan Cam — Image Preprocessing Service
Handles image loading, DICOM conversion, resizing, normalization, and quality checks.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Tuple, Optional

from backend.config import IMAGE_SIZE, ALLOWED_EXTENSIONS


def load_image_from_bytes(file_bytes: bytes, filename: str) -> Image.Image:
    """Load image from raw bytes. Handles standard formats and DICOM."""
    ext = Path(filename).suffix.lower()

    if ext in {".dcm", ".dicom"}:
        return _load_dicom(file_bytes)

    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _load_dicom(file_bytes: bytes) -> Image.Image:
    """Convert DICOM file to PIL Image."""
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = ds.pixel_array.astype(float)

        # Normalize to 0-255
        pixel_array = (
            (pixel_array - pixel_array.min())
            / (pixel_array.max() - pixel_array.min() + 1e-8)
            * 255.0
        )
        pixel_array = pixel_array.astype(np.uint8)

        image = Image.fromarray(pixel_array)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except ImportError:
        raise ValueError("pydicom is required for DICOM files. Install with: pip install pydicom")
    except Exception as e:
        raise ValueError(f"Failed to read DICOM file: {e}")


def preprocess_for_model(
    image: Image.Image,
    target_size: int = IMAGE_SIZE,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocess image for model inference.
    Returns (normalized_array, resized_pil_image).
    """
    # Resize preserving aspect ratio, then center crop
    resized = _resize_and_crop(image, target_size)

    # Convert to numpy array
    arr = np.array(resized, dtype=np.float32)

    # Normalize to [0, 1]
    arr = arr / 255.0

    return arr, resized


def _resize_and_crop(image: Image.Image, target_size: int) -> Image.Image:
    """Resize image to target_size x target_size, preserving aspect ratio with center crop."""
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))
    return image


def validate_image(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Validate uploaded image file.
    Returns error message if invalid, None if valid.
    """
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file format '{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}"

    if len(file_bytes) < 1024:
        return "File is too small — possibly corrupted."

    try:
        load_image_from_bytes(file_bytes, filename)
    except Exception as e:
        return f"Cannot read image: {e}"

    return None


def check_image_quality(image: Image.Image) -> dict:
    """
    Basic image quality assessment.
    Returns dict with quality metrics and pass/fail.
    """
    arr = np.array(image.convert("L"), dtype=np.float32)

    # Brightness (mean luminance)
    brightness = float(np.mean(arr))

    # Contrast (standard deviation)
    contrast = float(np.std(arr))

    # Blur detection (Laplacian variance)
    from PIL import ImageFilter
    laplacian = image.convert("L").filter(ImageFilter.FIND_EDGES)
    sharpness = float(np.var(np.array(laplacian)))

    is_good = brightness > 20 and contrast > 15 and sharpness > 50

    return {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "quality": "good" if is_good else "poor",
        "warnings": _get_quality_warnings(brightness, contrast, sharpness),
    }


def _get_quality_warnings(brightness: float, contrast: float, sharpness: float) -> list:
    """Generate quality warning messages."""
    warnings = []
    if brightness < 20:
        warnings.append("Image is very dark — may affect analysis accuracy.")
    elif brightness > 240:
        warnings.append("Image is overexposed — may affect analysis accuracy.")
    if contrast < 15:
        warnings.append("Low contrast detected — results may be less reliable.")
    if sharpness < 50:
        warnings.append("Image appears blurry — consider re-uploading a sharper image.")
    return warnings


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Convert PIL Image to base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
