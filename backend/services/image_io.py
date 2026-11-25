"""Photograph loading and validation for image-mode models (2-D casting photos).

Uses Pillow only. Rejects corrupt files, tiny images and decompression bombs; never
substitutes data for an unreadable image.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.services.scan_io import ScanError

Image.MAX_IMAGE_PIXELS = None  # we enforce our own (stricter, configurable) limit below


@dataclass
class ImageData:
    rgb: np.ndarray  # (H, W, 3) uint8
    width: int
    height: int
    fmt: str
    mode: str  # original colour mode, e.g. "L" for the grayscale Kaggle photos
    notes: List[str]


def load_image(raw: bytes, filename: str, max_pixels: int = 40_000_000, min_side: int = 32) -> ImageData:
    if not raw:
        raise ScanError("The uploaded file is empty.", "EMPTY_FILE")
    try:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        fmt = (img.format or "").lower()
        if w * h > max_pixels:  # checked from the header, before decoding pixels
            raise ScanError(f"Image is {w}x{h} ({w * h / 1e6:.0f} MP); the limit is {max_pixels / 1e6:.0f} MP.",
                            "IMAGE_TOO_LARGE")
        if min(w, h) < min_side:
            raise ScanError(f"Image is only {w}x{h}; at least {min_side} px per side are required.", "IMAGE_TOO_SMALL")
        mode = img.mode
        img = ImageOps.exif_transpose(img)
        img.load()  # force full decode so truncated files fail here
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except ScanError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ScanError(f"Could not read the image: {exc}", "CORRUPT_SCAN") from exc
    notes: List[str] = []
    if float(rgb.std()) < 1.0:
        notes.append("The image is almost uniform (no visible content); the prediction is unlikely to be meaningful.")
    return ImageData(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0], fmt=fmt, mode=mode, notes=notes)


def image_to_data_url(rgb: np.ndarray, max_side: int = 640, quality: int = 85) -> str:
    """Downscaled JPEG preview (data URL) stored with the record so history can show the photo."""
    import base64

    img = Image.fromarray(rgb)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_summary(data: ImageData) -> Dict:
    return {"input_format": data.fmt, "width": data.width, "height": data.height, "color_mode": data.mode}
