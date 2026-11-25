"""Persistence of the processed 3D geometry used by the viewer.

Each inspected scan stores a compressed ``.npz`` (positions, surface-variation scores,
outlier flags, optional faces) under ``data/geometry/<scan_id>.npz``. File names come only
from server-generated UUIDs, never from user input.
"""
import base64
import os
import re
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from backend.config import settings

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _path(scan_id: str):
    if not _UUID_RE.match(scan_id or ""):
        raise ValueError("invalid scan id")
    return settings.geometry_dir / f"{scan_id}.npz"


def save_geometry(scan_id: str, positions: np.ndarray, scores: np.ndarray, outlier: np.ndarray,
                  faces: Optional[np.ndarray]) -> int:
    """Atomically write the geometry file; returns its size in bytes."""
    target = _path(scan_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(fd)
    try:
        arrays = {
            "positions": positions.astype(np.float32),
            "scores": scores.astype(np.float32),
            "outlier": outlier.astype(np.uint8),
        }
        if faces is not None:
            arrays["faces"] = faces.astype(np.uint32)
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target.stat().st_size


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def load_geometry_payload(scan_id: str) -> Optional[Dict[str, Any]]:
    """JSON-ready payload (little-endian typed arrays as base64) or ``None`` if not stored."""
    try:
        path = _path(scan_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    with np.load(path) as z:
        pos = z["positions"].astype("<f4")
        scores = z["scores"].astype("<f4")
        outlier = z["outlier"].astype("u1")
        faces = z["faces"].astype("<u4") if "faces" in z.files else None
    return {
        "point_count": int(len(pos)),
        "face_count": 0 if faces is None else int(len(faces)),
        "bounds": {"min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
        "positions_b64": _b64(pos),
        "scores_b64": _b64(scores),
        "outlier_b64": _b64(outlier),
        "faces_b64": None if faces is None else _b64(faces),
        # Reserved for models that output spatial defect masks (per-point uint8). Always null
        # today: the current classifier has no localisation output, so nothing is fabricated.
        "defect_mask_b64": None,
        "score_reference": "surface_variation",
    }


def delete_geometry(scan_id: str) -> None:
    try:
        path = _path(scan_id)
    except ValueError:
        return
    if path.exists():
        path.unlink()
