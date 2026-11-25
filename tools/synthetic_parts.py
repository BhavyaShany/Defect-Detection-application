"""Synthetic cast-part point clouds for TESTING THE PIPELINE ONLY.

These are procedurally generated scans of simple parts with optional geometric
"defects" (pits, grooves, bumps). They exist so the loader, geometry analysis and
viewer can be exercised without a physical scanner. They are NOT training data and are
never inserted into the inspection database.
"""
from __future__ import annotations

import numpy as np

PARTS = ["engine_bracket", "brake_rotor", "turbine_impeller", "valve_body"]
DEFECTS = ["none", "porosity", "shrinkage", "crack", "inclusion", "flash"]


def _rect(rng, n, w, h, z=0.0):
    return np.stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-h / 2, h / 2, n), np.full(n, z)], 1)


def _annulus(rng, n, r0, r1, z=0.0):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], 1)


def _cyl_side(rng, n, r, z0, z1, cx=0.0, cy=0.0):
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), rng.uniform(z0, z1, n)], 1)


def _base_part(rng, part, n):
    if part == "brake_rotor":
        return np.vstack([
            _annulus(rng, int(n * 0.55), 0.22, 0.5, 0.0),
            _annulus(rng, int(n * 0.25), 0.0, 0.2, 0.06),
            _cyl_side(rng, int(n * 0.12), 0.21, 0.0, 0.06),
            _cyl_side(rng, int(n * 0.08), 0.5, -0.02, 0.0),
        ])
    if part == "turbine_impeller":
        pts = [_annulus(rng, int(n * 0.35), 0.0, 0.5, 0.0), _annulus(rng, int(n * 0.12), 0.0, 0.1, 0.1)]
        for b in range(6):
            ang = b * np.pi / 3
            blade = _rect(rng, int(n * 0.08), 0.38, 0.045, 0.08)
            blade[:, 0] += 0.31
            c, s = np.cos(ang), np.sin(ang)
            pts.append(np.stack([blade[:, 0] * c - blade[:, 1] * s, blade[:, 0] * s + blade[:, 1] * c, blade[:, 2]], 1))
        return np.vstack(pts)
    if part == "valve_body":
        top = _rect(rng, int(n * 0.6), 0.8, 0.6, 0.0)
        top = top[np.hypot(top[:, 0], top[:, 1]) > 0.13]
        floor = _annulus(rng, int(n * 0.12), 0.0, 0.12, -0.1)
        wall = _cyl_side(rng, int(n * 0.1), 0.12, -0.1, 0.0)
        return np.vstack([top, floor, wall])
    # engine_bracket (default)
    plate = _rect(rng, int(n * 0.6), 1.0, 0.7, 0.0)
    plate = plate[np.hypot(plate[:, 0] + 0.28, plate[:, 1]) > 0.17]
    return np.vstack([plate, _annulus(rng, int(n * 0.12), 0.0, 0.15, 0.08) + [-0.28, 0, 0],
                      _cyl_side(rng, int(n * 0.1), 0.15, 0.0, 0.08, -0.28, 0)])


def apply_defect(pts, defect, rng):
    """Displace flat-surface points to emulate a defect. Returns (points, ground_truth).

    Ground truth is for tests only: ``{"center": (x, y), "radius": r}`` of the geometry change.
    """
    if defect == "none":
        return pts, None
    cx, cy = 0.22, 0.12  # on the flat region of every part
    top = (np.abs(pts[:, 2]) < 0.005)
    p = pts.copy()
    d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    if defect == "porosity":
        offs = [(0, 0), (0.04, 0.01), (-0.03, 0.03), (0.01, -0.04)]
        for ox, oy in offs:
            dd = np.hypot(p[:, 0] - cx - ox, p[:, 1] - cy - oy)
            p[:, 2] -= top * 0.02 * np.exp(-(dd ** 2) / (2 * 0.009 ** 2))
        return p, {"center": (cx, cy), "radius": 0.07}
    if defect == "shrinkage":
        p[:, 2] -= top * 0.03 * np.exp(-(d ** 2) / (2 * 0.05 ** 2))
        return p, {"center": (cx, cy), "radius": 0.12}
    if defect == "inclusion":
        p[:, 2] += top * 0.03 * np.exp(-(d ** 2) / (2 * 0.012 ** 2))
        return p, {"center": (cx, cy), "radius": 0.04}
    if defect == "crack":  # narrow groove along a segment
        a, b = np.array([cx - 0.12, cy - 0.05]), np.array([cx + 0.12, cy + 0.05])
        ab = b - a
        t = np.clip(((p[:, :2] - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(p[:, :2] - (a + t[:, None] * ab), axis=1)
        p[:, 2] -= top * 0.025 * np.exp(-(dd ** 2) / (2 * 0.005 ** 2))
        return p, {"center": (cx, cy), "radius": 0.14}
    if defect == "flash":  # thin raised ridge
        dd = np.abs(p[:, 0] - cx)
        p[:, 2] += top * (np.abs(p[:, 1] - cy) < 0.15) * 0.02 * np.exp(-(dd ** 2) / (2 * 0.004 ** 2))
        return p, {"center": (cx, cy), "radius": 0.16}
    raise ValueError(f"unknown defect {defect!r}")


def generate_part(part="engine_bracket", defect="none", n=40000, seed=0, noise=0.0004):
    """Returns ``(points float32 (N,3), ground_truth_or_None)``."""
    rng = np.random.default_rng(seed)
    pts = _base_part(rng, part, n)
    pts, gt = apply_defect(pts, defect, rng)
    pts = pts + rng.normal(0, noise, pts.shape)
    return pts.astype(np.float32), gt


def write_ply_binary(path, pts):
    pts = np.asarray(pts, dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())
