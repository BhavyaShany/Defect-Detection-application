"""Scan preprocessing pipeline.

Stages (each timed individually so latency can be reported honestly):

    sanitise -> normalise pose / align -> outlier filtering ->
    surface-variation analysis -> depth/normal projection -> viewer subset

The pipeline never invents data: if a stage cannot run, it raises ``ScanError`` or adds a
warning to the result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.config import settings
from backend.services import geometry as G
from backend.services.scan_io import ScanData, ScanError, load_scan

SURFACE_VARIATION_REFERENCE = 0.05  # display scale: variation >= this renders as "max"


@dataclass
class PreprocessResult:
    # Geometry sent to the viewer / persisted (already normalised + aligned)
    view_positions: np.ndarray  # (V, 3) float32
    view_scores: np.ndarray  # (V,) float32 in 0..1 (surface variation / reference)
    view_outlier: np.ndarray  # (V,) bool
    view_faces: Optional[np.ndarray]  # (F, 3) int32 or None
    # 2D representations
    depth_map: np.ndarray  # (H, W) uint8, 0 = no data
    normal_map: np.ndarray  # (H, W, 3) uint8
    metrics: Dict = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


_reference_cache: Dict[str, np.ndarray] = {}


def _load_reference() -> Optional[np.ndarray]:
    path = settings.reference_scan_path
    if path is None:
        return None
    key = str(path)
    if key not in _reference_cache:
        if not path.exists():
            raise ScanError(f"REFERENCE_SCAN_PATH does not exist: {path}", "CONFIG_ERROR")
        ref = load_scan(path.read_bytes(), path.name, min_points=settings.MIN_POINTS)
        norm, _ = G.normalize_pose(ref.points, "pca")
        _reference_cache[key] = norm
    return _reference_cache[key]


def run_preprocessing(scan: ScanData) -> PreprocessResult:
    t_all = time.perf_counter()
    timings: Dict[str, float] = {}
    warnings: List[str] = list(scan.notes)
    points, faces = scan.points, scan.faces

    # -- cap very large inputs (deterministic subsample; faces cannot survive this)
    if len(points) > settings.MAX_POINTS_PROCESSING:
        keep = G.select_view_subset(len(points), settings.MAX_POINTS_PROCESSING, seed=1)
        warnings.append(
            f"Scan has {len(points):,} points; randomly subsampled to {len(keep):,} for processing."
        )
        points, faces = points[keep], None

    # -- pose normalisation / alignment
    t = time.perf_counter()
    mode = settings.ALIGNMENT_MODE.lower()
    if mode not in ("none", "pca", "icp"):
        raise ScanError(f"Invalid ALIGNMENT_MODE '{settings.ALIGNMENT_MODE}'.", "CONFIG_ERROR")
    norm_pts, pose = G.normalize_pose(points, mode)
    icp_info: Dict = {}
    if mode == "icp":
        ref = _load_reference()
        if ref is None:
            warnings.append("ALIGNMENT_MODE=icp but REFERENCE_SCAN_PATH is empty; used PCA alignment only.")
        else:
            norm_pts, rmse, fitness, tf = G.icp_align(
                norm_pts, ref, settings.ICP_MAX_CORRESPONDENCE_DISTANCE, settings.ICP_MAX_ITERATIONS
            )
            icp_info = {"icp_rmse": round(rmse, 5), "icp_fitness": round(fitness, 4)}
            if fitness < 0.3:
                warnings.append(
                    f"ICP fitness is low ({fitness:.2f}); the scan may not match the reference part."
                )
    timings["align_ms"] = (time.perf_counter() - t) * 1000

    # -- outlier filtering
    t = time.perf_counter()
    inlier = G.statistical_outlier_mask(norm_pts, settings.OUTLIER_NB_NEIGHBORS, settings.OUTLIER_STD_RATIO)
    timings["outlier_filter_ms"] = (time.perf_counter() - t) * 1000
    n_out = int((~inlier).sum())
    if inlier.sum() < settings.MIN_POINTS:
        raise ScanError("Too few points remain after noise filtering; the scan is unusable.", "TOO_FEW_POINTS")

    # -- geometric analysis (unsupervised; NOT a defect detector)
    t = time.perf_counter()
    variation, spacing = G.surface_variation(norm_pts, settings.SURFACE_VARIATION_K)
    var_summary = G.summarize_variation(variation[inlier], SURFACE_VARIATION_REFERENCE)
    timings["surface_analysis_ms"] = (time.perf_counter() - t) * 1000

    # -- 2D projection for the CNN
    t = time.perf_counter()
    maps = G.project_to_maps(norm_pts[inlier], settings.PROJECTION_RESOLUTION, point_spacing=spacing)
    timings["projection_ms"] = (time.perf_counter() - t) * 1000
    if maps["coverage"] < settings.MIN_PROJECTION_COVERAGE:
        raise ScanError(
            f"The scan covers only {maps['coverage']:.1%} of the projection plane; "
            "it is too sparse or too thin to inspect.",
            "SPARSE_SCAN",
        )
    if maps["coverage"] < 0.15:
        warnings.append(f"Low projection coverage ({maps['coverage']:.1%}); predictions may be unreliable.")

    # -- viewer subset
    t = time.perf_counter()
    n = len(norm_pts)
    view_faces = None
    if faces is not None:
        if n <= settings.MAX_VIEW_POINTS and len(faces) <= settings.MAX_VIEW_FACES:
            idx = np.arange(n)
            view_faces = faces
        else:
            idx = G.select_view_subset(n, settings.MAX_VIEW_POINTS)
            warnings.append("Mesh is too large for the viewer; showing a point subset instead of surfaces.")
    else:
        idx = G.select_view_subset(n, settings.MAX_VIEW_POINTS)
    scores = np.clip(variation / SURFACE_VARIATION_REFERENCE, 0.0, 1.0).astype(np.float32)
    timings["view_prep_ms"] = (time.perf_counter() - t) * 1000

    timings["preprocessing_total_ms"] = (time.perf_counter() - t_all) * 1000
    timings = {k: round(v, 2) for k, v in timings.items()}

    metrics = {
        "input_format": scan.fmt,
        "points_in_file": scan.raw_point_count,
        "points_processed": int(n),
        "outliers_removed": n_out,
        "has_faces": scan.faces is not None,
        "mean_point_spacing": round(float(spacing), 5),
        "original_extents": [round(v, 4) for v in pose["original_extents"]],
        "alignment": pose["alignment"],
        "projection_coverage": round(maps["coverage"], 4),
        "projection_hit_ratio": round(maps["hit_ratio"], 4),
        "surface_variation": {k: round(v, 5) for k, v in var_summary.items()},
        **icp_info,
    }
    return PreprocessResult(
        view_positions=norm_pts[idx].astype(np.float32),
        view_scores=scores[idx],
        view_outlier=(~inlier)[idx],
        view_faces=view_faces,
        depth_map=maps["depth"],
        normal_map=maps["normal"],
        metrics=metrics,
        timings_ms=timings,
        warnings=warnings,
    )
