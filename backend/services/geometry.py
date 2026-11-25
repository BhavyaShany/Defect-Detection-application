"""Point-cloud geometry routines (NumPy / SciPy, with optional Open3D acceleration).

Everything in this module is deterministic and framework-free so it can be unit tested
without FastAPI, TensorFlow or Open3D installed.

Honesty notes
-------------
* ``surface_variation`` is an *unsupervised geometric descriptor* (local curvature /
  roughness). It is **not** a defect detector and the API labels it as such.
* Alignment defaults to PCA canonical orientation. ICP is only run when a real
  reference scan is configured; the original code aligned every scan to a made-up
  cylinder, which distorted the data.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

try:  # optional accelerator
    import open3d as o3d  # type: ignore

    HAS_OPEN3D = True
except Exception:  # ImportError or binary incompatibility
    o3d = None
    HAS_OPEN3D = False


# ----------------------------------------------------------------------- pose
def normalize_pose(points: np.ndarray, mode: str = "pca") -> Tuple[np.ndarray, Dict]:
    """Centre the cloud, optionally rotate to principal axes, scale so the part is ~1 unit.

    Returns ``(points_normalised, info)``. ``info`` holds the original extents so physical
    size is not lost by normalisation.
    """
    pts = points.astype(np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    original_extents = np.ptp(pts, axis=0)
    rotation = np.eye(3)

    if mode in ("pca", "icp"):
        cov = np.cov(centred.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]  # largest variance -> x, smallest -> z (depth axis)
        rotation = evecs[:, order]
        proj = centred @ rotation
        # Resolve the sign ambiguity deterministically via skewness (third moment)
        for ax in range(3):
            if np.mean(proj[:, ax] ** 3) < 0:
                rotation[:, ax] *= -1
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        centred = centred @ rotation

    scale = float(np.ptp(centred, axis=0).max())
    scale = scale if scale > 1e-12 else 1.0
    normalised = (centred / scale).astype(np.float32)
    return normalised, {
        "alignment": mode,
        "scale": scale,
        "original_extents": [float(v) for v in original_extents],
        "rotation": rotation.tolist(),
    }


# -------------------------------------------------------------------- outliers
def statistical_outlier_mask(points: np.ndarray, nb_neighbors: int = 20,
                             std_ratio: float = 2.0) -> np.ndarray:
    """Boolean inlier mask using the mean k-NN distance criterion (as Open3D does)."""
    n = len(points)
    if n <= nb_neighbors + 1:
        return np.ones(n, dtype=bool)

    if HAS_OPEN3D:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            mask = np.zeros(n, dtype=bool)
            mask[np.asarray(ind, dtype=np.int64)] = True
            return mask
        except Exception:
            pass  # fall through to SciPy

    tree = cKDTree(points)
    mean_d = np.empty(n, dtype=np.float64)
    chunk = 200_000
    for s in range(0, n, chunk):
        d, _ = tree.query(points[s:s + chunk], k=nb_neighbors + 1, workers=-1)
        mean_d[s:s + chunk] = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d <= thr


# ------------------------------------------------------------------------- ICP
def icp_align(source: np.ndarray, target: np.ndarray, max_dist: float,
              max_iter: int = 50) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Rigid point-to-point ICP of ``source`` onto ``target``.

    Returns ``(aligned, rmse, fitness, 4x4 transform)``; fitness is the inlier fraction.
    """
    if HAS_OPEN3D:
        try:
            s = o3d.geometry.PointCloud()
            s.points = o3d.utility.Vector3dVector(source.astype(np.float64))
            t = o3d.geometry.PointCloud()
            t.points = o3d.utility.Vector3dVector(target.astype(np.float64))
            reg = o3d.pipelines.registration.registration_icp(
                s, t, max_dist, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
            )
            s.transform(reg.transformation)
            return (np.asarray(s.points, dtype=np.float32), float(reg.inlier_rmse),
                    float(reg.fitness), np.asarray(reg.transformation))
        except Exception:
            pass

    cur = source.astype(np.float64).copy()
    tree = cKDTree(target)
    total = np.eye(4)
    prev = np.inf
    rmse = 0.0
    dists = np.zeros(len(cur))
    for _ in range(max_iter):
        dists, idx = tree.query(cur, workers=-1)
        valid = dists < max_dist
        if valid.sum() < 6:
            break
        src, tgt = cur[valid], target[idx[valid]]
        rmse = float(np.sqrt(np.mean(dists[valid] ** 2)))
        if abs(prev - rmse) < 1e-6:
            break
        prev = rmse
        cs, ct = src.mean(0), tgt.mean(0)
        u, _, vt = np.linalg.svd((src - cs).T @ (tgt - ct))
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[2] *= -1
            r = vt.T @ u.T
        t = ct - r @ cs
        cur = cur @ r.T + t
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = r, t
        total = step @ total
    dists, _ = tree.query(cur, workers=-1)
    fitness = float(np.mean(dists < max_dist))
    return cur.astype(np.float32), rmse, fitness, total


# ------------------------------------------------------------ surface variation
def surface_variation(points: np.ndarray, k: int = 16) -> Tuple[np.ndarray, float]:
    """Per-point *surface variation* λ0 / (λ0 + λ1 + λ2) of the k-NN covariance.

    0 means locally planar; larger values mean strong local curvature / roughness / edges
    (maximum 1/3). Returns ``(variation, mean_nearest_neighbour_spacing)``.
    """
    n = len(points)
    k = int(max(4, min(k, n - 1)))
    tree = cKDTree(points)
    var = np.empty(n, dtype=np.float32)
    nn_dist_sum = 0.0
    chunk = 50_000
    for s in range(0, n, chunk):
        dists, idx = tree.query(points[s:s + chunk], k=k, workers=-1)
        nn_dist_sum += float(dists[:, 1].sum())
        nb = points[idx].astype(np.float64)  # (c, k, 3)
        d = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("cki,ckj->cij", d, d) / k
        ev = np.linalg.eigvalsh(cov)  # ascending
        var[s:s + chunk] = (ev[:, 0] / (ev.sum(axis=1) + 1e-18)).astype(np.float32)
    return var, nn_dist_sum / n


def summarize_variation(var: np.ndarray, ref: float) -> Dict[str, float]:
    return {
        "p50": float(np.percentile(var, 50)),
        "p95": float(np.percentile(var, 95)),
        "p99": float(np.percentile(var, 99)),
        "max": float(var.max()),
        "reference": float(ref),
        "fraction_above_reference": float(np.mean(var > ref)),
    }


# ------------------------------------------------------------------ projection
def project_to_maps(points: np.ndarray, size: int = 224, point_spacing: Optional[float] = None) -> Dict:
    """Orthographic top-down depth and surface-normal maps (aligned z = depth axis).

    * depth is scaled by a *fixed physical reference* (not min-max stretched), so a flat
      nominal part with sensor noise does not turn into a high-contrast image;
    * 0 in the depth image means "no data"; valid pixels are 1..255;
    * holes inside the part silhouette are inpainted, the background is left empty;
    * ``point_spacing`` (mean nearest-neighbour distance, same units as ``points``) sizes
      the hole-closing kernel so sparse scans/meshes are not misread as empty;
    * ``coverage`` is the fraction of the image occupied by the (closed) silhouette and
      ``hit_ratio`` the fraction of pixels that received at least one raw point.
    """
    x, y, z = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), points[:, 2].astype(np.float64)
    span = float(max(np.ptp(x), np.ptp(y), 1e-9))
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    u = np.clip(np.round(((x - cx) / span * 0.92 + 0.5) * (size - 1)).astype(np.int64), 0, size - 1)
    v = np.clip(np.round((0.5 - (y - cy) / span * 0.92) * (size - 1)).astype(np.int64), 0, size - 1)

    z_ref = max(float(np.ptp(z)), 0.2 * span)
    zmin = float(z.min())
    d = np.clip((z - zmin) / z_ref, 0.0, 1.0)

    buf = np.full(size * size, -1.0, dtype=np.float64)
    np.maximum.at(buf, v * size + u, d)  # top-most surface wins (z-buffer)
    buf = buf.reshape(size, size)
    valid = buf >= 0
    hit_ratio = float(valid.mean())

    pix_world = span / (0.92 * (size - 1))  # world units per pixel
    ksz = 7
    if point_spacing:
        ksz = int(np.clip(np.ceil(2.5 * point_spacing / pix_world), 3, 31))
        ksz += 1 - ksz % 2  # odd
    depth8 = np.where(valid, 1 + np.round(buf * 254), 0).astype(np.uint8)
    valid_u8 = valid.astype(np.uint8)
    silhouette = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    coverage = float((silhouette > 0).mean())
    holes = ((silhouette > 0) & (~valid)).astype(np.uint8) * 255
    if holes.any():
        depth8 = cv2.inpaint(depth8, holes, 3, cv2.INPAINT_TELEA)

    # Surface normals from the (hole-filled) depth image
    depth_f = np.where(depth8 > 0, (depth8.astype(np.float32) - 1) / 254.0, 0.0)
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    sx, sy = gx * z_ref / pix_world, gy * z_ref / pix_world
    nx, ny, nz = -sx, sy, np.ones_like(sx)  # image y axis points down
    mag = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    normal8 = np.stack([(nx / mag * 0.5 + 0.5), (ny / mag * 0.5 + 0.5), (nz / mag * 0.5 + 0.5)], axis=-1)
    normal8 = (normal8 * 255).astype(np.uint8)
    normal8[depth8 == 0] = 0
    return {"depth": depth8, "normal": normal8, "coverage": coverage, "hit_ratio": hit_ratio, "z_reference": z_ref}


def build_model_input(depth: np.ndarray, normal: np.ndarray, mode: str = "depth") -> np.ndarray:
    """(H, W, 3) float32 in 0..255, ready for an EfficientNet-style model.

    The mode must match what the model was trained on (stored in ``labels.json``).
    """
    if mode == "depth":
        arr = np.stack([depth] * 3, axis=-1)
    elif mode == "normal":
        arr = normal
    elif mode == "depth_normal":
        arr = np.stack([depth, normal[..., 0], normal[..., 1]], axis=-1)
    else:
        raise ValueError(f"Unknown input mode {mode!r} (expected depth | normal | depth_normal)")
    return arr.astype(np.float32)


def encode_png_data_url(img: np.ndarray) -> str:
    import base64

    arr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------- viewer
def select_view_subset(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Sorted indices of the points sent to the viewer (all of them if under the cap)."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))
