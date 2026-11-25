"""3D scan loading and validation.

Pure NumPy parsers for PLY (ascii / binary), PCD (ascii / binary), STL (ascii / binary)
and OBJ. They do not depend on Open3D, so uploads behave identically on every machine
and never need temporary files on disk.

Design rule: a file that cannot be parsed is *rejected* with a ``ScanError``. The original
implementation silently replaced unparseable input with random points, which produced
fabricated inspection results.
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class ScanError(Exception):
    """A scan could not be accepted. ``code`` is a stable machine-readable identifier."""

    def __init__(self, message: str, code: str = "INVALID_SCAN"):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class ScanData:
    points: np.ndarray  # (N, 3) float32, finite
    faces: Optional[np.ndarray]  # (M, 3) int32 or None for pure point clouds
    fmt: str  # "ply" | "pcd" | "stl" | "obj"
    raw_point_count: int  # points present in the file before cleaning
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- PLY
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
    "uint": "u4", "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1",
    "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4",
    "float64": "f8",
}


def _triangulate_fan(poly: List[int]) -> List[List[int]]:
    return [[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)]


def _parse_ply(raw: bytes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not raw.startswith(b"ply"):
        raise ScanError("Not a valid PLY file (missing 'ply' header).", "CORRUPT_SCAN")
    idx = raw.find(b"end_header")
    if idx < 0:
        raise ScanError("Not a valid PLY file (missing 'end_header').", "CORRUPT_SCAN")
    body_start = idx + len(b"end_header")
    if raw[body_start:body_start + 2] == b"\r\n":
        body_start += 2
    elif raw[body_start:body_start + 1] in (b"\n", b"\r"):
        body_start += 1

    header = raw[:idx].decode("ascii", errors="replace").splitlines()
    fmt = None
    elements: List[dict] = []
    for line in header[1:]:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1] if len(tok) > 1 else None
        elif tok[0] == "element":
            elements.append({"name": tok[1], "count": int(tok[2]), "props": []})
        elif tok[0] == "property":
            if not elements:
                raise ScanError("Malformed PLY header (property before element).", "CORRUPT_SCAN")
            if tok[1] == "list":
                elements[-1]["props"].append((tok[4], "list", tok[2], tok[3]))
            else:
                elements[-1]["props"].append((tok[2], tok[1]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ScanError(f"Unsupported or missing PLY format: {fmt!r}.", "CORRUPT_SCAN")
    if not any(e["name"] == "vertex" for e in elements):
        raise ScanError("PLY file has no 'vertex' element.", "CORRUPT_SCAN")

    vertices: Optional[np.ndarray] = None
    faces_list: List[np.ndarray] = []

    if fmt == "ascii":
        lines = [ln for ln in raw[body_start:].decode("ascii", errors="replace").splitlines() if ln.strip()]
        ptr = 0
        for el in elements:
            n = el["count"]
            chunk = lines[ptr:ptr + n]
            if len(chunk) < n:
                raise ScanError("PLY file is truncated (fewer data lines than declared).", "CORRUPT_SCAN")
            ptr += n
            has_list = any(p[1] == "list" for p in el["props"])
            if not has_list:
                arr = np.fromstring("\n".join(chunk), sep=" ", dtype=np.float64)
                if arr.size != n * len(el["props"]):
                    raise ScanError("PLY element data does not match its declared properties.", "CORRUPT_SCAN")
                arr = arr.reshape(n, len(el["props"]))
                if el["name"] == "vertex":
                    names = [p[0] for p in el["props"]]
                    if not all(c in names for c in ("x", "y", "z")):
                        raise ScanError("PLY vertex element lacks x/y/z properties.", "CORRUPT_SCAN")
                    vertices = arr[:, [names.index("x"), names.index("y"), names.index("z")]]
            elif el["name"] == "face":
                if el["props"][0][1] != "list":
                    raise ScanError("Unsupported PLY face layout.", "CORRUPT_SCAN")
                tris: List[List[int]] = []
                for ln in chunk:
                    v = ln.split()
                    k = int(v[0])
                    tris.extend(_triangulate_fan([int(t) for t in v[1:1 + k]]))
                if tris:
                    faces_list.append(np.asarray(tris, dtype=np.int64))
    else:
        end = "<" if fmt == "binary_little_endian" else ">"
        offset = body_start
        for el in elements:
            n = el["count"]
            props = el["props"]
            has_list = any(p[1] == "list" for p in props)
            if not has_list:
                dt = np.dtype([(p[0], end + _PLY_TYPES[p[1]]) for p in props])
                need = dt.itemsize * n
                if offset + need > len(raw):
                    raise ScanError("PLY file is truncated.", "CORRUPT_SCAN")
                arr = np.frombuffer(raw, dtype=dt, count=n, offset=offset)
                offset += need
                if el["name"] == "vertex":
                    if not all(c in dt.names for c in ("x", "y", "z")):
                        raise ScanError("PLY vertex element lacks x/y/z properties.", "CORRUPT_SCAN")
                    vertices = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            else:
                if len(props) == 1 and n > 0:
                    _, _, ctype, itype = props[0]
                    ct, it = end + _PLY_TYPES[ctype], end + _PLY_TYPES[itype]
                    csz, isz = np.dtype(ct).itemsize, np.dtype(it).itemsize
                    if offset + csz > len(raw):
                        raise ScanError("PLY file is truncated.", "CORRUPT_SCAN")
                    k0 = int(np.frombuffer(raw, dtype=ct, count=1, offset=offset)[0])
                    rec = csz + k0 * isz
                    fast_ok = offset + rec * n <= len(raw) and k0 >= 3
                    if fast_ok:
                        dt = np.dtype([("n", ct), ("i", it, (k0,))])
                        arr = np.frombuffer(raw, dtype=dt, count=n, offset=offset)
                        if np.all(arr["n"] == k0):
                            idx_arr = arr["i"].astype(np.int64)
                            offset += rec * n
                            if el["name"] == "face":
                                for k in range(1, k0 - 1):
                                    faces_list.append(idx_arr[:, [0, k, k + 1]])
                            continue
                # generic (slow) path for mixed polygon sizes / extra properties
                tris = []
                for _ in range(n):
                    for p in props:
                        if p[1] == "list":
                            ct, it = end + _PLY_TYPES[p[2]], end + _PLY_TYPES[p[3]]
                            csz, isz = np.dtype(ct).itemsize, np.dtype(it).itemsize
                            if offset + csz > len(raw):
                                raise ScanError("PLY file is truncated.", "CORRUPT_SCAN")
                            k = int(np.frombuffer(raw, dtype=ct, count=1, offset=offset)[0])
                            offset += csz
                            if offset + k * isz > len(raw):
                                raise ScanError("PLY file is truncated.", "CORRUPT_SCAN")
                            poly = np.frombuffer(raw, dtype=it, count=k, offset=offset).astype(np.int64).tolist()
                            offset += k * isz
                            if el["name"] == "face":
                                tris.extend(_triangulate_fan(poly))
                        else:
                            sz = np.dtype(_PLY_TYPES[p[1]]).itemsize
                            offset += sz
                if tris:
                    faces_list.append(np.asarray(tris, dtype=np.int64))

    if vertices is None:
        raise ScanError("PLY vertex data could not be read.", "CORRUPT_SCAN")
    faces = np.concatenate(faces_list, axis=0) if faces_list else None
    return vertices, faces


# --------------------------------------------------------------------------- PCD
def _parse_pcd(raw: bytes) -> Tuple[np.ndarray, None]:
    header: dict = {}
    pos = 0
    while True:
        if pos > 65536:
            raise ScanError("PCD header is too long or malformed.", "CORRUPT_SCAN")
        nl = raw.find(b"\n", pos)
        if nl < 0:
            raise ScanError("Not a valid PCD file (no DATA line).", "CORRUPT_SCAN")
        line = raw[pos:nl].decode("ascii", errors="replace").strip()
        pos = nl + 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]
        if parts[0].upper() == "DATA":
            break

    for key in ("FIELDS", "SIZE", "TYPE", "DATA"):
        if key not in header:
            raise ScanError(f"PCD header is missing {key}.", "CORRUPT_SCAN")
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ScanError("PCD header field lists have inconsistent lengths.", "CORRUPT_SCAN")
    if "POINTS" in header:
        n_points = int(header["POINTS"][0])
    elif "WIDTH" in header and "HEIGHT" in header:
        n_points = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    else:
        raise ScanError("PCD header is missing POINTS/WIDTH.", "CORRUPT_SCAN")
    for c in ("x", "y", "z"):
        if c not in fields:
            raise ScanError("PCD file lacks x/y/z fields.", "CORRUPT_SCAN")

    data_kind = header["DATA"][0].lower()
    if data_kind == "ascii":
        total_cols = sum(counts)
        arr = np.fromstring(raw[pos:].decode("ascii", errors="replace"), sep=" ", dtype=np.float64)
        if arr.size != n_points * total_cols:
            raise ScanError("PCD data size does not match the header.", "CORRUPT_SCAN")
        arr = arr.reshape(n_points, total_cols)
        col = {}
        c0 = 0
        for f, c in zip(fields, counts):
            col[f] = c0
            c0 += c
        pts = arr[:, [col["x"], col["y"], col["z"]]]
    elif data_kind == "binary":
        tmap = {"F": "f", "U": "u", "I": "i"}
        dt_fields = []
        for i, (f, s, t, c) in enumerate(zip(fields, sizes, types, counts)):
            if t not in tmap:
                raise ScanError(f"Unsupported PCD field type {t!r}.", "CORRUPT_SCAN")
            name = f if f not in [d[0] for d in dt_fields] else f"{f}_{i}"
            code = f"<{tmap[t]}{s}"
            dt_fields.append((name, code) if c == 1 else (name, code, (c,)))
        dt = np.dtype(dt_fields)
        need = dt.itemsize * n_points
        if pos + need > len(raw):
            raise ScanError("PCD file is truncated.", "CORRUPT_SCAN")
        rec = np.frombuffer(raw, dtype=dt, count=n_points, offset=pos)
        pts = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    else:
        raise ScanError(
            f"PCD DATA type '{data_kind}' is not supported. Re-export the cloud as ascii or binary.",
            "UNSUPPORTED_FORMAT",
        )
    return pts, None


# --------------------------------------------------------------------------- STL
def _parse_stl(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    verts = None
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n == len(raw) and n > 0:
            dt = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            rec = np.frombuffer(raw, dtype=dt, count=n, offset=84)
            verts = rec["v"].reshape(-1, 3).astype(np.float64)
    if verts is None:
        text = raw.decode("ascii", errors="replace")
        found = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
        if not found or len(found) % 3 != 0:
            raise ScanError("Not a valid STL file.", "CORRUPT_SCAN")
        verts = np.asarray(found, dtype=np.float64)
    verts = verts + 0.0  # normalise -0.0 so identical vertices merge
    uniq, inv = np.unique(verts, axis=0, return_inverse=True)
    faces = inv.reshape(-1).reshape(-1, 3)
    return uniq, faces


# --------------------------------------------------------------------------- OBJ
def _parse_obj(raw: bytes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    verts: List[Tuple[float, float, float]] = []
    tris: List[List[int]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            poly = []
            for tok in line.split()[1:]:
                i = int(tok.split("/")[0])
                poly.append(i - 1 if i > 0 else len(verts) + i)
            if len(poly) >= 3:
                tris.extend(_triangulate_fan(poly))
    if not verts:
        raise ScanError("OBJ file contains no vertices.", "CORRUPT_SCAN")
    return np.asarray(verts, dtype=np.float64), (np.asarray(tris, dtype=np.int64) if tris else None)


# --------------------------------------------------------------------------- API
_PARSERS = {".ply": ("ply", _parse_ply), ".pcd": ("pcd", _parse_pcd),
            ".stl": ("stl", _parse_stl), ".obj": ("obj", _parse_obj)}


def load_scan(raw: bytes, filename: str, min_points: int = 200,
              allowed: Optional[List[str]] = None) -> ScanData:
    """Parse and validate an uploaded scan. Raises ``ScanError`` on any problem."""
    ext = os.path.splitext(filename or "")[1].lower()
    allowed = allowed or list(_PARSERS)
    if ext not in _PARSERS or ext not in allowed:
        raise ScanError(
            f"Unsupported file type '{ext or filename}'. Allowed: {', '.join(sorted(allowed))}.",
            "UNSUPPORTED_FORMAT",
        )
    if len(raw) == 0:
        raise ScanError("The uploaded file is empty.", "EMPTY_FILE")

    fmt, parser = _PARSERS[ext]
    try:
        pts, faces = parser(raw)
    except ScanError:
        raise
    except Exception as exc:  # parser-specific numeric/IO errors -> clean rejection
        raise ScanError(f"Could not read {fmt.upper()} data: {exc}", "CORRUPT_SCAN") from exc

    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ScanError("Scan geometry is not an N x 3 array of coordinates.", "CORRUPT_SCAN")
    raw_count = int(pts.shape[0])
    notes: List[str] = []

    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        notes.append(f"Dropped {int((~finite).sum())} non-finite (NaN/Inf) points.")
        remap = np.cumsum(finite) - 1
        pts = pts[finite]
        if faces is not None:
            keep = finite[faces].all(axis=1)
            faces = remap[faces[keep]]
    if pts.shape[0] < min_points:
        raise ScanError(
            f"Scan contains only {pts.shape[0]} valid points; at least {min_points} are required.",
            "TOO_FEW_POINTS",
        )
    if faces is not None:
        if faces.size == 0:
            faces = None
        elif faces.min() < 0 or faces.max() >= pts.shape[0]:
            raise ScanError("Mesh references vertices that do not exist.", "CORRUPT_SCAN")
    if float(np.ptp(pts, axis=0).max()) < 1e-9:
        raise ScanError("Scan geometry is degenerate (all points coincide).", "DEGENERATE_SCAN")

    return ScanData(
        points=pts.astype(np.float32),
        faces=None if faces is None else faces.astype(np.int32),
        fmt=fmt,
        raw_point_count=raw_count,
        notes=notes,
    )
