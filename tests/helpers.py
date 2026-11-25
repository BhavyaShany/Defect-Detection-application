"""Writers for test scan files (all formats the loader supports) and a synthetic part."""
import struct
import numpy as np


def make_plate_with_boss(n=6000, seed=0, noise=0.0005):
    """Nominal synthetic part: flat plate with a raised circular boss (point cloud, ~1 unit wide)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.35, 0.35, n)
    r = np.hypot(x, y)
    z = np.where(r < 0.15, 0.08, 0.0) + rng.normal(0, noise, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def ply_ascii(points, faces=None):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
             "property float x", "property float y", "property float z"]
    if faces is not None:
        lines += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    lines.append("end_header")
    lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points]
    if faces is not None:
        lines += [f"3 {f[0]} {f[1]} {f[2]}" for f in faces]
    return ("\n".join(lines) + "\n").encode()


def ply_binary(points, faces=None, big_endian=False):
    e = ">" if big_endian else "<"
    fmt = "binary_big_endian" if big_endian else "binary_little_endian"
    head = ["ply", f"format {fmt} 1.0", f"element vertex {len(points)}",
            "property float x", "property float y", "property float z", "property uchar red"]
    if faces is not None:
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    head.append("end_header\n")
    out = bytearray("\n".join(head).encode())
    vdt = np.dtype([("x", e + "f4"), ("y", e + "f4"), ("z", e + "f4"), ("r", "u1")])
    arr = np.zeros(len(points), dtype=vdt)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    out += arr.tobytes()
    if faces is not None:
        fdt = np.dtype([("n", "u1"), ("i", e + "i4", (3,))])
        f = np.zeros(len(faces), dtype=fdt)
        f["n"] = 3
        f["i"] = faces
        out += f.tobytes()
    return bytes(out)


def pcd_ascii(points):
    h = ["# .PCD v0.7", "VERSION 0.7", "FIELDS x y z", "SIZE 4 4 4", "TYPE F F F", "COUNT 1 1 1",
         f"WIDTH {len(points)}", "HEIGHT 1", "VIEWPOINT 0 0 0 1 0 0 0", f"POINTS {len(points)}", "DATA ascii"]
    return ("\n".join(h + [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points]) + "\n").encode()


def pcd_binary(points):
    h = ["VERSION 0.7", "FIELDS x y z intensity", "SIZE 4 4 4 4", "TYPE F F F F", "COUNT 1 1 1 1",
         f"WIDTH {len(points)}", "HEIGHT 1", f"POINTS {len(points)}", "DATA binary"]
    arr = np.zeros((len(points), 4), dtype="<f4")
    arr[:, :3] = points
    return ("\n".join(h) + "\n").encode() + arr.tobytes()


def stl_binary(points_tri):
    """points_tri: (M,3,3) triangle vertices."""
    m = len(points_tri)
    out = bytearray(b"\0" * 80 + struct.pack("<I", m))
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    a = np.zeros(m, dtype=dt)
    a["v"] = points_tri
    return bytes(out) + a.tobytes()


def stl_ascii(points_tri):
    s = ["solid t"]
    for tri in points_tri:
        s.append("facet normal 0 0 1\n outer loop")
        s += [f"  vertex {v[0]} {v[1]} {v[2]}" for v in tri]
        s.append(" endloop\nendfacet")
    s.append("endsolid t")
    return "\n".join(s).encode()


def obj_text(points, faces=None):
    lines = [f"v {p[0]} {p[1]} {p[2]}" for p in points]
    if faces is not None:
        lines += [f"f {f[0]+1}/1/1 {f[1]+1}/2/2 {f[2]+1}/3/3" for f in faces]
    return "\n".join(lines).encode()


def grid_mesh(n=20):
    """Small flat triangulated grid for mesh-format tests."""
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return pts, np.asarray(faces, dtype=np.int32)
