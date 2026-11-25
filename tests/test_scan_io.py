import unittest
import numpy as np

from backend.services.scan_io import load_scan, ScanError
from tests import helpers as H


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.pts = H.make_plate_with_boss(1500)

    def _roundtrip(self, raw, name, tol=1e-4):
        s = load_scan(raw, name, min_points=100)
        self.assertEqual(s.points.shape, self.pts.shape)
        np.testing.assert_allclose(s.points, self.pts, atol=tol)
        return s

    def test_ply_ascii(self):
        self.assertEqual(self._roundtrip(H.ply_ascii(self.pts), "a.ply").fmt, "ply")

    def test_ply_binary_le_and_be(self):
        self._roundtrip(H.ply_binary(self.pts), "a.ply")
        self._roundtrip(H.ply_binary(self.pts, big_endian=True), "a.PLY")

    def test_ply_mesh_binary_and_ascii(self):
        pts, faces = H.grid_mesh(15)
        for raw in (H.ply_binary(pts, faces), H.ply_ascii(pts, faces)):
            s = load_scan(raw, "m.ply", min_points=100)
            self.assertEqual(s.faces.shape, faces.shape)
            np.testing.assert_array_equal(s.faces, faces)

    def test_pcd_ascii_and_binary(self):
        self._roundtrip(H.pcd_ascii(self.pts), "a.pcd")
        self._roundtrip(H.pcd_binary(self.pts), "a.pcd")

    def test_stl_binary_and_ascii_share_vertices(self):
        pts, faces = H.grid_mesh(12)
        tri = pts[faces]
        for raw in (H.stl_binary(tri), H.stl_ascii(tri)):
            s = load_scan(raw, "m.stl", min_points=100)
            self.assertEqual(len(s.points), len(pts))  # vertices deduplicated
            self.assertEqual(len(s.faces), len(faces))

    def test_obj(self):
        pts, faces = H.grid_mesh(12)
        s = load_scan(H.obj_text(pts, faces), "m.obj", min_points=100)
        self.assertEqual(len(s.points), len(pts))
        np.testing.assert_array_equal(s.faces, faces)

    def test_nan_points_are_dropped_and_reported(self):
        p = self.pts.copy()
        p[5] = np.nan
        s = load_scan(H.pcd_ascii(p).replace(b"nan", b"nan"), "a.pcd", min_points=100)
        self.assertEqual(len(s.points), len(p) - 1)
        self.assertTrue(any("non-finite" in n for n in s.notes))


class TestRejection(unittest.TestCase):
    def expect(self, raw, name, code, min_points=100):
        with self.assertRaises(ScanError) as cm:
            load_scan(raw, name, min_points=min_points)
        self.assertEqual(cm.exception.code, code, cm.exception.message)

    def test_unsupported_extension(self):
        self.expect(b"hello", "scan.jpg", "UNSUPPORTED_FORMAT")
        self.expect(b"hello", "scan", "UNSUPPORTED_FORMAT")

    def test_empty_file(self):
        self.expect(b"", "a.ply", "EMPTY_FILE")

    def test_garbage_with_valid_extension_is_rejected_not_fabricated(self):
        self.expect(b"\xff\xd8\xff\xe0 not a ply", "photo.ply", "CORRUPT_SCAN")
        self.expect(b"random text", "a.pcd", "CORRUPT_SCAN")
        self.expect(b"random text", "a.stl", "CORRUPT_SCAN")
        self.expect(b"random text", "a.obj", "CORRUPT_SCAN")

    def test_truncated_binary_ply(self):
        raw = H.ply_binary(H.make_plate_with_boss(500))
        self.expect(raw[: len(raw) // 2], "a.ply", "CORRUPT_SCAN")

    def test_truncated_binary_pcd(self):
        raw = H.pcd_binary(H.make_plate_with_boss(500))
        self.expect(raw[:-100], "a.pcd", "CORRUPT_SCAN")

    def test_too_few_points(self):
        self.expect(H.ply_ascii(H.make_plate_with_boss(20)), "a.ply", "TOO_FEW_POINTS", min_points=100)

    def test_degenerate_geometry(self):
        pts = np.zeros((300, 3), dtype=np.float32)
        self.expect(H.ply_ascii(pts), "a.ply", "DEGENERATE_SCAN")

    def test_unsupported_pcd_compression(self):
        raw = H.pcd_binary(H.make_plate_with_boss(300)).replace(b"DATA binary", b"DATA binary_compressed")
        self.expect(raw, "a.pcd", "UNSUPPORTED_FORMAT")

    def test_bad_face_indices(self):
        pts, faces = H.grid_mesh(12)
        faces = faces.copy()
        faces[0, 0] = 99999
        self.expect(H.ply_ascii(pts, faces), "a.ply", "CORRUPT_SCAN")


if __name__ == "__main__":
    unittest.main()
