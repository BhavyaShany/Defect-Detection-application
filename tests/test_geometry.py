import time
import unittest
import numpy as np

from backend.services import geometry as G
from backend.services.preprocessing import run_preprocessing, SURFACE_VARIATION_REFERENCE
from backend.services.scan_io import load_scan, ScanError
from tools.synthetic_parts import generate_part, write_ply_binary, PARTS, DEFECTS
from tests import helpers as H


def _scan(part, defect, n=40000, seed=3):
    pts, gt = generate_part(part, defect, n=n, seed=seed)
    return load_scan(H.ply_binary(pts), "t.ply", min_points=200), pts, gt


class TestPose(unittest.TestCase):
    def test_pca_is_rotation_invariant_and_unit_scaled(self):
        pts, _ = generate_part("engine_bracket", "none", n=20000, seed=1)
        ang = 0.9
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        rot = (pts @ R.T + [5, -3, 2]).astype(np.float32)
        a, _ = G.normalize_pose(pts, "pca")
        b, _ = G.normalize_pose(rot, "pca")
        self.assertAlmostEqual(float(np.ptp(a, axis=0).max()), 1.0, places=4)
        # same shape after canonicalisation (compare sorted extents)
        np.testing.assert_allclose(np.sort(np.ptp(a, axis=0)), np.sort(np.ptp(b, axis=0)), atol=5e-3)

    def test_original_extents_preserved_in_info(self):
        pts = (H.make_plate_with_boss(2000) * 10).astype(np.float32)
        _, info = G.normalize_pose(pts, "none")
        self.assertAlmostEqual(info["original_extents"][0], float(np.ptp(pts[:, 0])), places=3)


class TestOutliers(unittest.TestCase):
    def test_injected_far_points_are_removed(self):
        pts = H.make_plate_with_boss(5000)
        noise = np.random.default_rng(1).uniform(-3, 3, (30, 3)).astype(np.float32)
        allp = np.vstack([pts, noise])
        mask = G.statistical_outlier_mask(allp, 20, 2.0)
        self.assertFalse(mask[-30:].any(), "injected outliers must be rejected")
        self.assertGreater(mask[:5000].mean(), 0.9, "most genuine points must be kept")


class TestICP(unittest.TestCase):
    def test_recovers_known_rigid_transform(self):
        tgt = H.make_plate_with_boss(3000, seed=2)
        ang = 0.15
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], dtype=np.float32)
        src = tgt @ R.T + np.array([0.03, -0.02, 0.01], dtype=np.float32)
        aligned, rmse, fit, _ = G.icp_align(src, tgt, max_dist=0.2, max_iter=100)
        self.assertLess(float(np.mean(np.linalg.norm(aligned - tgt, axis=1))), 0.01)
        self.assertGreater(fit, 0.9)


class TestSurfaceVariation(unittest.TestCase):
    def test_flat_plane_is_near_zero(self):
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-.5, .5, 8000), rng.uniform(-.5, .5, 8000), np.zeros(8000)], 1).astype(np.float32)
        v, _ = G.surface_variation(pts, 16)
        self.assertLess(float(np.median(v)), 1e-6)

    def test_curved_and_noisy_surfaces_score_above_a_plane(self):
        rng = np.random.default_rng(0)
        plane = np.stack([rng.uniform(-.5, .5, 8000), rng.uniform(-.5, .5, 8000), np.zeros(8000)], 1).astype(np.float32)
        noisy = plane.copy()
        noisy[:, 2] += rng.normal(0, 0.01, 8000)
        d = rng.normal(size=(8000, 3))
        sphere = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)
        vp = np.median(G.surface_variation(plane, 16)[0])
        vn = np.median(G.surface_variation(noisy, 16)[0])
        vs = np.median(G.surface_variation(sphere, 16)[0])
        self.assertGreater(vn, 100 * vp + 1e-4)
        self.assertGreater(vs, 100 * vp + 1e-5)

    def test_defect_regions_score_higher_than_flat_surface(self):
        """Pits, grooves, bumps and ridges must stand out in the geometric analysis.

        Gentle shrinkage depressions are deliberately NOT asserted: at k=16 they are
        indistinguishable from flat surface (measured ratio ~1.1) -- a documented limitation.
        """
        for defect in ("porosity", "crack", "inclusion", "flash"):
            pts, gt = generate_part("engine_bracket", defect, n=60000, seed=5)
            norm, _ = G.normalize_pose(pts, "none")  # translation+scale only -> easy to map coordinates
            scale = float(np.ptp(pts, axis=0).max())
            centroid = pts.mean(0)
            v, _ = G.surface_variation(norm, 16)
            to_n = lambda xy: (np.asarray(xy) - centroid[:2]) / scale
            c = to_n(gt["center"])
            xy = norm[:, :2]
            near = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) < gt["radius"] / scale
            f = to_n((0.3, -0.2))  # flat plate region, far from boss and defect
            far_flat = (np.abs(xy[:, 0] - f[0]) < 0.08) & (np.abs(xy[:, 1] - f[1]) < 0.08)
            self.assertGreater(near.sum(), 50)
            self.assertGreater(far_flat.sum(), 200)
            self.assertGreater(v[near].mean(), 2.5 * v[far_flat].mean(), defect)

    def test_nominal_scores_lower_than_defective_overall(self):
        nom, _, _ = _scan("engine_bracket", "none")
        bad, _, _ = _scan("engine_bracket", "porosity")
        a = run_preprocessing(nom).metrics["surface_variation"]
        b = run_preprocessing(bad).metrics["surface_variation"]
        self.assertGreater(b["fraction_above_reference"], a["fraction_above_reference"])


class TestProjection(unittest.TestCase):
    def test_shapes_range_and_no_data_semantics(self):
        pts, _ = generate_part("brake_rotor", "none", n=40000, seed=1)
        norm, _ = G.normalize_pose(pts, "pca")
        m = G.project_to_maps(norm, 224)
        self.assertEqual(m["depth"].shape, (224, 224))
        self.assertEqual(m["normal"].shape, (224, 224, 3))
        self.assertEqual(m["depth"].dtype, np.uint8)
        self.assertGreater(m["coverage"], 0.3)
        self.assertEqual(m["depth"][0, 0], 0, "corner of a disc scan must be background")

    def test_flat_noisy_part_does_not_become_high_contrast(self):
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-.5, .5, 40000), rng.uniform(-.5, .5, 40000), rng.normal(0, 0.0005, 40000)], 1)
        m = G.project_to_maps(pts.astype(np.float32), 224)
        vals = m["depth"][m["depth"] > 0]
        self.assertLess(int(vals.max()) - int(vals.min()), 40)

    def test_pit_is_visible_in_depth_map(self):
        flat, _ = generate_part("engine_bracket", "none", n=60000, seed=2)
        pit, _ = generate_part("engine_bracket", "inclusion", n=60000, seed=2)
        a = G.project_to_maps(G.normalize_pose(flat, "none")[0], 224)["depth"].astype(int)
        b = G.project_to_maps(G.normalize_pose(pit, "none")[0], 224)["depth"].astype(int)
        self.assertGreater(np.abs(a - b).max(), 20)

    def test_model_input_modes(self):
        d = np.zeros((224, 224), np.uint8)
        n = np.zeros((224, 224, 3), np.uint8)
        for mode in ("depth", "normal", "depth_normal"):
            self.assertEqual(G.build_model_input(d, n, mode).shape, (224, 224, 3))
        with self.assertRaises(ValueError):
            G.build_model_input(d, n, "rgb")


class TestPipeline(unittest.TestCase):
    def test_every_part_and_defect_runs_and_reports_timings(self):
        for part in PARTS:
            for defect in DEFECTS:
                scan, _, _ = _scan(part, defect, n=20000)
                r = run_preprocessing(scan)
                for key in ("align_ms", "outlier_filter_ms", "surface_analysis_ms", "projection_ms", "preprocessing_total_ms"):
                    self.assertIn(key, r.timings_ms)
                self.assertEqual(len(r.view_positions), len(r.view_scores))
                self.assertTrue(np.isfinite(r.view_positions).all())
                self.assertTrue(((r.view_scores >= 0) & (r.view_scores <= 1)).all())

    def test_large_cloud_is_capped_for_viewer(self):
        scan, _, _ = _scan("engine_bracket", "none", n=120000)
        r = run_preprocessing(scan)
        self.assertLessEqual(len(r.view_positions), 60000)

    def test_mesh_keeps_faces_when_small(self):
        pts, faces = H.grid_mesh(30)
        scan = load_scan(H.ply_binary(pts, faces), "m.ply", min_points=100)
        r = run_preprocessing(scan)
        self.assertIsNotNone(r.view_faces)
        self.assertEqual(r.view_faces.max(), len(r.view_positions) - 1)

    def test_sparse_thin_scan_is_rejected(self):
        pts = np.stack([np.linspace(0, 1, 400), np.zeros(400), np.zeros(400)], 1).astype(np.float32)
        pts[:, 1] += np.random.default_rng(0).normal(0, 1e-4, 400)
        with self.assertRaises(ScanError):
            run_preprocessing(load_scan(H.ply_binary(pts), "line.ply", min_points=200))

    def test_timing_of_typical_scan_is_recorded_not_assumed(self):
        scan, _, _ = _scan("engine_bracket", "porosity", n=100000)
        t = time.perf_counter()
        r = run_preprocessing(scan)
        wall = (time.perf_counter() - t) * 1000
        self.assertAlmostEqual(r.timings_ms["preprocessing_total_ms"], wall, delta=max(50, wall * 0.1))


if __name__ == "__main__":
    unittest.main()
