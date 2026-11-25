import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.services.model_inference import DefectInferenceEngine
from backend.services.ood import OODDetector, color_saturation, fit_gaussian, squared_mahalanobis

LABELS = {"class_names": ["def_front", "ok_front"], "pass_classes": ["ok_front"],
          "input_mode": "image", "img_size": 64, "model_name": "stub"}
D = 12


def _detector(grayscale=True, pct=99.5):
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (3000, D))
    mean, prec = fit_gaussian(train, 0.05)
    held = rng.normal(0, 1, (2000, D))
    thr = float(np.percentile(squared_mahalanobis(held, mean, prec), pct))
    return OODDetector(mean, prec, thr, {"grayscale_training": grayscale, "color_tolerance": 8.0, "percentile": pct})


class TestMath(unittest.TestCase):
    def test_in_distribution_scores_about_dimension_and_far_points_score_high(self):
        det = _detector()
        rng = np.random.default_rng(1)
        s = squared_mahalanobis(rng.normal(0, 1, (4000, D)), det.mean, det.precision)
        self.assertAlmostEqual(float(s.mean()), D, delta=1.0)           # chi-square(D) mean
        self.assertLess(float(np.mean(s > det.threshold)), 0.02)        # ~0.5 % flagged by construction
        self.assertGreater(det.score(np.full(D, 8.0)), 10 * det.threshold)

    def test_shrinkage_makes_degenerate_features_invertible(self):
        x = np.random.default_rng(0).normal(size=(50, 30))
        x[:, 5] = x[:, 4]                                               # perfectly correlated -> singular covariance
        mean, prec = fit_gaussian(x, 0.1)
        self.assertTrue(np.isfinite(prec).all())

    def test_save_load_roundtrip_without_pickle(self):
        det = _detector()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ood.npz"
            det.save(p)
            back = OODDetector.load(p)
        self.assertAlmostEqual(back.threshold, det.threshold)
        self.assertEqual(back.meta["grayscale_training"], True)
        v = np.full(D, 2.0)
        self.assertAlmostEqual(back.score(v), det.score(v), places=3)

    def test_color_saturation(self):
        g = np.full((20, 20, 3), 90, np.uint8)
        c = g.copy()
        c[..., 0] = 200
        self.assertEqual(color_saturation(g), 0.0)
        self.assertGreater(color_saturation(c), 50)


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((40, 40, 3), 120, np.uint8)
        self.color = self.gray.copy()
        self.color[..., 2] = 250

    def test_distance_gate(self):
        det = _detector()
        ok = det.check(np.zeros(D), self.gray)
        bad = det.check(np.full(D, 9.0), self.gray)
        self.assertFalse(ok["flagged"])
        self.assertEqual(bad["reasons"], ["unusual_appearance"])
        self.assertIn("not classified", bad["message"])

    def test_color_gate_only_when_training_was_grayscale(self):
        flagged = _detector(grayscale=True).check(np.zeros(D), self.color)
        self.assertEqual(flagged["reasons"], ["color_image"])
        self.assertFalse(_detector(grayscale=False).check(np.zeros(D), self.color)["flagged"])
        self.assertFalse(_detector(grayscale=True).check(np.zeros(D), self.color, color_check=False)["flagged"])


class TestEngineWithInputCheck(unittest.TestCase):
    def _engine(self, feats):
        e = DefectInferenceEngine(Path("x"), Path("y"), threshold=0.75)
        e.attach(lambda b: (np.array([0.02, 0.98]), feats), LABELS, ood=_detector())
        return e

    def test_genuine_image_is_classified_and_reports_the_check(self):
        r = self._engine(np.zeros(D)).predict_image(np.full((90, 90, 3), 100, np.uint8))
        self.assertEqual((r["status"], r["predicted_class"]), ("PASS", "ok_front"))
        self.assertFalse(r["ood"]["flagged"])
        self.assertNotIn("_feats", r)

    def test_non_casting_image_gets_no_class_and_no_confidence_despite_confident_softmax(self):
        r = self._engine(np.full(D, 9.0)).predict_image(np.full((90, 90, 3), 100, np.uint8))
        self.assertEqual(r["status"], "UNCLASSIFIED")
        self.assertIsNone(r["predicted_class"])
        self.assertIsNone(r["confidence"])
        self.assertEqual(r["probabilities"], {})
        self.assertTrue(r["ood"]["flagged"])

    def test_colour_photo_is_refused_even_with_in_distribution_features(self):
        colour = np.zeros((90, 90, 3), np.uint8)
        colour[..., 0] = 220
        r = self._engine(np.zeros(D)).predict_image(colour)
        self.assertEqual(r["status"], "UNCLASSIFIED")
        self.assertEqual(r["ood"]["reasons"], ["color_image"])

    def test_without_detector_behaviour_is_unchanged_and_check_reported_inactive(self):
        e = DefectInferenceEngine(Path("x"), Path("y"))
        e.attach(lambda b: np.array([0.02, 0.98]), LABELS)
        r = e.predict_image(np.full((90, 90, 3), 100, np.uint8))
        self.assertEqual(r["status"], "PASS")
        self.assertIsNone(r["ood"])
        info = e.info()["ood"]
        self.assertFalse(info["enabled"])

    def test_info_exposes_threshold_when_enabled(self):
        info = self._engine(np.zeros(D)).info()["ood"]
        self.assertTrue(info["enabled"])
        self.assertGreater(info["threshold"], 0)


class TestFitHelpers(unittest.TestCase):
    def test_synthetic_negatives_and_unreadable_negatives_are_skipped(self):
        import cv2
        from ml.fit_ood import read_images, synthetic_negatives
        neg = synthetic_negatives(n_each=3, size=64)
        self.assertEqual({len(v) for v in neg.values()}, {3})
        self.assertEqual(neg["flat image"][0].shape, (64, 64, 3))
        d = Path(tempfile.mkdtemp())
        cv2.imwrite(str(d / "ok.png"), np.random.default_rng(0).integers(0, 255, (50, 50), dtype=np.uint8))
        (d / "broken.jpg").write_bytes(b"not an image")
        self.assertEqual(len(read_images(d)), 1)


if __name__ == "__main__":
    unittest.main()


class TestOriginPolicy(unittest.TestCase):
    def test_policy(self):
        from backend.services.origin import origin_allowed as ok
        allowed = ["http://localhost:5173"]
        self.assertTrue(ok(None, "x", allowed))                                   # non-browser client
        self.assertTrue(ok("http://localhost:5173", "localhost:8000", allowed))   # dev dashboard
        self.assertTrue(ok("https://demo.example.com", "demo.example.com", allowed))  # same-origin (served by backend)
        self.assertTrue(ok("https://abc.trycloudflare.com", "ABC.trycloudflare.com", allowed))
        self.assertFalse(ok("https://evil.example", "demo.example.com", allowed)) # cross-site page
        self.assertFalse(ok("https://evil.example", None, allowed))
        self.assertFalse(ok("http://localhost:5174", "localhost:8000", allowed))
