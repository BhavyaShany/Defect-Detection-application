"""End-to-end API tests (FastAPI TestClient). Skipped automatically if FastAPI is not installed.

Run:  pip install -r requirements-dev.txt && python -m unittest tests.test_api -v
Uses a throw-away SQLite database and data directory; attaches a stub model so the full
PASS / FAIL / REVIEW paths are exercised without TensorFlow.
"""
import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="defect_test_")
os.environ["DATA_DIR"] = _TMP
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["MODEL_PATH"] = os.path.join(_TMP, "none.keras")  # absent on purpose
os.environ["MAX_UPLOAD_MB"] = "2"

try:
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.services.runtime import inference_engine
    HAVE_STACK = True
except ImportError:  # pragma: no cover
    HAVE_STACK = False

import io

import numpy as np
from PIL import Image
from tests import helpers as H

LABELS = {"class_names": ["defective", "ok"], "pass_classes": ["ok"], "input_mode": "depth",
          "img_size": 224, "model_name": "stub-model"}


def _ply(n=3000, seed=0):
    return H.ply_binary(H.make_plate_with_boss(n, seed=seed))


@unittest.skipUnless(HAVE_STACK, "FastAPI stack not installed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()  # runs lifespan (DB init + model load attempt)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def upload(self, raw, name="a.ply", **form):
        return self.client.post("/api/scans/upload", files={"file": (name, raw)}, data=form)

    # ---- health / no-model honesty
    def test_01_health_reports_model_not_loaded_truthfully(self):
        j = self.client.get("/api/health").json()
        self.assertFalse(j["model_loaded"])
        self.assertTrue(j["model_error"])

    def test_02_empty_database_has_null_not_fake_metrics(self):
        s = self.client.get("/api/metrics/summary").json()
        self.assertEqual(s["total_scanned"], 0)
        self.assertIsNone(s["pass_rate_pct"])
        self.assertIsNone(s["avg_latency_ms"])
        self.assertEqual(self.client.get("/api/scans").json()["total"], 0)

    def test_03_without_model_scan_is_unclassified_with_no_fabricated_values(self):
        r = self.upload(_ply())
        self.assertEqual(r.status_code, 201, r.text)
        j = r.json()
        self.assertEqual(j["status"], "UNCLASSIFIED")
        self.assertIsNone(j["confidence"])
        self.assertIsNone(j["defect_type"])
        self.assertFalse(j["localization"]["available"])
        self.assertTrue(any("No trained model" in w for w in j["warnings"]))
        for k in ("server_total_ms", "parse_ms", "projection_ms", "surface_analysis_ms"):
            self.assertIn(k, j["timings"])
        g = self.client.get(f"/api/scans/{j['id']}/geometry")
        self.assertEqual(g.status_code, 200)
        self.assertGreater(g.json()["point_count"], 100)

    # ---- with a stub model
    def test_04_pass_fail_review_paths(self):
        cases = [([0.05, 0.95], "PASS", "ok"), ([0.9, 0.1], "FAIL", "defective"), ([0.45, 0.55], "REVIEW", "ok")]
        for probs, status, cls in cases:
            inference_engine.attach(lambda b, p=probs: np.array(p), LABELS)
            r = self.upload(_ply(seed=hash(status) % 1000), name=f"{status}.ply")
            self.assertEqual(r.status_code, 201, r.text)
            j = r.json()
            self.assertEqual((j["status"], j["defect_type"]), (status, cls))
            self.assertAlmostEqual(j["confidence"], max(probs), places=3)
            self.assertIn("inference_ms", j["timings"])
            self.assertEqual(j["model_name"], "stub-model")

    def test_05_statistics_and_history_reflect_real_records(self):
        s = self.client.get("/api/metrics/summary").json()
        self.assertGreaterEqual(s["total_scanned"], 4)
        self.assertEqual(s["passed"] + s["failed"] + s["needs_review"] + s["unclassified"], s["total_scanned"])
        items = self.client.get("/api/scans?status=FAIL").json()["items"]
        self.assertTrue(items and all(i["status"] == "FAIL" for i in items))
        self.assertTrue(any(p["defect_type"] == "defective" for p in self.client.get("/api/metrics/pareto").json()))
        self.assertIn("stages_ms", self.client.get("/api/metrics/performance").json())
        self.assertEqual(len(self.client.get("/api/metrics/activity?hours=6").json()), 6)

    def test_06_duplicate_upload_is_flagged_not_blocked(self):
        raw = _ply(seed=99)
        first = self.upload(raw, name="dup.ply").json()
        second = self.upload(raw, name="dup.ply").json()
        self.assertEqual(second["duplicate_of"], first["id"])

    # ---- validation / failure paths
    def test_07_invalid_inputs_fail_gracefully(self):
        cases = [
            (b"hello", "a.jpg", 415),
            (b"", "a.ply", 400),
            (b"\xff\xd8\xff garbage", "a.ply", 422),
            (H.ply_binary(H.make_plate_with_boss(500))[:300], "t.ply", 422),
            (H.ply_ascii(H.make_plate_with_boss(20)), "few.ply", 422),
        ]
        for raw, name, code in cases:
            r = self.upload(raw, name=name)
            self.assertEqual(r.status_code, code, f"{name}: {r.text}")
            d = r.json()["detail"]
            self.assertIn("code", d)
            self.assertIn("message", d)

    def test_08_oversized_upload_rejected(self):
        r = self.upload(b"0" * (3 * 1024 * 1024), name="big.ply")
        self.assertEqual(r.status_code, 413)

    def test_09_inference_failure_degrades_to_unclassified(self):
        def boom(batch):
            raise RuntimeError("bad model")
        inference_engine.attach(boom, LABELS)
        j = self.upload(_ply(seed=5), name="x.ply").json()
        self.assertEqual(j["status"], "UNCLASSIFIED")
        self.assertTrue(any("inference failed" in w.lower() for w in j["warnings"]))

    def test_10_unknown_ids_and_bad_geometry_ids(self):
        self.assertEqual(self.client.get("/api/scans/does-not-exist").status_code, 404)
        self.assertEqual(self.client.get("/api/scans/does-not-exist/geometry").status_code, 404)

    def test_11_websocket_events(self):
        inference_engine.attach(lambda b: np.array([0.1, 0.9]), LABELS)
        with self.client.websocket_connect("/ws/live") as ws:
            self.assertEqual(ws.receive_json()["type"], "hello")
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json()["type"], "pong")
            self.assertEqual(self.upload(_ply(seed=123), name="ws.ply").status_code, 201)
            types = []
            while "result" not in types:
                types.append(ws.receive_json()["type"])
            self.assertIn("progress", types)
            self.upload(b"", name="bad.ply")
            seen = []
            while "failed" not in seen and len(seen) < 6:
                seen.append(ws.receive_json()["type"])
            self.assertIn("failed", seen)

    def test_12_export_is_formula_injection_safe(self):
        self.upload(_ply(seed=7), name="=cmd.ply")
        csv = self.client.get("/api/metrics/export?format=csv").text
        self.assertNotIn(",=cmd.ply", csv)
        self.assertIn("'=cmd.ply", csv)

    def test_13_system_status(self):
        j = self.client.get("/api/system/status").json()
        self.assertEqual(j["database"]["backend"], "sqlite")
        self.assertIn("model", j)

    # ---- photo (image-mode model) inspection
    def _photo(self, seed=0, size=(300, 300)):
        rng = np.random.default_rng(seed)
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, size[::-1], dtype=np.uint8), "L").save(buf, format="PNG")
        return buf.getvalue()

    def test_14_photos_are_refused_unless_an_image_model_is_loaded(self):
        inference_engine.attach(lambda b: np.array([0.1, 0.9]), LABELS)  # depth-mode model
        r = self.upload(self._photo(1), name="p.png")
        self.assertEqual(r.status_code, 415, r.text)
        self.assertFalse(self.client.get("/api/system/status").json()["limits"]["image_inspection"])

    def test_15_photo_inspection_with_image_model(self):
        labels = {**LABELS, "class_names": ["def_front", "ok_front"], "pass_classes": ["ok_front"],
                  "input_mode": "image"}
        inference_engine.attach(lambda b: np.array([0.92, 0.08]), labels)
        st = self.client.get("/api/system/status").json()["limits"]
        self.assertTrue(st["image_inspection"])
        self.assertIn(".png", st["allowed_extensions"])
        r = self.upload(self._photo(2), name="casting.png")
        self.assertEqual(r.status_code, 201, r.text)
        j = r.json()
        self.assertEqual((j["input_type"], j["status"], j["defect_type"]), ("image", "FAIL", "def_front"))
        self.assertTrue(j["image_base64"].startswith("data:image/jpeg"))
        self.assertIn("decode_ms", j["timings"])
        self.assertFalse(j["has_geometry"])
        self.assertEqual(self.client.get(f"/api/scans/{j['id']}/geometry").status_code, 404)
        # corrupt / tiny photos fail cleanly
        self.assertEqual(self.upload(b"not an image", name="x.png").status_code, 422)
        tiny = io.BytesIO()
        Image.new("L", (8, 8)).save(tiny, format="PNG")
        self.assertEqual(self.upload(tiny.getvalue(), name="tiny.png").status_code, 422)

    def test_16_3d_scan_with_image_model_is_unclassified_with_explanation(self):
        labels = {**LABELS, "input_mode": "image"}
        inference_engine.attach(lambda b: np.array([0.1, 0.9]), labels)
        j = self.upload(_ply(seed=77), name="scan.ply").json()
        self.assertEqual(j["status"], "UNCLASSIFIED")
        self.assertTrue(any("classifies photographs" in w for w in j["warnings"]))

    def test_17_input_check_refuses_non_casting_photo_with_explanation(self):
        from backend.services.ood import OODDetector
        det = OODDetector(np.zeros(8), np.eye(8), 100.0, {"grayscale_training": True, "color_tolerance": 8.0})
        feats = {"v": np.zeros(8)}
        labels = {**LABELS, "class_names": ["def_front", "ok_front"], "pass_classes": ["ok_front"],
                  "input_mode": "image"}
        inference_engine.attach(lambda b: (np.array([0.01, 0.99]), feats["v"]), labels, ood=det)
        good = self.upload(self._photo(11), name="casting.png").json()
        self.assertEqual((good["status"], good["defect_type"]), ("PASS", "ok_front"))
        self.assertFalse(good["metrics"]["ood"]["flagged"])
        feats["v"] = np.full(8, 50.0)  # far from the training photos, yet the softmax is 99 % "ok_front"
        r = self.upload(self._photo(12), name="selfie.png")
        self.assertEqual(r.status_code, 201, r.text)
        j = r.json()
        self.assertEqual(j["status"], "UNCLASSIFIED")
        self.assertIsNone(j["defect_type"])
        self.assertIsNone(j["confidence"])
        self.assertTrue(j["metrics"]["ood"]["flagged"])
        self.assertTrue(any("does not look like the casting photos" in w for w in j["warnings"]))
        st = self.client.get("/api/system/status").json()["model"]["ood"]
        self.assertTrue(st["enabled"])
        inference_engine.ood = None


if __name__ == "__main__":
    unittest.main()
