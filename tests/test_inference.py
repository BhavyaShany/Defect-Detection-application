import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.services.model_inference import (
    DefectInferenceEngine, InferenceError, ModelUnavailableError,
    interpret_prediction, to_probabilities, validate_labels,
)

LABELS = {"class_names": ["def_front", "ok_front"], "pass_classes": ["ok_front"],
          "input_mode": "depth", "img_size": 224, "model_name": "stub"}
D = np.full((224, 224), 128, np.uint8)
N = np.full((224, 224, 3), 128, np.uint8)


class TestDecisionLogic(unittest.TestCase):
    def test_pass_fail_review(self):
        names, passes = LABELS["class_names"], LABELS["pass_classes"]
        self.assertEqual(interpret_prediction(np.array([0.05, 0.95]), names, passes, 0.75)["status"], "PASS")
        self.assertEqual(interpret_prediction(np.array([0.92, 0.08]), names, passes, 0.75)["status"], "FAIL")
        r = interpret_prediction(np.array([0.40, 0.60]), names, passes, 0.75)
        self.assertEqual(r["status"], "REVIEW")  # low confidence is never auto-passed
        self.assertEqual(r["predicted_class"], "ok_front")
        self.assertAlmostEqual(r["confidence"], 0.6)

    def test_multiclass_names_come_from_labels(self):
        names = ["crack", "ok", "porosity"]
        r = interpret_prediction(np.array([0.1, 0.1, 0.8]), names, ["ok"], 0.5)
        self.assertEqual((r["predicted_class"], r["status"]), ("porosity", "FAIL"))
        self.assertEqual(set(r["probabilities"]), set(names))

    def test_probability_normalisation(self):
        np.testing.assert_allclose(to_probabilities(np.array([0.3, 0.7]), 2), [0.3, 0.7])
        p = to_probabilities(np.array([2.0, -1.0, 0.5]), 3)  # logits -> softmax
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_allclose(to_probabilities(np.array([0.8]), 2), [0.2, 0.8])  # sigmoid unit
        with self.assertRaises(InferenceError):
            to_probabilities(np.array([0.5, 0.3, 0.2]), 2)


class TestLabelsValidation(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_labels(LABELS)["pass_classes"], ["ok_front"])

    def test_invalid_variants(self):
        bad = [
            {**LABELS, "class_names": ["only_one"]},
            {**LABELS, "class_names": ["a", "a"]},
            {**LABELS, "pass_classes": ["not_a_class"]},
            {**LABELS, "pass_classes": []},
            {**LABELS, "input_mode": "rgb"},
            {**LABELS, "img_size": 5},
        ]
        for b in bad:
            with self.assertRaises(ValueError, msg=str(b)):
                validate_labels(b)


class TestEngine(unittest.TestCase):
    def test_missing_files_report_unavailable_never_fabricate(self):
        with tempfile.TemporaryDirectory() as d:
            e = DefectInferenceEngine(Path(d) / "m.keras", Path(d) / "labels.json")
            e.load()
            self.assertFalse(e.ready)
            self.assertIn("labels file not found", e.error)
            with self.assertRaises(ModelUnavailableError):
                e.predict(D, N)

    def test_labels_present_model_missing(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "labels.json").write_text(json.dumps(LABELS))
            e = DefectInferenceEngine(Path(d) / "m.keras", Path(d) / "labels.json")
            e.load()
            self.assertFalse(e.ready)
            self.assertIn("model not found", e.error)

    def test_invalid_labels_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "labels.json").write_text("{not json")
            e = DefectInferenceEngine(Path(d) / "m.keras", Path(d) / "labels.json")
            e.load()
            self.assertFalse(e.ready)
            self.assertIn("invalid labels", e.error)

    def test_attached_model_predict_contract(self):
        e = DefectInferenceEngine(Path("x"), Path("y"), threshold=0.75)
        seen = {}

        def fake(batch):
            seen["shape"], seen["max"] = batch.shape, float(batch.max())
            return np.array([0.1, 0.9])

        e.attach(fake, LABELS)
        r = e.predict(D, N)
        self.assertEqual(seen["shape"], (1, 224, 224, 3))
        self.assertEqual(seen["max"], 128.0)  # 0..255 float input, no hidden rescale
        self.assertEqual((r["status"], r["predicted_class"]), ("PASS", "ok_front"))
        self.assertEqual(r["model_name"], "stub")
        self.assertGreaterEqual(r["inference_ms"], 0)
        self.assertTrue(e.info()["ready"])

    def test_model_exception_becomes_inference_error(self):
        e = DefectInferenceEngine(Path("x"), Path("y"))

        def boom(batch):
            raise RuntimeError("GPU exploded")

        e.attach(boom, LABELS)
        with self.assertRaises(InferenceError):
            e.predict(D, N)

    def test_wrong_output_size_is_an_error(self):
        e = DefectInferenceEngine(Path("x"), Path("y"))
        e.attach(lambda b: np.array([0.2, 0.3, 0.5]), LABELS)
        with self.assertRaises(InferenceError):
            e.predict(D, N)


if __name__ == "__main__":
    unittest.main()
