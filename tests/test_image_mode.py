import io
import unittest

import numpy as np
from PIL import Image

from backend.services.image_io import image_to_data_url, load_image
from backend.services.model_inference import DefectInferenceEngine, ModelUnavailableError, validate_labels
from backend.services.scan_io import ScanError
from pathlib import Path

IMG_LABELS = {"class_names": ["def_front", "ok_front"], "pass_classes": ["ok_front"],
              "input_mode": "image", "img_size": 224, "model_name": "photo-stub"}


def _img_bytes(fmt="PNG", size=(300, 300), mode="L", noise=True):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size[::-1], dtype=np.uint8) if noise else np.full(size[::-1], 7, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").convert(mode).save(buf, format=fmt)
    return buf.getvalue()


class TestImageLoading(unittest.TestCase):
    def test_grayscale_photo_becomes_rgb(self):
        for fmt in ("PNG", "JPEG", "BMP"):
            d = load_image(_img_bytes(fmt), "a." + fmt.lower())
            self.assertEqual(d.rgb.shape, (300, 300, 3))
            self.assertEqual(d.mode, "L")
        d = load_image(_img_bytes("PNG"), "a.png")
        np.testing.assert_array_equal(d.rgb[..., 0], d.rgb[..., 2])

    def test_rejections(self):
        cases = [(b"", "EMPTY_FILE"), (b"not an image at all", "CORRUPT_SCAN"),
                 (_img_bytes("PNG")[:200], "CORRUPT_SCAN"), (_img_bytes("PNG", (10, 10)), "IMAGE_TOO_SMALL")]
        for raw, code in cases:
            with self.assertRaises(ScanError) as cm:
                load_image(raw, "x.png")
            self.assertEqual(cm.exception.code, code)
        with self.assertRaises(ScanError) as cm:
            load_image(_img_bytes("PNG"), "x.png", max_pixels=1000)
        self.assertEqual(cm.exception.code, "IMAGE_TOO_LARGE")

    def test_uniform_image_gets_a_warning_not_a_rejection(self):
        self.assertTrue(load_image(_img_bytes("PNG", noise=False), "x.png").notes)

    def test_preview_data_url(self):
        d = load_image(_img_bytes("PNG"), "a.png")
        self.assertTrue(image_to_data_url(d.rgb).startswith("data:image/jpeg;base64,"))


class TestImageEngine(unittest.TestCase):
    def test_labels_accept_image_mode(self):
        self.assertEqual(validate_labels(IMG_LABELS)["input_mode"], "image")

    def test_image_prediction_contract(self):
        e = DefectInferenceEngine(Path("x"), Path("y"), threshold=0.75)
        seen = {}

        def fake(batch):
            seen["shape"], seen["mean"] = batch.shape, float(batch.mean())
            return np.array([0.9, 0.1])

        e.attach(fake, IMG_LABELS)
        self.assertEqual(e.input_kind, "image")
        rgb = np.full((300, 300, 3), 100, np.uint8)
        r = e.predict_image(rgb)
        self.assertEqual(seen["shape"], (1, 224, 224, 3))
        self.assertAlmostEqual(seen["mean"], 100.0, places=3)  # 0..255, no rescale
        self.assertEqual((r["status"], r["predicted_class"], r["input_mode"]), ("FAIL", "def_front", "image"))

    def test_mode_mismatch_is_refused_both_ways(self):
        img = DefectInferenceEngine(Path("x"), Path("y"))
        img.attach(lambda b: np.array([0.5, 0.5]), IMG_LABELS)
        with self.assertRaises(ModelUnavailableError):
            img.predict(np.zeros((224, 224), np.uint8), np.zeros((224, 224, 3), np.uint8))
        depth = DefectInferenceEngine(Path("x"), Path("y"))
        depth.attach(lambda b: np.array([0.5, 0.5]), {**IMG_LABELS, "input_mode": "depth"})
        self.assertEqual(depth.input_kind, "scan3d")
        with self.assertRaises(ModelUnavailableError):
            depth.predict_image(np.zeros((300, 300, 3), np.uint8))

    def test_not_loaded_has_no_kind(self):
        self.assertIsNone(DefectInferenceEngine(Path("x"), Path("y")).input_kind)


if __name__ == "__main__":
    unittest.main()
