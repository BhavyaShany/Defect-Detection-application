import pathlib
import tempfile
import unittest

import cv2
import numpy as np

from ml import build_depth_dataset as B
from ml import train_model as T
from tools.synthetic_parts import generate_part, write_ply_binary


def _make_images(root: pathlib.Path, spec, offset=0):
    """spec: {class: n}. Writes distinct small PNGs."""
    for cls, n in spec.items():
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((40, 40), (i + offset) % 250, np.uint8)
            img[0, 0] = (i * 7 + offset) % 255
            img[1, 1] = (i * 13 + sum(map(ord, cls))) % 255
            cv2.imwrite(str(root / cls / f"{cls}_{i + offset}.png"), img)


class TestDepthDatasetBuilder(unittest.TestCase):
    def test_skips_broken_files_and_writes_images(self):
        d = pathlib.Path(tempfile.mkdtemp())
        for cls, defect in (("ok", "none"), ("porosity", "porosity")):
            (d / "scans" / cls).mkdir(parents=True)
            for i in range(2):
                pts, _ = generate_part("engine_bracket", defect, n=12000, seed=i)
                write_ply_binary(d / "scans" / cls / f"p{i}.ply", pts)
        (d / "scans" / "ok" / "broken.ply").write_bytes(b"junk")
        self.assertEqual(B.main(["--scans", str(d / "scans"), "--out", str(d / "ds")]), 0)
        pngs = sorted((d / "ds").rglob("*.png"))
        self.assertEqual(len(pngs), 4)
        self.assertEqual(cv2.imread(str(pngs[0])).shape, (224, 224, 3))


class TestTrainingHelpers(unittest.TestCase):
    def test_metrics(self):
        m = T.metrics_from_confusion(np.array([[8, 2], [1, 9]]), ["a", "b"])
        self.assertAlmostEqual(m["accuracy"], 0.85)
        self.assertAlmostEqual(m["per_class"]["a"]["recall"], 0.8)

    def test_single_folder_layout_splits_without_overlap(self):
        root = pathlib.Path(tempfile.mkdtemp())
        _make_images(root, {"def_front": 40, "ok_front": 30})
        classes, parts, layout = T.build_splits(root, seed=1)
        self.assertEqual((layout, classes), ("single", ["def_front", "ok_front"]))
        allf = [f for k in parts for f, _ in parts[k]]
        self.assertEqual(len(allf), 70)
        self.assertEqual(len(allf), len(set(allf)))
        self.assertTrue(all(len(parts[k]) > 0 for k in parts))

    def test_official_kaggle_layout_with_wrapper_folder(self):
        root = pathlib.Path(tempfile.mkdtemp())
        inner = root / "casting_data" / "casting_data"
        _make_images(inner / "train", {"def_front": 30, "ok_front": 25})
        _make_images(inner / "test", {"def_front": 8, "ok_front": 6}, offset=500)
        classes, parts, layout = T.build_splits(root, seed=1)
        self.assertEqual(layout, "official")
        self.assertEqual(len(parts["test"]), 14)  # the official test folder, untouched
        self.assertTrue(all("test" in str(p) for p, _ in parts["test"]))
        self.assertTrue(all("train" in str(p) for p, _ in parts["train"] + parts["val"]))
        self.assertEqual(T.find_overlap(parts["train"] + parts["val"], parts["test"]), [])

    def test_class_mismatch_between_train_and_test_is_an_error(self):
        root = pathlib.Path(tempfile.mkdtemp())
        _make_images(root / "train", {"a": 20, "b": 20})
        _make_images(root / "test", {"a": 5, "c": 5})
        with self.assertRaises(SystemExit):
            T.build_splits(root, 1)

    def test_duplicate_leakage_is_detected(self):
        root = pathlib.Path(tempfile.mkdtemp())
        _make_images(root / "train", {"a": 20, "b": 20})
        _make_images(root / "test", {"a": 4, "b": 4}, offset=500)
        dup = next((root / "train" / "a").iterdir())
        (root / "test" / "a" / "copy.png").write_bytes(dup.read_bytes())
        _, parts, _ = T.build_splits(root, 1)
        self.assertEqual(len(T.find_overlap(parts["train"] + parts["val"], parts["test"])), 1)

    def test_drop_duplicates_cleans_leakage_conflicts_and_repeats(self):
        root = pathlib.Path(tempfile.mkdtemp())
        _make_images(root / "train", {"a": 30, "b": 30})
        _make_images(root / "test", {"a": 6, "b": 6}, offset=500)
        a0 = next(sorted((root / "train" / "a").iterdir()).__iter__())
        a1 = sorted((root / "train" / "a").iterdir())[1]
        (root / "test" / "a" / "leak.png").write_bytes(a0.read_bytes())          # leaked into test
        (root / "train" / "a" / "repeat.png").write_bytes(a1.read_bytes())       # repeated inside train
        b0 = sorted((root / "train" / "b").iterdir())[0]
        (root / "train" / "a" / "conflict.png").write_bytes(b0.read_bytes())     # same image, other label
        with self.assertRaises(SystemExit):                                       # default: refuse to proceed silently
            _, parts, _ = T.build_splits(root, 1)
            if T.find_overlap(parts["train"] + parts["val"], parts["test"]):
                raise SystemExit("leak")
        stats = {}
        _, parts, _ = T.build_splits(root, 1, drop_duplicates=True, stats=stats)
        self.assertEqual(stats["test_leaked_removed"], 1)
        self.assertEqual(stats["train_duplicates_removed"], 1)
        self.assertEqual(stats["label_conflicts_removed"], 2)  # conflict.png and the b original
        self.assertEqual(T.find_overlap(parts["train"] + parts["val"], parts["test"]), [])
        allf = [f for k in parts for f, _ in parts[k]]
        self.assertEqual(len(allf), len(set(allf)))

    def test_too_small_class_is_rejected(self):
        root = pathlib.Path(tempfile.mkdtemp())
        _make_images(root, {"a": 20, "b": 4})
        with self.assertRaises(SystemExit):
            T.build_splits(root, 1)

    def test_threshold_analysis_counts_missed_defects_and_review(self):
        # classes: 0 = defective, 1 = ok (pass).  4 samples
        probs = np.array([[0.1, 0.9],    # predicted ok, true defective, confident -> missed defect
                          [0.45, 0.55],  # low confidence (ok) -> review at t >= 0.6
                          [0.95, 0.05],  # predicted defective, true defective
                          [0.9, 0.1]])   # predicted defective, true ok -> false reject
        y = np.array([0, 1, 0, 1])
        rows = {r["threshold"]: r for r in T.threshold_analysis(probs, y, {1}, thresholds=(0.5, 0.6, 0.95))}
        self.assertEqual(rows[0.5]["missed_defects"], 1)
        self.assertEqual(rows[0.5]["false_rejects"], 1)
        self.assertEqual(rows[0.5]["review_pct"], 0.0)
        self.assertEqual(rows[0.6]["review_pct"], 25.0)
        self.assertEqual(rows[0.95]["missed_defects"], 0)
        self.assertEqual(T.recommend_threshold(list(rows.values())), 0.95)

    def test_acceptance_notes_flag_low_defect_recall(self):
        rep = {"test_images": 50, "per_class": {"def": {"recall": 0.9, "support": 20}, "ok": {"recall": 0.99, "support": 30}}}
        notes = T.acceptance_notes(rep, ["ok"])
        self.assertTrue(any("missed" in n for n in notes))
        self.assertTrue(any("Only 50 test images" in n for n in notes))


if __name__ == "__main__":
    unittest.main()


class TestVerifyModel(unittest.TestCase):
    def test_evaluate_runs_images_through_the_real_loader_and_engine(self):
        from pathlib import Path as P
        from backend.services.model_inference import DefectInferenceEngine
        from tools.verify_model import evaluate

        d = pathlib.Path(tempfile.mkdtemp())
        items = []
        rng = np.random.default_rng(0)
        for ci, (cls, base) in enumerate((("a", 30), ("b", 210))):
            (d / cls).mkdir()
            for i in range(6):
                img = np.clip(base + rng.integers(-10, 10, (60, 60)), 0, 255).astype(np.uint8)
                f = d / cls / f"{i}.png"
                cv2.imwrite(str(f), img)
                items.append((f, ci))
        # one deliberately mislabelled item so a "wrong" entry must be reported
        items.append((d / "a" / "0.png", 1))

        e = DefectInferenceEngine(P("x"), P("y"), threshold=0.75)
        e.attach(lambda b: np.array([0.05, 0.95]) if b.mean() > 128 else np.array([0.95, 0.05]),
                 {"class_names": ["a", "b"], "pass_classes": ["b"], "input_mode": "image",
                  "img_size": 64, "model_name": "stub"})
        res = evaluate(e, items, ["a", "b"])
        self.assertEqual(res["images"], 13)
        self.assertEqual(len(res["wrong"]), 1)
        self.assertAlmostEqual(res["accuracy"], 12 / 13)
        self.assertEqual(res["statuses"], {"PASS": 6, "FAIL": 7})
