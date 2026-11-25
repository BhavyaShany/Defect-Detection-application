"""Verify the installed model through the APP'S OWN code path and compare with the training report.

    python -m tools.verify_model --data-dir <path to casting_data>

Loads models/defect_model.keras + labels.json exactly as the server does, runs every image of the
(duplicate-cleaned) official test split through ``load_image`` -> ``predict_image``, and prints accuracy,
confusion matrix, PASS/FAIL/REVIEW counts at the configured CONFIDENCE_THRESHOLD, inference time, and a
comparison with models/evaluation_report.json. A few images may differ from the report (different JPEG
decoder/resize rounding); a larger gap means the app is not preprocessing images the way training did.
Needs TensorFlow (the model) and the dataset folder; uses the same duplicate cleaning as training.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from backend.config import settings
from backend.services.image_io import load_image
from backend.services.model_inference import DefectInferenceEngine
from ml.train_model import build_splits


def evaluate(engine, items, classes, max_pixels=40_000_000):
    """items: [(path, true_class_index)] -> result dict (uses the real engine API)."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    statuses, wrong, times, rejected = Counter(), [], [], []
    for path, ti in items:
        p = Path(path)
        img = load_image(p.read_bytes(), p.name, max_pixels)
        r = engine.predict_image(img.rgb)
        statuses[r["status"]] += 1
        times.append(r["inference_ms"])
        if r["predicted_class"] is None:  # input check refused to classify this image
            rejected.append((str(p), classes[ti], (r.get("ood") or {}).get("ratio")))
            continue
        pi = classes.index(r["predicted_class"])
        cm[ti, pi] += 1
        if pi != ti:
            wrong.append((str(p), classes[ti], r["predicted_class"], r["confidence"]))
    total = int(cm.sum())
    recalls = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
    return {"images": total, "rejected_by_input_check": rejected, "accuracy": float(np.trace(cm) / max(total, 1)), "recall": recalls,
            "confusion_matrix": cm.tolist(), "statuses": dict(statuses), "wrong": wrong,
            "inference_ms_mean": float(np.mean(times)) if times else None,
            "inference_ms_p95": float(np.percentile(times, 95)) if times else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="dataset folder containing train/ and test/")
    ap.add_argument("--report", default=str(settings.base_dir / "models" / "evaluation_report.json"))
    ap.add_argument("--limit", type=int, default=0, help="only check the first N test images (quick run)")
    ap.add_argument("--no-input-check", action="store_true",
                    help="ignore models/ood.npz (reproduce the training report exactly)")
    ap.add_argument("--tolerance", type=float, default=0.005, help="max accepted accuracy gap vs the report")
    args = ap.parse_args(argv)

    engine = DefectInferenceEngine(settings.model_path, settings.labels_path, settings.CONFIDENCE_THRESHOLD,
                                   ood_path=None if args.no_input_check else settings.ood_path,
                                   ood_color_check=settings.OOD_COLOR_CHECK)
    engine.load()
    if not engine.ready:
        print(f"MODEL NOT LOADED: {engine.error}")
        return 2
    if engine.input_kind != "image":
        print("The installed model is not an image-mode model; this check is for photo models.")
        return 2
    classes = engine.meta["class_names"]

    cleaning = {}
    ds_classes, parts, layout = build_splits(Path(args.data_dir), seed=42, drop_duplicates=True, stats=cleaning)
    if ds_classes != classes:
        print(f"Class mismatch: dataset {ds_classes} vs model {classes}")
        return 2
    items = parts["test"][: args.limit or None]
    ood = engine.info()["ood"]
    print("Input check:", "ACTIVE (threshold %s)" % ood["threshold"] if ood["enabled"] else f"off ({ood['error']})")
    print(f"Model: {engine.meta['model_name']} ({engine.model_version}) | threshold {engine.threshold} | "
          f"layout {layout} | test images {len(items)} | cleaning {cleaning}")

    res = evaluate(engine, items, classes, settings.MAX_IMAGE_PIXELS)
    print(f"\nAccuracy: {res['accuracy']:.4f}  ({res['images']} images)")
    for c, r in res["recall"].items():
        print(f"  recall {c}: {r:.4f}")
    print("Confusion matrix (rows = truth):", classes, "\n", np.array(res["confusion_matrix"]))
    print("Decisions at this threshold:", res["statuses"])
    rej = res["rejected_by_input_check"]
    if rej:
        print(f"Input check refused to classify {len(rej)} of {len(items)} genuine test images "
              f"({100 * len(rej) / len(items):.1f}%), e.g. {Path(rej[0][0]).name} (distance {rej[0][2]}x limit).")
    print(f"Inference time: mean {res['inference_ms_mean']:.1f} ms, p95 {res['inference_ms_p95']:.1f} ms (this machine)")
    if res["wrong"]:
        print("\nMisclassified (first 10):")
        for path, t, p, conf in res["wrong"][:10]:
            print(f"  {Path(path).name}: truth={t} predicted={p} conf={conf:.3f}")

    rp = Path(args.report)
    if rp.exists() and not args.limit:
        rep = json.loads(rp.read_text(encoding="utf-8"))
        gap = abs(res["accuracy"] - rep["accuracy"])
        print(f"\nTraining report accuracy: {rep['accuracy']:.4f} on {rep['test_images']} images | "
              f"app accuracy: {res['accuracy']:.4f} | gap {gap:.4f}")
        if res["images"] != rep["test_images"]:
            print("NOTE: image counts differ from the report (input-check rejections or a different dataset copy); "
                  "run with --no-input-check for an exact comparison.")
        if gap <= args.tolerance:
            print("RESULT: MATCH - the app reproduces the training evaluation.")
            return 0
        print("RESULT: MISMATCH - the app's results differ from training by more than the tolerance. "
              "Check image preprocessing, TensorFlow version and labels.json before relying on this model.")
        return 1
    print("\n(No report comparison: report missing or --limit used.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
