"""Fit the input check ("is this a casting photo?") for an installed image-mode model.

    python -m ml.fit_ood --data-dir <casting_data> [--ood-dir <folder of NON-casting images>]

No retraining: it loads models/defect_model.keras, extracts the EfficientNet features of the training photos
(same loading/resizing as the live app), fits a Gaussian to them, and calibrates the distance limit on held-out
in-distribution images (validation + test) so that about (100 - percentile) % of genuine photos are flagged.
It writes models/ood.npz (+ ood_report.md) and prints how well non-casting images are rejected.

``--ood-dir`` should contain images that are NOT castings: selfies, random photos, other parts, screenshots...
They are used ONLY to report the rejection rate, never to fit anything. Synthetic negatives (noise, flat images,
gradients) are always tested too. Re-run after every re-training (the file is tied to one model).
Needs TensorFlow. Takes a few minutes on CPU (use --max-fit to trade accuracy for speed).
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.services.image_io import load_image
from backend.services.model_inference import _fingerprint, validate_labels
from backend.services.ood import (OODDetector, build_feature_model, color_saturation, fit_gaussian,
                                  squared_mahalanobis)
from ml.train_model import build_splits

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def synthetic_negatives(n_each=10, size=300, seed=0):
    rng = np.random.default_rng(seed)
    out = {"color noise": [], "gray noise": [], "flat image": [], "gradient": [], "dark gradient (color)": []}
    for i in range(n_each):
        out["color noise"].append(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        g = rng.integers(0, 255, (size, size), dtype=np.uint8)
        out["gray noise"].append(np.stack([g] * 3, -1))
        out["flat image"].append(np.full((size, size, 3), int(rng.integers(0, 255)), np.uint8))
        ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
        if i % 2:
            ramp = ramp.T
        out["gradient"].append(np.stack([ramp] * 3, -1))
        c = np.stack([ramp, ramp // 2, 255 - ramp], -1).astype(np.uint8)
        out["dark gradient (color)"].append(c)
    return out


def read_images(folder: Path, limit=200):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)[:limit]
    imgs = []
    for f in files:
        try:
            imgs.append(load_image(f.read_bytes(), f.name, settings.MAX_IMAGE_PIXELS).rgb)
        except Exception as exc:  # unreadable negatives are skipped, not fatal
            print(f"  skipped {f.name}: {exc}")
    return imgs


def prep(rgb, size):
    return cv2.resize(rgb.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def extract(dual, rgbs, size, batch=32):
    feats = []
    for i in range(0, len(rgbs), batch):
        x = np.stack([prep(r, size) for r in rgbs[i:i + batch]])
        f, _ = dual.predict(x, verbose=0)
        feats.append(np.asarray(f))
    return np.concatenate(feats) if feats else np.zeros((0, 1))


def load_items(items, limit=0, seed=0):
    items = list(items)
    if limit and len(items) > limit:
        items = random.Random(seed).sample(items, limit)
    rgbs = []
    for p, _ in items:
        rgbs.append(load_image(Path(p).read_bytes(), Path(p).name, settings.MAX_IMAGE_PIXELS).rgb)
    return rgbs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="the dataset folder (train/ and test/) used for training")
    ap.add_argument("--ood-dir", default=None, help="folder of NON-casting images used to report rejection rates")
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="distance limit = this percentile of held-out genuine photos (default 99.5 -> ~0.5%% flagged)")
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--max-fit", type=int, default=2500, help="training images used to fit (0 = all)")
    ap.add_argument("--out", default=str(settings.ood_path))
    args = ap.parse_args(argv)

    import tensorflow as tf

    meta = validate_labels(json.loads(settings.labels_path.read_text(encoding="utf-8")))
    if meta["input_mode"] != "image":
        raise SystemExit("The input check is only needed for image-mode (photo) models.")
    size = meta["img_size"]
    print("Loading model ...")
    model = tf.keras.models.load_model(str(settings.model_path), compile=False)
    version = _fingerprint(settings.model_path)
    dual = build_feature_model(model)

    _, parts, layout = build_splits(Path(args.data_dir), seed=42, drop_duplicates=True)
    fit_rgb = load_items(parts["train"], args.max_fit)
    cal_rgb = load_items(parts["val"])                              # validation photos: calibrate the limit
    test_rgb = load_items(parts["test"])                            # held-out test photos: report false-flag rate
    print(f"Images: fit {len(fit_rgb)}, calibration {len(cal_rgb)}, held-out test {len(test_rgb)}")

    # Safety check: the feature model must reproduce the trained model's probabilities
    probe = np.stack([prep(r, size) for r in fit_rgb[:16]])
    ref = model.predict(probe, verbose=0)
    _, alt = dual.predict(probe, verbose=0)
    diff = float(np.abs(ref - np.asarray(alt)).max())
    print(f"Feature-model self-check: max probability difference {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("The feature model does not reproduce the trained model; refusing to write ood.npz.")

    print("Extracting features ...")
    f_fit = extract(dual, fit_rgb, size)
    f_cal = extract(dual, cal_rgb, size)
    f_test = extract(dual, test_rgb, size)
    mean, precision = fit_gaussian(f_fit, args.shrinkage)
    s_cal = squared_mahalanobis(f_cal, mean, precision)
    s_test = squared_mahalanobis(f_test, mean, precision)
    threshold = float(np.percentile(s_cal, args.percentile))

    sat = np.array([color_saturation(r) for r in fit_rgb])
    p99 = float(np.percentile(sat, 99))
    gray_training = p99 < 3.0
    det_meta = {"percentile": args.percentile, "shrinkage": args.shrinkage, "model_version": version,
                "grayscale_training": bool(gray_training), "color_tolerance": float(max(8.0, 3 * p99)),
                "n_fit": len(fit_rgb), "n_calibration": len(cal_rgb), "feature_dim": int(mean.shape[0]),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    det = OODDetector(mean=mean, precision=precision, threshold=threshold, meta=det_meta)

    def flag_rate(rgbs, feats):
        if len(rgbs) == 0:
            return None
        flagged = sum(det.check(f, r, True)["flagged"] for f, r in zip(feats, rgbs))
        by_dist = float(np.mean(squared_mahalanobis(feats, mean, precision) > threshold))
        by_color = float(np.mean([gray_training and color_saturation(r) > det_meta["color_tolerance"] for r in rgbs]))
        return {"n": len(rgbs), "flagged_pct": round(100 * flagged / len(rgbs), 1),
                "by_distance_pct": round(100 * by_dist, 1), "by_color_pct": round(100 * by_color, 1)}

    rows = {"Held-out TEST photos (should be ~0 %)": flag_rate(test_rgb, f_test)}
    for name, imgs in synthetic_negatives().items():
        rows[f"Synthetic: {name} (should be 100 %)"] = flag_rate(imgs, extract(dual, imgs, size))
    if args.ood_dir:
        negs = read_images(Path(args.ood_dir))
        print(f"Loaded {len(negs)} negative images from {args.ood_dir}")
        rows[f"YOUR negatives: {Path(args.ood_dir).name} (should be 100 %)"] = flag_rate(negs, extract(dual, negs, size))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    det.save(out)
    md = ["# Input-check report", "", f"Model fingerprint: `{version}` · layout: {layout} · fit images: {len(fit_rgb)} · "
          f"limit = {args.percentile}th percentile of {len(cal_rgb)} held-out photos (squared distance {threshold:.1f}) · "
          f"grayscale training data: {gray_training}", "",
          "| Image set | Images | Flagged (either check) % | by distance % | by colour % |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['flagged_pct']} | {v['by_distance_pct']} | {v['by_color_pct']} |"
           for k, v in rows.items() if v]
    md += ["", "Interpretation: the first row is the price of the check (genuine casting photos sent to manual handling); "
               "the other rows are how many non-casting images are rejected. Images that closely resemble the training "
               "photos (e.g. a different metal part under the same lighting) may still pass - test with your own negatives."]
    out.with_name("ood_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out}. Restart the backend; the dashboard banner about the input check should disappear.")


if __name__ == "__main__":
    sys.exit(main())
