"""Train the EfficientNetB0 transfer-learning classifier and export it with its labels.

Photographs (e.g. the Kaggle casting-product dataset):

    python -m ml.train_model --data-dir casting_data --pass-classes ok_front --input-mode image

Depth maps from 3D scans (see ml/build_depth_dataset.py):

    python -m ml.train_model --data-dir dataset --pass-classes ok --input-mode depth

Accepted layouts for ``--data-dir`` (class name = folder name):

  1. ``train/<class>/*`` + ``test/<class>/*``  (official split, e.g. Kaggle). The test folder is used
     ONLY for the final evaluation; a validation set is carved out of ``train``.
     A single wrapper folder (``casting_data/casting_data/train``) is detected automatically.
  2. ``<class>/*``  (one folder per class): split 70 / 15 / 15 (train / val / test), stratified, seeded.

Outputs in ``--out-dir`` (default ``models/``):

    defect_model.keras      trained model (expects 0..255 RGB; EfficientNet rescales internally)
    labels.json             class_names, pass_classes, input_mode, img_size, model_name
    evaluation_report.json  accuracy, per-class precision/recall/F1, confusion matrix, threshold analysis
    evaluation_report.md    the same, human readable

Add ``--smoke-test`` for a 1-minute run on a tiny subset to check your setup (the model is NOT usable).
Requires tensorflow (``pip install tensorflow``); everything else here is NumPy only.
"""
import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ------------------------------------------------------------------ data layout
def _class_dirs(d: Path):
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def _class_files(d: Path):
    return {c.name: sorted(f for f in c.iterdir() if f.suffix.lower() in IMG_EXT) for c in _class_dirs(d)}


def resolve_layout(data_dir: Path):
    """Return ("official", train_dir, test_dir) or ("single", data_dir, None).

    Looks for a ``train`` folder with a sibling ``test`` folder up to three levels down, so the
    nested layout produced by unzipping Kaggle (``casting_data/casting_data/train``) is found.
    """
    d = Path(data_dir)
    cands = [d] + sorted((p for p in d.rglob("*") if p.is_dir() and len(p.relative_to(d).parts) <= 2),
                         key=lambda p: len(p.parts))
    for cand in cands:
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return "official", cand / "train", cand / "test"
    return "single", d, None


def split_items(files, seed, val_frac, test_frac):
    """Stratified split of {class: [paths]} -> {"train","val","test"} lists of (path, class_index)."""
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for ci, (c, fs) in enumerate(sorted(files.items())):
        fs = fs[:]
        rng.shuffle(fs)
        n_test = max(1, int(len(fs) * test_frac)) if test_frac else 0
        n_val = max(1, int(len(fs) * val_frac))
        parts["test"] += [(f, ci) for f in fs[:n_test]]
        parts["val"] += [(f, ci) for f in fs[n_test:n_test + n_val]]
        parts["train"] += [(f, ci) for f in fs[n_test + n_val:]]
    for k in parts:
        rng.shuffle(parts[k])
    return parts


def dedupe_files(train_files, test_files):
    """Remove byte-identical images that would invalidate evaluation.

    * an image present under two different labels is ambiguous -> dropped everywhere;
    * duplicates inside train are reduced to one copy (so validation cannot share images with train);
    * test images that also occur in train are dropped from TEST (train keeps them); test duplicates keep one copy.
    Returns (train_files, test_files, stats).
    """
    cache = {}

    def h(p):
        if p not in cache:
            cache[p] = file_hash(p)
        return cache[p]

    first_cls, conflicts = {}, set()
    for c, fs in train_files.items():
        for f in fs:
            if first_cls.setdefault(h(f), c) != c:
                conflicts.add(h(f))
    stats = {"train_duplicates_removed": 0, "label_conflicts_removed": 0,
             "test_leaked_removed": 0, "test_duplicates_removed": 0}
    kept, new_train = set(), {}
    for c, fs in train_files.items():
        out = []
        for f in fs:
            x = h(f)
            if x in conflicts:
                stats["label_conflicts_removed"] += 1
            elif x in kept:
                stats["train_duplicates_removed"] += 1
            else:
                kept.add(x)
                out.append(f)
        new_train[c] = out
    seen, new_test = set(), {}
    for c, fs in (test_files or {}).items():
        out = []
        for f in fs:
            x = h(f)
            if x in kept or x in conflicts:
                stats["test_leaked_removed"] += 1
            elif x in seen:
                stats["test_duplicates_removed"] += 1
            else:
                seen.add(x)
                out.append(f)
        new_test[c] = out
    return new_train, new_test, stats


def build_splits(data_dir: Path, seed: int, drop_duplicates: bool = False, stats: dict = None):
    """-> (class_names, parts, layout). Class lists must match between train and test.

    With ``drop_duplicates`` exact-duplicate images are removed first (see ``dedupe_files``); the
    removal counts are written into ``stats`` if a dict is passed.
    """
    layout, train_dir, test_dir = resolve_layout(data_dir)
    train_files = _class_files(train_dir)
    classes = sorted(train_files)
    if len(classes) < 2:
        raise SystemExit(f"Need at least two class folders; found {classes} under {train_dir}.")
    for c, fs in train_files.items():
        if len(fs) < 10:
            raise SystemExit(f"Class '{c}' has only {len(fs)} images; need at least 10.")
    test_files = None
    if layout == "official":
        test_files = _class_files(test_dir)
        if sorted(test_files) != classes:
            raise SystemExit(f"train and test class folders differ: {classes} vs {sorted(test_files)}")
    if drop_duplicates:
        train_files, test_files, st = dedupe_files(train_files, test_files)
        if stats is not None:
            stats.update(st)
        for c, fs in train_files.items():
            if len(fs) < 10:
                raise SystemExit(f"Class '{c}' has only {len(fs)} images left after removing duplicates.")
    if layout == "official":
        parts = split_items(train_files, seed, 0.15, 0.0)
        parts["test"] = [(f, classes.index(c)) for c, fs in sorted(test_files.items()) for f in fs]
    else:
        parts = split_items(train_files, seed, 0.15, 0.15)
    return classes, parts, layout


def file_hash(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find_overlap(a_items, b_items):
    """Exact-duplicate files present in both lists (train/test leakage). Returns [(path_a, path_b)]."""
    seen = {}
    for p, _ in a_items:
        seen.setdefault(file_hash(p), p)
    return [(seen[h], p) for p, _ in b_items for h in [file_hash(p)] if h in seen]


# ------------------------------------------------------------------- evaluation
def metrics_from_confusion(cm: np.ndarray, names):
    out = {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)), "per_class": {}}
    f1s = []
    for i, n in enumerate(names):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        out["per_class"][n] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                               "support": int(cm[i, :].sum())}
    out["macro_f1"] = float(np.mean(f1s))
    return out


def threshold_analysis(probs, y_true, pass_idx, thresholds=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """What each CONFIDENCE_THRESHOLD would do on the test set.

    A prediction is auto-decided when its top probability >= t, otherwise it goes to manual REVIEW.
    ``missed_defects`` = auto-PASSed parts that are actually defective (the costly error);
    ``false_rejects``  = auto-FAILed parts that are actually good.
    """
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    pred = probs.argmax(1)
    conf = probs.max(1)
    is_pass_pred = np.isin(pred, list(pass_idx))
    is_pass_true = np.isin(y_true, list(pass_idx))
    rows = []
    for t in thresholds:
        auto = conf >= t
        n_auto = int(auto.sum())
        rows.append({
            "threshold": t,
            "auto_decided_pct": round(100 * n_auto / max(len(conf), 1), 2),
            "review_pct": round(100 * (1 - n_auto / max(len(conf), 1)), 2),
            "accuracy_on_auto_pct": round(100 * float((pred[auto] == y_true[auto]).mean()), 2) if n_auto else None,
            "missed_defects": int((auto & is_pass_pred & ~is_pass_true).sum()),
            "false_rejects": int((auto & ~is_pass_pred & is_pass_true).sum()),
        })
    return rows


def recommend_threshold(rows):
    """Smallest threshold with zero missed defects on the test set (indicative only), else None."""
    ok = [r["threshold"] for r in rows if r["missed_defects"] == 0 and r["auto_decided_pct"] > 0]
    return min(ok) if ok else None


def acceptance_notes(report, pass_classes):
    """Plain-language caveats printed with the report."""
    notes = []
    n = report["test_images"]
    if n < 200:
        notes.append(f"Only {n} test images: accuracy estimates are rough (a few images change them by >1 %).")
    for c, m in report["per_class"].items():
        if c not in pass_classes:
            if m["recall"] < 0.95:
                notes.append(f"Recall for defective class '{c}' is {m['recall']:.1%}: defects are being missed. "
                             "For quality control this is the number that matters most.")
            if m["support"] < 30:
                notes.append(f"Only {m['support']} test images of '{c}': too few for a reliable estimate.")
    if report.get("data_cleaning"):
        notes.append("Byte-identical duplicates were removed (" + ", ".join(
            f"{k}={v}" for k, v in report["data_cleaning"].items()) + "). Near-duplicates (rotated, re-saved or "
            "lightly edited copies of the same image) cannot be detected, so test accuracy may still be optimistic.")
    return notes


# --------------------------------------------------------------------- training
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--pass-classes", nargs="+", required=True, help="class name(s) meaning 'not defective'")
    ap.add_argument("--input-mode", required=True, choices=["image", "depth", "normal", "depth_normal"],
                    help="'image' for photographs; depth/normal/depth_normal for 3D-scan projections. "
                         "Stored in labels.json and used at inference.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--fine-tune-layers", type=int, default=30)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' (downloads once) or 'none'")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="remove byte-identical images (train/test leakage, repeated or conflicting files) "
                         "before splitting; counts are recorded in the report")
    ap.add_argument("--smoke-test", action="store_true", help="tiny quick run to verify the setup")
    args = ap.parse_args(argv)

    cleaning = {}
    classes, parts, layout = build_splits(Path(args.data_dir), args.seed, args.drop_duplicates, cleaning)
    if args.drop_duplicates:
        print("Duplicate cleaning:", cleaning)
    if not set(args.pass_classes) <= set(classes):
        raise SystemExit(f"--pass-classes must be among {classes}")
    print(f"Layout: {layout}. Classes: {classes}")
    print("Images:", {k: len(v) for k, v in parts.items()})
    print("Test class balance:", {c: sum(1 for _, l in parts['test'] if l == i) for i, c in enumerate(classes)})

    overlap = find_overlap(parts["train"] + parts["val"], parts["test"])
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} test images are byte-identical to training/validation images "
                         f"(e.g. {overlap[0][1]} == {overlap[0][0]}). Re-run with --drop-duplicates to remove them "
                         "(they are dropped from the test set); otherwise the test results would be meaningless.")
    if args.smoke_test:
        parts = {k: v[:48] for k, v in parts.items()}
        args.epochs, args.fine_tune_epochs = 1, 0
        print("SMOKE TEST: tiny subset, 1 epoch. The resulting model is NOT usable.")

    import tensorflow as tf  # imported late so the helpers above work without TensorFlow

    tf.keras.utils.set_random_seed(args.seed)
    size = args.img_size

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        # default bilinear resize == cv2.INTER_LINEAR used by the inference service (no train/serve skew)
        return tf.cast(tf.image.resize(img, (size, size)), tf.float32), label  # keep 0..255

    def ds(items, train):
        d = tf.data.Dataset.from_tensor_slices(([str(p) for p, _ in items], [l for _, l in items]))
        if train:
            d = d.shuffle(len(items), seed=args.seed)
        return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds, test_ds = ds(parts["train"], True), ds(parts["val"], False), ds(parts["test"], False)

    counts = np.array([sum(1 for _, l in parts["train"] if l == i) for i in range(len(classes))], dtype=float)
    class_weight = {i: float(counts.sum() / (len(classes) * max(c, 1))) for i, c in enumerate(counts)}
    print("Class weights:", {classes[i]: round(w, 3) for i, w in class_weight.items()})

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None if args.weights == "none" else args.weights,
        input_shape=(size, size, 3), pooling="avg")
    base.trainable = False
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")
    inp = tf.keras.Input((size, size, 3))
    x = aug(inp)  # active only during training
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
          tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, monitor="val_loss")]
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, callbacks=cb)
    history = {k: [float(v) for v in vs] for k, vs in h1.history.items()}

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen while fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                       class_weight=class_weight, callbacks=cb)
        history["fine_tune"] = {k: [float(v) for v in vs] for k, vs in h2.history.items()}

    # ---- held-out evaluation (test split only, never used for training/selection)
    probs = model.predict(test_ds, verbose=0)
    y_true = np.array([l for _, l in parts["test"]])
    y_pred = probs.argmax(1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    name = args.model_name or f"EfficientNetB0-{args.input_mode}"
    pass_idx = {classes.index(c) for c in args.pass_classes}
    rows = threshold_analysis(probs, y_true, pass_idx)
    report = {
        "model_name": name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke_test": bool(args.smoke_test), "input_mode": args.input_mode, "layout": layout,
        "classes": classes, "pass_classes": args.pass_classes, "test_images": int(len(y_true)),
        "images": {k: len(v) for k, v in parts.items()}, "confusion_matrix": cm.tolist(),
        **metrics_from_confusion(cm, classes), "threshold_analysis": rows,
        "indicative_threshold": recommend_threshold(rows),
        "class_balance_train": {c: int(n) for c, n in zip(classes, counts)},
        "data_cleaning": cleaning or None,
        "environment": {"tensorflow": tf.__version__, "python": platform.python_version(), "seed": args.seed,
                        "epochs": args.epochs, "fine_tune_epochs": args.fine_tune_epochs},
        "history": history,
    }
    report["notes"] = acceptance_notes(report, args.pass_classes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "defect_model.keras")
    (out_dir / "labels.json").write_text(json.dumps({
        "class_names": classes, "pass_classes": args.pass_classes, "input_mode": args.input_mode,
        "img_size": size, "model_name": name}, indent=2), encoding="utf-8")
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [f"# Evaluation report - {name}", ""]
    if args.smoke_test:
        md += ["> **SMOKE TEST - this model is not usable.**", ""]
    md += [f"Held-out test images: **{len(y_true)}** · Accuracy: **{report['accuracy']:.3f}** · "
           f"Macro-F1: **{report['macro_f1']:.3f}**", "",
           "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    md += [f"| {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
           for n, m in report["per_class"].items()]
    md += ["", "Confusion matrix (rows = true, columns = predicted):", "", "```",
           "classes: " + ", ".join(classes), str(cm), "```", "",
           "## Confidence-threshold analysis (test set)", "",
           "| Threshold | Auto-decided % | Sent to REVIEW % | Accuracy on auto-decided % | Missed defects | False rejects |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['threshold']} | {r['auto_decided_pct']} | {r['review_pct']} | {r['accuracy_on_auto_pct']} | "
           f"{r['missed_defects']} | {r['false_rejects']} |" for r in rows]
    md += ["", f"Indicative threshold (smallest with zero missed defects on this test set): "
               f"**{report['indicative_threshold']}** - treat as a starting point; confirm on fresh data."]
    if report["notes"]:
        md += ["", "## Cautions", ""] + [f"- {n}" for n in report["notes"]]
    (out_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved model + labels + report to {out_dir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
