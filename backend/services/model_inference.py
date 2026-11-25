"""CNN inference wrapper.

The engine runs the *actual trained model* and nothing else. If the model or its
``labels.json`` is missing or invalid the engine reports ``ready == False`` with a reason,
and the API stores scans as ``UNCLASSIFIED`` instead of inventing a class and confidence.
(The previous implementation fell back to ``random.uniform`` confidences and used the
simulator's injected defect label as the "prediction".)

Classes come from ``labels.json`` (written by ``ml/train_model.py``) -- never from a
hard-coded list.

Expected ``labels.json``::

    {
      "class_names": ["def_front", "ok_front"],     # index i <-> model output i
      "pass_classes": ["ok_front"],                 # classes that mean "not defective"
      "input_mode": "depth",                        # depth | normal | depth_normal | image
                                                    # ("image" = 2-D photographs, e.g. the Kaggle casting set)
      "img_size": 224,
      "model_name": "EfficientNetB0-casting-v1"
    }
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.services.geometry import build_model_input
from backend.services.ood import OODDetector, build_feature_model

VALID_INPUT_MODES = ("depth", "normal", "depth_normal", "image")


class ModelUnavailableError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


def validate_labels(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return a cleaned labels dict or raise ``ValueError`` with a precise reason."""
    names = meta.get("class_names")
    if not isinstance(names, list) or len(names) < 2 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("labels.json needs 'class_names': a list of at least two class-name strings.")
    if len(set(names)) != len(names):
        raise ValueError("labels.json 'class_names' contains duplicates.")
    passes = meta.get("pass_classes")
    if not isinstance(passes, list) or not passes or not set(passes) <= set(names):
        raise ValueError("labels.json needs 'pass_classes': a non-empty subset of 'class_names'.")
    mode = meta.get("input_mode", "depth")
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"labels.json 'input_mode' must be one of {VALID_INPUT_MODES}.")
    size = int(meta.get("img_size", 224))
    if not 32 <= size <= 1024:
        raise ValueError("labels.json 'img_size' must be between 32 and 1024.")
    return {
        "class_names": list(names),
        "pass_classes": list(passes),
        "input_mode": mode,
        "img_size": size,
        "model_name": str(meta.get("model_name", "unnamed-model")),
    }


def to_probabilities(raw: np.ndarray, n_classes: int) -> np.ndarray:
    """Normalise raw model output to a probability vector of length ``n_classes``."""
    out = np.asarray(raw, dtype=np.float64).reshape(-1)
    if out.size == 1 and n_classes == 2:  # single sigmoid unit -> [P(class0), P(class1)]
        p1 = float(out[0])
        if not 0.0 <= p1 <= 1.0:
            p1 = 1.0 / (1.0 + np.exp(-p1))
        return np.array([1.0 - p1, p1])
    if out.size != n_classes:
        raise InferenceError(f"Model produced {out.size} outputs but labels.json lists {n_classes} classes.")
    if np.all(out >= 0) and abs(out.sum() - 1.0) < 1e-3:
        return out
    e = np.exp(out - out.max())  # treat as logits
    return e / e.sum()


def interpret_prediction(probs: np.ndarray, class_names: List[str], pass_classes: List[str],
                         threshold: float) -> Dict[str, Any]:
    """Turn probabilities into a decision. Pure function (unit tested without TensorFlow)."""
    idx = int(np.argmax(probs))
    cls = class_names[idx]
    conf = float(probs[idx])
    if conf < threshold:
        status = "REVIEW"  # not confident enough to auto-accept or auto-reject
    else:
        status = "PASS" if cls in pass_classes else "FAIL"
    return {
        "predicted_class": cls,
        "confidence": round(conf, 6),
        "status": status,
        "probabilities": {n: round(float(p), 6) for n, p in zip(class_names, probs)},
    }


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(path)).encode())
                h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


class DefectInferenceEngine:
    def __init__(self, model_path: Path, labels_path: Path, threshold: float = 0.75,
                 ood_path: Optional[Path] = None, ood_color_check: bool = True):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = threshold
        self.ood_path = Path(ood_path) if ood_path else None
        self.ood_color_check = ood_color_check
        self.ood: Optional[OODDetector] = None
        self.ood_error: Optional[str] = None
        self.ready = False
        self.error: Optional[str] = "Model not loaded yet."
        self.meta: Dict[str, Any] = {}
        self.model_version: Optional[str] = None
        self.backend: Optional[str] = None
        self.warmup_ms: Optional[float] = None
        self._forward = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load model + labels. Never raises; inspect ``ready`` / ``error`` afterwards."""
        self.ready, self.error = False, None
        try:
            if not self.labels_path.exists():
                raise ModelUnavailableError(
                    f"labels file not found: {self.labels_path}. Train a model with ml/train_model.py "
                    "or copy its labels.json here."
                )
            try:
                self.meta = validate_labels(json.loads(self.labels_path.read_text(encoding="utf-8")))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ModelUnavailableError(f"invalid labels file: {exc}") from exc
            if not self.model_path.exists():
                raise ModelUnavailableError(f"model not found: {self.model_path}")
            try:
                import tensorflow as tf  # noqa: F401
            except ImportError as exc:
                raise ModelUnavailableError("TensorFlow is not installed (pip install tensorflow).") from exc

            model, backend = self._load_tf(self.model_path)
            self.backend = backend
            self._forward = self._make_forward(model)
            self.model_version = _fingerprint(self.model_path)
            self._load_ood(model)
            self._warm_up()
            self.ready = True
        except ModelUnavailableError as exc:
            self.error = str(exc)
        except Exception as exc:  # corrupt model file, TF/Keras incompatibility, ...
            self.error = f"model failed to load: {type(exc).__name__}: {exc}"

    @staticmethod
    def _load_tf(path: Path):
        import tensorflow as tf

        try:
            return tf.keras.models.load_model(str(path), compile=False), "keras"
        except Exception as first:
            if path.is_dir():  # Keras 3 cannot load a raw SavedModel; use it as an inference layer
                try:
                    import keras

                    return keras.layers.TFSMLayer(str(path), call_endpoint="serving_default"), "savedmodel-layer"
                except Exception:
                    pass
            raise first

    def _make_forward(self, model):
        def forward(batch: np.ndarray) -> np.ndarray:
            try:
                out = model(batch, training=False)
            except TypeError:  # layers/callables that do not accept `training`
                out = model(batch)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            return np.asarray(out)[0]

        return forward

    def _load_ood(self, model) -> None:
        """Enable the input check if ``ood.npz`` exists and matches this model; otherwise record why not."""
        self.ood, self.ood_error = None, None
        if not self.ood_path or not self.ood_path.exists():
            self.ood_error = "not set up (run: python -m ml.fit_ood)"
            return
        original = self._forward
        try:
            det = OODDetector.load(self.ood_path)
            if det.meta.get("model_version") != self.model_version:
                raise ValueError("it was fitted for a different model; re-run ml.fit_ood")
            dual = build_feature_model(model)

            def forward(batch: np.ndarray):
                feats, probs = dual(batch, training=False)
                return np.asarray(probs)[0], np.asarray(feats)[0]

            size = self.meta["img_size"]
            _, feats = forward(np.zeros((1, size, size, 3), dtype=np.float32))
            if feats.shape[-1] != det.dim:
                raise ValueError(f"feature size {feats.shape[-1]} != fitted size {det.dim}")
            self._forward, self.ood = forward, det
        except Exception as exc:
            self._forward = original
            self.ood_error = f"input check disabled: {exc}"

    def attach(self, forward_fn, labels: Dict[str, Any], version: str = "test",
               ood: Optional[OODDetector] = None) -> None:
        """Install a callable ``(batch NHWC float32) -> probs`` or ``-> (probs, features)`` (tests / alternative runtimes)."""
        self.ood = ood
        self.meta = validate_labels(labels)
        self._forward, self.model_version, self.backend = forward_fn, version, "callable"
        self.ready, self.error = True, None

    def _warm_up(self) -> None:
        size = self.meta["img_size"]
        t = time.perf_counter()
        self._forward(np.zeros((1, size, size, 3), dtype=np.float32))
        self.warmup_ms = round((time.perf_counter() - t) * 1000, 1)

    # --------------------------------------------------------------- inspection
    def info(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "error": self.error,
            "model_name": self.meta.get("model_name"),
            "model_version": self.model_version,
            "class_names": self.meta.get("class_names", []),
            "pass_classes": self.meta.get("pass_classes", []),
            "input_mode": self.meta.get("input_mode"),
            "confidence_threshold": self.threshold,
            "backend": self.backend,
            "warmup_ms": self.warmup_ms,
            "ood": {
                "enabled": self.ood is not None,
                "error": self.ood_error if self.ood is None else None,
                "threshold": None if self.ood is None else round(self.ood.threshold, 2),
                "percentile": None if self.ood is None else self.ood.meta.get("percentile"),
            },
        }

    @property
    def input_kind(self) -> Optional[str]:
        """"image" for photo models, "scan3d" for depth/normal models, None if not loaded."""
        if not self.ready:
            return None
        return "image" if self.meta.get("input_mode") == "image" else "scan3d"

    def predict_image(self, rgb: np.ndarray) -> Dict[str, Any]:
        """Classify an (H, W, 3) uint8 RGB photograph with an ``input_mode == "image"`` model."""
        if not self.ready or self._forward is None:
            raise ModelUnavailableError(self.error or "Model is not loaded.")
        if self.meta["input_mode"] != "image":
            raise ModelUnavailableError("The loaded model expects 3D-scan projections, not photographs.")
        size = self.meta["img_size"]
        # INTER_LINEAR matches tf.image.resize's default bilinear used at training time
        x = cv2.resize(rgb.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        result = self._run(x)
        feats = result.pop("_feats", None)
        result["ood"] = None
        if self.ood is not None and feats is not None:
            check = self.ood.check(feats, rgb, self.ood_color_check)
            result["ood"] = check
            if check["flagged"]:  # not a casting photo: do not present a class or a confidence
                result.update({"status": "UNCLASSIFIED", "predicted_class": None, "confidence": None,
                               "probabilities": {}})
        return result

    def predict(self, depth: np.ndarray, normal: np.ndarray) -> Dict[str, Any]:
        if not self.ready or self._forward is None:
            raise ModelUnavailableError(self.error or "Model is not loaded.")
        if self.meta["input_mode"] == "image":
            raise ModelUnavailableError("The loaded model expects photographs, not 3D-scan projections.")
        size = self.meta["img_size"]
        x = build_model_input(depth, normal, self.meta["input_mode"])
        if x.shape[:2] != (size, size):
            x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        result = self._run(x)
        result.pop("_feats", None)
        return result

    def _run(self, x: np.ndarray) -> Dict[str, Any]:
        t = time.perf_counter()
        try:
            with self._lock:
                out = self._forward(x[None, ...])
            raw, feats = out if isinstance(out, tuple) else (out, None)
            probs = to_probabilities(raw, len(self.meta["class_names"]))
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {type(exc).__name__}: {exc}") from exc
        elapsed = (time.perf_counter() - t) * 1000
        result = interpret_prediction(probs, self.meta["class_names"], self.meta["pass_classes"], self.threshold)
        result.update({
            "inference_ms": round(elapsed, 2),
            "model_name": self.meta["model_name"],
            "model_version": self.model_version,
            "input_mode": self.meta["input_mode"],
            "_feats": feats,
        })
        return result
