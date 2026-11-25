"""Input check: "does this image look like what the model was trained on?"

A softmax classifier must answer with one of its classes, so a selfie can come back as
``ok_front`` with 99 % "confidence". This module adds a separate, explicit check:

* **Feature distance** - squared Mahalanobis distance of the image's EfficientNet feature vector (1280-d) from
  the Gaussian fitted to the TRAINING images' features. The limit is calibrated on held-out in-distribution
  images (``ml/fit_ood.py``), so a known fraction of genuine images is sent for manual handling.
* **Colour check** - the training photos are grayscale; a clearly coloured image is a mismatch.

If either fires the app does not classify the image (status ``UNCLASSIFIED`` + explanation).
This is a statistical safeguard, not a guarantee: images that merely resemble the training photos
(e.g. a different metal part under similar lighting) can still pass. Validate it on your own negatives.

Pure NumPy except ``build_feature_model`` (TensorFlow, imported lazily).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def color_saturation(rgb: np.ndarray) -> float:
    """Mean absolute difference between colour channels (0 for grayscale; 0..255 scale)."""
    a = rgb.astype(np.int16)
    return float(((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])) / 2.0).mean())


def fit_gaussian(features: np.ndarray, shrinkage: float = 0.1):
    """Mean and (shrunk) precision matrix of ``features`` (N, D). Shrinkage keeps the covariance invertible."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 10:
        raise ValueError("need at least 10 feature vectors of shape (N, D)")
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(len(x) - 1, 1)
    d = cov.shape[0]
    cov = (1.0 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)
    return mean, precision


def squared_mahalanobis(feats: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    d = np.asarray(feats, dtype=np.float64) - mean
    return np.einsum("...i,ij,...j->...", d, precision, d)


@dataclass
class OODDetector:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def score(self, feats: np.ndarray) -> float:
        return float(squared_mahalanobis(np.asarray(feats).reshape(-1), self.mean, self.precision))

    def check(self, feats: np.ndarray, rgb: Optional[np.ndarray] = None, color_check: bool = True) -> Dict[str, Any]:
        score = self.score(feats)
        reasons, parts = [], []
        if score > self.threshold:
            reasons.append("unusual_appearance")
            parts.append(f"its features are {score / self.threshold:.1f}x beyond the limit seen in the training photos")
        sat = None
        if rgb is not None:
            sat = color_saturation(rgb)
            if color_check and self.meta.get("grayscale_training") and sat > float(self.meta.get("color_tolerance", 8.0)):
                reasons.append("color_image")
                parts.append("it is a colour image but the model was trained on grayscale photos")
        flagged = bool(reasons)
        message = ("This image does not look like the casting photos the model was trained on (" + "; ".join(parts)
                   + "). It was not classified.") if flagged else None
        return {"score": round(score, 2), "threshold": round(self.threshold, 2),
                "ratio": round(score / self.threshold, 3), "flagged": flagged, "reasons": reasons,
                "color_saturation": None if sat is None else round(sat, 2), "message": message}

    # ---- persistence (no pickle) -------------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(path, mean=self.mean.astype(np.float32), precision=self.precision.astype(np.float32),
                            threshold=np.float64(self.threshold), meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path) -> "OODDetector":
        with np.load(path, allow_pickle=False) as z:
            return cls(mean=z["mean"].astype(np.float64), precision=z["precision"].astype(np.float64),
                       threshold=float(z["threshold"]), meta=json.loads(str(z["meta"])))


def build_feature_model(model):
    """Keras model returning ``[features(1280), probs]`` in ONE forward pass.

    Works for models trained with ``ml/train_model.py`` (augmentation -> EfficientNet base with global pooling
    -> dropout -> Dense). The base and the final Dense layer are re-applied to a fresh input, sharing the
    trained weights, so probabilities equal the original model's (augmentation/dropout are inactive at inference).
    """
    import tensorflow as tf

    base = next((l for l in model.layers if l.name.lower().startswith("efficientnet")), None)
    head = model.layers[-1]
    if base is None or not isinstance(head, tf.keras.layers.Dense):
        raise ValueError("model architecture not recognised (expected an EfficientNet base followed by a Dense head)")
    inp = tf.keras.Input(shape=tuple(model.input_shape[1:]))
    feats = base(inp, training=False)
    return tf.keras.Model(inp, [feats, head(feats)])
