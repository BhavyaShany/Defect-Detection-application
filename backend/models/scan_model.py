import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

STATUSES = ("PASS", "FAIL", "REVIEW", "UNCLASSIFIED")

LOCALIZATION_NOTE = (
    "The classifier performs whole-input classification only and does not output defect locations. "
    "(For 3D scans, the colour map in the viewer is an unsupervised geometric surface-variation "
    "analysis, not a defect-detection result.)"
)


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone support); serialised with a 'Z' suffix."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso_utc(dt):
    return dt.isoformat(timespec="milliseconds") + "Z" if dt else None


def _loads(text, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


class ScanRecord(Base):
    __tablename__ = "scans"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    part_id = Column(String(64), nullable=False, index=True)
    part_type = Column(String(64), nullable=False, default="unspecified")
    timestamp = Column(DateTime, default=utcnow, index=True)

    # Inspection outcome. PASS / FAIL / REVIEW (low confidence) / UNCLASSIFIED (no model result).
    status = Column(String(16), nullable=False, index=True)
    defect_type = Column(String(64), nullable=True)  # predicted class name; NULL when UNCLASSIFIED
    confidence = Column(Float, nullable=True)  # NULL when UNCLASSIFIED -- never a placeholder
    latency_ms = Column(Float, nullable=False, default=0.0)  # server-side processing time

    # MES context
    batch_id = Column(String(64), nullable=False, default="UNSPECIFIED")
    shift = Column(String(32), nullable=False, default="UNSPECIFIED")
    operator_id = Column(String(32), nullable=False, default="UNSPECIFIED")

    # Source file
    filename = Column(String(256), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    file_sha256 = Column(String(64), nullable=True, index=True)

    # Model provenance
    model_name = Column(String(128), nullable=True)
    model_version = Column(String(32), nullable=True)

    # "scan3d" (point cloud / mesh) or "image" (2-D photograph)
    input_type = Column(String(16), nullable=False, default="scan3d")
    image_base64 = Column(Text, nullable=True)  # downscaled JPEG preview, image inputs only

    # Derived data
    depth_map_base64 = Column(Text, nullable=True)
    normal_map_base64 = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    timings_json = Column(Text, nullable=True)
    probabilities_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    has_geometry = Column(Boolean, nullable=False, default=False)

    # ---- serialisation -------------------------------------------------
    def to_summary(self) -> dict:
        """Lightweight row used by lists and WebSocket events (no images, no big JSON)."""
        return {
            "id": self.id,
            "part_id": self.part_id,
            "part_type": self.part_type,
            "timestamp": iso_utc(self.timestamp),
            "status": self.status,
            "defect_type": self.defect_type,
            "defect_class": self.defect_type,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "latency_ms": round(self.latency_ms or 0.0, 2),
            "batch_id": self.batch_id,
            "shift": self.shift,
            "operator_id": self.operator_id,
            "filename": self.filename,
            "file_path": self.filename,  # backward-compatible alias
            "is_defective": self.status == "FAIL",
            "model_name": self.model_name,
            "has_geometry": bool(self.has_geometry),
            "input_type": self.input_type or "scan3d",
        }

    def to_dict(self) -> dict:
        """Full detail record."""
        data = self.to_summary()
        data.update({
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
            "model_version": self.model_version,
            "image_base64": self.image_base64,
            "depth_map_base64": self.depth_map_base64,
            "normal_map_base64": self.normal_map_base64,
            "metrics": _loads(self.metrics_json, {}),
            "timings": _loads(self.timings_json, {}),
            "probabilities": _loads(self.probabilities_json, {}),
            "warnings": _loads(self.warnings_json, []),
            "localization": {"available": False, "reason": LOCALIZATION_NOTE},
        })
        return data
