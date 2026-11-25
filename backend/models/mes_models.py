"""Tables backing MES integration (delivery outbox) and automated alerting.

``MesDelivery`` is a transactional outbox: every inspection event is written in the same
request that stores the inspection, so a MES outage can never lose a quality result. A
background dispatcher (or ``POST /api/mes/dispatch``) retries with exponential back-off.

``Alert`` is the output of the rule engine in ``services/mes.py`` (defect detected, defect
streak, defect-rate spike, manual review required, missing model, MES delivery failure).
"""
import json
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from backend.models.scan_model import Base, iso_utc, utcnow

DELIVERY_STATUSES = ("PENDING", "RETRY", "SENT", "DEAD")
ALERT_SEVERITIES = ("INFO", "WARNING", "CRITICAL")


def _loads(text, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


class MesDelivery(Base):
    """One outbound MES event and its delivery state."""

    __tablename__ = "mes_outbox"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id = Column(String(36), nullable=True, index=True)
    event_type = Column(String(48), nullable=False, default="inspection.completed")
    status = Column(String(12), nullable=False, default="PENDING", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    next_attempt_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    last_attempt_at = Column(DateTime, nullable=True)
    last_status_code = Column(Integer, nullable=True)  # HTTP status returned by the MES
    last_error = Column(String(512), nullable=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    sent_at = Column(DateTime, nullable=True)

    def to_dict(self, include_payload: bool = False) -> dict:
        data = {
            "id": self.id,
            "scan_id": self.scan_id,
            "event_type": self.event_type,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "next_attempt_at": iso_utc(self.next_attempt_at),
            "last_attempt_at": iso_utc(self.last_attempt_at),
            "last_status_code": self.last_status_code,
            "last_error": self.last_error,
            "created_at": iso_utc(self.created_at),
            "sent_at": iso_utc(self.sent_at),
        }
        if include_payload:
            data["payload"] = _loads(self.payload_json, {})
        return data


class Alert(Base):
    """A quality alert produced by the rule engine (never seeded, never simulated)."""

    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id = Column(String(36), nullable=True, index=True)
    code = Column(String(48), nullable=False, index=True)
    severity = Column(String(12), nullable=False, default="WARNING", index=True)
    title = Column(String(160), nullable=False)
    message = Column(Text, nullable=False)
    part_type = Column(String(64), nullable=True)
    batch_id = Column(String(64), nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(64), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "part_type": self.part_type,
            "batch_id": self.batch_id,
            "details": _loads(self.details_json, {}),
            "created_at": iso_utc(self.created_at),
            "acknowledged": bool(self.acknowledged),
            "acknowledged_at": iso_utc(self.acknowledged_at),
            "acknowledged_by": self.acknowledged_by,
        }

    def age_minutes(self, now: datetime = None) -> float:
        return ((now or utcnow()) - self.created_at).total_seconds() / 60.0
