"""Manufacturing Execution System (MES) integration and automated alert generation.

Two responsibilities, both driven by real inspection records only:

1. **Outbox delivery.** Every inspection (and every alert) becomes a canonical JSON event in
   ``mes_outbox`` inside the same request that stores the inspection. A background dispatcher
   POSTs them to ``MES_WEBHOOK_URL`` with exponential back-off; failures are retried and, after
   ``MES_MAX_ATTEMPTS``, marked ``DEAD`` with a ``MES_DELIVERY_FAILED`` alert. Because the event
   is committed before the HTTP call, a MES outage never loses a quality decision.

2. **Alert rule engine.** ``evaluate_scan`` applies the rules below to the current record and
   the stored history (no thresholds are invented: every number is a config value or measured).

   | code | severity | trigger |
   |---|---|---|
   | ``DEFECT_DETECTED`` | WARNING | inspection stored as ``FAIL`` (``ALERT_ON_EACH_DEFECT``) |
   | ``REVIEW_REQUIRED`` | INFO | inspection stored as ``REVIEW`` (confidence below threshold) |
   | ``CONSECUTIVE_DEFECTS`` | CRITICAL | the last ``ALERT_CONSECUTIVE_DEFECTS`` inspections are all ``FAIL`` |
   | ``DEFECT_RATE_SPIKE`` | CRITICAL | defect rate over the last ``ALERT_RATE_WINDOW_MINUTES`` min >= ``ALERT_RATE_THRESHOLD_PCT`` (at least ``ALERT_RATE_MIN_SAMPLES`` classified scans; de-duplicated) |
   | ``MODEL_UNAVAILABLE`` | WARNING | inspection stored as ``UNCLASSIFIED`` because no model was loaded (de-duplicated) |
   | ``MES_DELIVERY_FAILED`` | WARNING | an event exhausted its retries |

Events are pushed to the MES as ``inspection.completed`` and ``quality.alert``.
The MES is never contacted when ``MES_ENABLED`` is false or no endpoint is configured, and the
payload builder is a pure function so ``GET /api/mes/payload/{scan_id}`` can preview exactly
what would be sent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.mes_models import ALERT_SEVERITIES, DELIVERY_STATUSES, Alert, MesDelivery
from backend.models.scan_model import ScanRecord, iso_utc, utcnow

log = logging.getLogger("defect.mes")

CLASSIFIED = ("PASS", "FAIL", "REVIEW")

# Quality-control verdict vocabulary understood by MES / ERP systems.
VERDICT = {"PASS": "ACCEPT", "FAIL": "REJECT", "REVIEW": "HOLD", "UNCLASSIFIED": "NO_DECISION"}

_dispatch_lock = threading.Lock()  # one dispatcher pass at a time (worker + manual flush)
_sender = None  # type: ignore[var-annotated]  # overridable transport (tests)


# --------------------------------------------------------------------------- transport
def send_http(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
    """POST one JSON event. Returns ``(status_code, response_snippet)``; raises on failure."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200)), response.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # 4xx/5xx: the MES answered but rejected the event
        detail = exc.read(200).decode("utf-8", "replace").strip()
        raise RuntimeError(f"MES responded HTTP {exc.code}{(': ' + detail) if detail else ''}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach the MES endpoint: {exc.reason}") from exc
    except Exception as exc:  # timeouts, TLS errors, ...
        raise RuntimeError(f"MES request failed: {exc}") from exc


def redact_url(url: str) -> str:
    """Show scheme/host/path only -- never userinfo or query strings (may carry tokens)."""
    if not url:
        return ""
    try:
        parts = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc.split("@")[-1], parts.path, "", ""))
    except ValueError:
        return "(invalid URL)"


# ------------------------------------------------------------------------ event payload
def build_inspection_event(scan: ScanRecord, probabilities: Optional[dict] = None) -> Dict[str, Any]:
    """Canonical MES event for one inspection (pure function, no I/O)."""
    return {
        "event": "inspection.completed",
        "event_id": scan.id,
        "occurred_at": iso_utc(scan.timestamp),
        "source": {
            "system": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "site": settings.MES_SITE_ID or None,
            "line": settings.MES_LINE_ID or None,
            "station": settings.MES_STATION_ID or None,
        },
        "part": {
            "part_id": scan.part_id,
            "part_type": scan.part_type,
            "batch_id": scan.batch_id,
            "shift": scan.shift,
            "operator_id": scan.operator_id,
            "filename": scan.filename,
            "file_sha256": scan.file_sha256,
            "input_type": scan.input_type or "scan3d",
        },
        "result": {
            "status": scan.status,
            "verdict": VERDICT.get(scan.status, "NO_DECISION"),
            "predicted_class": scan.defect_type,
            "confidence": None if scan.confidence is None else round(float(scan.confidence), 6),
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "probabilities": probabilities,
            "model_name": scan.model_name,
            "model_version": scan.model_version,
            "localization": {"available": False},
        },
        "inspection": {
            "latency_ms": None if scan.latency_ms is None else round(float(scan.latency_ms), 2),
            "timings": _json_field(scan.timings_json),
            "metrics": _json_field(scan.metrics_json),
            "warnings": _json_field(scan.warnings_json, []),
        },
    }


def build_alert_event(alert: Alert) -> Dict[str, Any]:
    data = alert.to_dict()
    return {
        "event": "quality.alert",
        "event_id": alert.id,
        "occurred_at": data["created_at"],
        "source": {
            "system": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "site": settings.MES_SITE_ID or None,
            "line": settings.MES_LINE_ID or None,
            "station": settings.MES_STATION_ID or None,
        },
        "alert": {
            "code": alert.code,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "scan_id": alert.scan_id,
            "part_type": alert.part_type,
            "batch_id": alert.batch_id,
            "details": data["details"],
        },
    }


def _json_field(raw: Optional[str], default=None):
    if not raw:
        return {} if default is None else default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {} if default is None else default


# ------------------------------------------------------------------------- outbox queue
def enqueue(db: Session, scan: ScanRecord, payload: Optional[Dict[str, Any]] = None,
            event_type: str = "inspection.completed") -> Optional[MesDelivery]:
    """Queue one event for the MES. Returns ``None`` when the integration is switched off."""
    if not settings.mes_ready:
        return None
    payload = payload if payload is not None else build_inspection_event(scan)
    delivery = MesDelivery(
        id=str(uuid.uuid4()),
        scan_id=scan.id,
        event_type=event_type,
        status="PENDING",
        attempts=0,
        max_attempts=max(1, settings.MES_MAX_ATTEMPTS),
        next_attempt_at=utcnow(),
        payload_json=json.dumps(payload, default=str),
        created_at=utcnow(),
    )
    db.add(delivery)
    db.commit()
    db.refresh(delivery)
    return delivery


def enqueue_alert(db: Session, alert: Alert) -> Optional[MesDelivery]:
    if not settings.mes_ready:
        return None
    delivery = MesDelivery(
        id=str(uuid.uuid4()), scan_id=alert.scan_id, event_type="quality.alert", status="PENDING",
        attempts=0, max_attempts=max(1, settings.MES_MAX_ATTEMPTS), next_attempt_at=utcnow(),
        payload_json=json.dumps(build_alert_event(alert), default=str), created_at=utcnow(),
    )
    db.add(delivery)
    return delivery


# ------------------------------------------------------------------------ alert engine
def _create_alert(db: Session, scan: Optional[ScanRecord], code: str, severity: str, title: str,
                  message: str, details: Optional[dict] = None) -> Alert:
    if severity not in ALERT_SEVERITIES:
        severity = "WARNING"
    alert = Alert(
        id=str(uuid.uuid4()), scan_id=scan.id if scan else None, code=code, severity=severity,
        title=title[:160], message=message,
        part_type=scan.part_type if scan else None, batch_id=scan.batch_id if scan else None,
        details_json=json.dumps(details or {}, default=str), created_at=utcnow(), acknowledged=False,
    )
    db.add(alert)
    return alert


def _recent_same_code(db: Session, code: str, minutes: int) -> bool:
    """True if the same alert code fired within the de-duplication window (avoids alert storms)."""
    since = utcnow() - timedelta(minutes=max(1, minutes))
    return db.query(Alert.id).filter(Alert.code == code, Alert.created_at >= since).first() is not None


def evaluate_scan(db: Session, scan: ScanRecord) -> List[Alert]:
    """Apply the alert rules to a freshly stored inspection. Returns the alerts it created.

    The rules read the stored history, so an alert always reflects records the system actually
    holds; nothing is triggered by a simulated or assumed value.
    """
    alerts: List[Alert] = []

    if scan.status == "FAIL" and settings.ALERT_ON_EACH_DEFECT:
        alerts.append(_create_alert(
            db, scan, "DEFECT_DETECTED", "WARNING",
            f"Defect detected on {scan.part_id}",
            f"Part {scan.part_id} ({scan.part_type}, batch {scan.batch_id}) was rejected as "
            f"'{scan.defect_type}' with confidence "
            f"{(scan.confidence or 0) * 100:.1f}% (threshold {settings.CONFIDENCE_THRESHOLD * 100:.0f}%).",
            {"predicted_class": scan.defect_type, "confidence": scan.confidence,
             "batch_id": scan.batch_id, "shift": scan.shift},
        ))

    if scan.status == "REVIEW":
        # REVIEW has two causes: a low-confidence decision, or an image the input check flagged.
        # Say which one it was, so the message never claims a low confidence that isn't there.
        ood = (_json_field(scan.metrics_json) or {}).get("ood") or {}
        if ood.get("flagged"):
            reason = (f"the input check flagged this image ({ood.get('ratio')}x beyond the distance limit "
                      f"of the training photos)")
        else:
            reason = (f"confidence {(scan.confidence or 0) * 100:.1f}% is below the "
                      f"{settings.CONFIDENCE_THRESHOLD * 100:.0f}% threshold")
        alerts.append(_create_alert(
            db, scan, "REVIEW_REQUIRED", "INFO",
            f"Manual review required for {scan.part_id}",
            f"{reason[0].upper()}{reason[1:]}; the part was not auto-decided.",
            {"confidence": scan.confidence, "predicted_class": scan.defect_type,
             "input_check_flagged": bool(ood.get("flagged"))},
        ))

    streak_len = max(2, settings.ALERT_CONSECUTIVE_DEFECTS)
    recent = (db.query(ScanRecord).order_by(desc(ScanRecord.timestamp)).limit(streak_len).all())
    if len(recent) >= streak_len and all(r.status == "FAIL" for r in recent) and recent[0].id == scan.id:
        part_types = {r.part_type for r in recent}
        alerts.append(_create_alert(
            db, scan, "CONSECUTIVE_DEFECTS", "CRITICAL",
            f"{streak_len} consecutive rejected parts",
            f"The last {streak_len} inspections were all rejected"
            + (f" (part type '{part_types.pop()}')" if len(part_types) == 1 else "")
            + ". Stop-and-check the process before continuing.",
            {"streak": streak_len, "scan_ids": [r.id for r in recent],
             "batches": sorted({r.batch_id for r in recent})},
        ))

    window = max(1, settings.ALERT_RATE_WINDOW_MINUTES)
    since = utcnow() - timedelta(minutes=window)
    rows = (db.query(ScanRecord.status).filter(ScanRecord.timestamp >= since,
                                               ScanRecord.status.in_(CLASSIFIED)).all())
    classified = len(rows)
    failures = sum(1 for (s,) in rows if s == "FAIL")
    if classified >= max(1, settings.ALERT_RATE_MIN_SAMPLES):
        rate = failures / classified * 100.0
        if rate >= settings.ALERT_RATE_THRESHOLD_PCT and not _recent_same_code(db, "DEFECT_RATE_SPIKE",
                                                                             settings.ALERT_DEDUP_MINUTES):
            alerts.append(_create_alert(
                db, scan, "DEFECT_RATE_SPIKE", "CRITICAL",
                f"Defect rate {rate:.1f}% in the last {window} min",
                f"{failures} of {classified} classified inspections in the last {window} minutes "
                f"were rejected ({rate:.1f}%), above the {settings.ALERT_RATE_THRESHOLD_PCT:.0f}% alert level.",
                {"window_minutes": window, "classified": classified, "failed": failures,
                 "defect_rate_pct": round(rate, 2)},
            ))

    if scan.status == "UNCLASSIFIED" and not scan.model_name and not _recent_same_code(
            db, "MODEL_UNAVAILABLE", settings.ALERT_DEDUP_MINUTES):
        alerts.append(_create_alert(
            db, scan, "MODEL_UNAVAILABLE", "WARNING",
            "No model loaded: inspection not classified",
            "Scans are being stored as UNCLASSIFIED because no trained model is loaded. "
            "Quality decisions are not being made automatically.",
            {"scan_id": scan.id},
        ))

    return alerts


def on_inspection_stored(db: Session, scan: ScanRecord) -> Dict[str, Any]:
    """Hook called after an inspection is stored: queue the MES event and run the alert rules.

    Never raises -- a MES/alerting problem must not turn a successful inspection into an error.
    """
    result: Dict[str, Any] = {"alerts": [], "delivery_id": None, "errors": []}
    try:
        delivery = enqueue(db, scan, build_inspection_event(scan, _json_field(scan.probabilities_json) or None))
        if delivery is not None:
            result["delivery_id"] = delivery.id
        alerts = evaluate_scan(db, scan)
        for alert in alerts:
            enqueue_alert(db, alert)
        db.commit()
        result["alerts"] = [a.to_dict() for a in alerts]
    except Exception as exc:  # pragma: no cover - defensive
        db.rollback()
        log.exception("MES hook failed for scan %s", scan.id)
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    _publish_mes_audit(scan, result)
    return result


def _publish_mes_audit(scan: ScanRecord, result: Dict[str, Any]) -> None:
    """Push a quality-control line to ``/ws/mes``. Never fails the inspection."""
    try:
        from src.api.mes import hub

        hub.publish({
            "type": "qc_log",
            "scan_id": scan.id,
            "part_id": scan.part_id,
            "part_type": scan.part_type,
            "batch_id": scan.batch_id,
            "status": scan.status,
            "defect_type": scan.defect_type,
            "confidence": scan.confidence,
            "delivery_id": result.get("delivery_id"),
            "alerts": [
                {"code": item.get("code"), "severity": item.get("severity"), "title": item.get("title")}
                for item in (result.get("alerts") or [])
            ],
            "errors": result.get("errors") or [],
        })
        for item in result.get("alerts") or []:
            hub.publish({
                "type": "operator_alert",
                "scan_id": scan.id,
                "code": item.get("code"),
                "severity": item.get("severity"),
                "title": item.get("title"),
                "message": item.get("message"),
            })
    except Exception:
        log.exception("Could not publish MES audit event for scan %s", getattr(scan, "id", None))


# ---------------------------------------------------------------------- delivery worker
def _header_snapshot() -> Dict[str, str]:
    headers = dict(settings.mes_headers)
    return headers


def deliver(db: Session, delivery: MesDelivery, sender=None) -> Tuple[str, Optional[str]]:
    """Attempt one delivery. Returns ``(status, error)`` with status in RETRY/SENT/DEAD."""
    sender = sender or _sender or send_http
    payload = _json_field(delivery.payload_json)
    delivery.attempts = int(delivery.attempts or 0) + 1
    delivery.last_attempt_at = utcnow()
    try:
        code, snippet = sender(settings.MES_WEBHOOK_URL, payload, _header_snapshot(),
                               settings.MES_TIMEOUT_SECONDS)
        delivery.status = "SENT"
        delivery.sent_at = utcnow()
        delivery.last_status_code = code
        delivery.last_error = None
        db.commit()
        log.info("MES delivery %s (%s) accepted with HTTP %s", delivery.id, delivery.event_type, code)
        return "SENT", None
    except Exception as exc:
        message = str(exc)[:500]
        delivery.last_error = message
        code = None
        if "HTTP " in message:  # keep the MES status code for diagnostics
            try:
                code = int(message.split("HTTP ", 1)[1].split(":", 1)[0].strip())
            except (IndexError, ValueError):
                code = None
        delivery.last_status_code = code
        if delivery.attempts >= int(delivery.max_attempts or 1):
            delivery.status = "DEAD"
            delivery.next_attempt_at = utcnow()
            log.error("MES delivery %s (%s) abandoned after %d attempts: %s",
                      delivery.id, delivery.event_type, delivery.attempts, message)
            _dead_letter_alert(db, delivery)
        else:
            backoff = min(settings.MES_RETRY_BASE_SECONDS * (2 ** (delivery.attempts - 1)),
                          settings.MES_RETRY_MAX_SECONDS)
            delivery.status = "RETRY"
            delivery.next_attempt_at = utcnow() + timedelta(seconds=backoff)
            log.warning("MES delivery %s failed (attempt %d), retrying in %.0fs: %s",
                        delivery.id, delivery.attempts, backoff, message)
        db.commit()
        return delivery.status, message


def _dead_letter_alert(db: Session, delivery: MesDelivery) -> None:
    alert = _create_alert(
        db, None, "MES_DELIVERY_FAILED", "WARNING",
        "MES did not accept an inspection event",
        f"Event {delivery.id} ({delivery.event_type}, scan {delivery.scan_id or '-'}) could not be "
        f"delivered after {delivery.attempts} attempts: {delivery.last_error or 'unknown error'}",
        {"delivery_id": delivery.id, "event_type": delivery.event_type,
         "last_status_code": delivery.last_status_code},
    )
    db.add(alert)
    db.flush()


def dispatch_pending(db: Session, limit: Optional[int] = None, sender=None,
                     force: bool = False) -> Dict[str, Any]:
    """Deliver due events. ``force`` retries everything queued, ignoring the back-off clock."""
    if not settings.mes_ready:
        return {"enabled": False, "attempted": 0, "sent": 0, "retry": 0, "dead": 0,
                "reason": "MES integration is disabled or MES_WEBHOOK_URL is empty."}
    summary = {"enabled": True, "attempted": 0, "sent": 0, "retry": 0, "dead": 0, "reason": None}
    with _dispatch_lock:
        now = utcnow()
        query = db.query(MesDelivery).filter(MesDelivery.status.in_(("PENDING", "RETRY")))
        if not force:
            query = query.filter(MesDelivery.next_attempt_at <= now)
        rows = query.order_by(MesDelivery.next_attempt_at).limit(int(limit or settings.MES_BATCH_SIZE)).all()
        for delivery in rows:
            status, _ = deliver(db, delivery, sender=sender)
            summary["attempted"] += 1
            summary["sent" if status == "SENT" else "dead" if status == "DEAD" else "retry"] += 1
    return summary


def dispatch_once(limit: Optional[int] = None, sender=None) -> Dict[str, Any]:
    """Open a short-lived session and run one dispatch pass (used by the background worker)."""
    from backend.database.database import SessionLocal

    db = SessionLocal()
    try:
        return dispatch_pending(db, limit=limit, sender=sender)
    finally:
        db.close()


async def dispatch_loop() -> None:
    """Background MES dispatcher started by the app lifespan when the integration is enabled."""
    from fastapi.concurrency import run_in_threadpool

    log.info("MES dispatcher started (endpoint %s, interval %.1fs)",
             redact_url(settings.MES_WEBHOOK_URL), settings.MES_DISPATCH_INTERVAL_SECONDS)
    while True:
        try:
            summary = await run_in_threadpool(dispatch_once)
            if summary.get("attempted"):
                log.info("MES dispatch: %s", summary)
        except asyncio.CancelledError:
            log.info("MES dispatcher stopped")
            raise
        except Exception:
            log.exception("MES dispatch pass failed")
        await asyncio.sleep(max(1.0, settings.MES_DISPATCH_INTERVAL_SECONDS))


# ----------------------------------------------------------------------------- queries
def queue_counts(db: Session) -> Dict[str, int]:
    counts = {s: 0 for s in DELIVERY_STATUSES}
    for status, n in db.query(MesDelivery.status, func.count(MesDelivery.id)).group_by(MesDelivery.status):
        counts[status or "PENDING"] = n
    counts["total"] = sum(counts[s] for s in DELIVERY_STATUSES)
    return counts


def list_deliveries(db: Session, status: Optional[str] = None, limit: int = 50,
                    include_payload: bool = False) -> List[Dict[str, Any]]:
    query = db.query(MesDelivery)
    if status and status.upper() in DELIVERY_STATUSES:
        query = query.filter(MesDelivery.status == status.upper())
    rows = query.order_by(desc(MesDelivery.created_at)).limit(max(1, min(limit, 200))).all()
    return [r.to_dict(include_payload=include_payload) for r in rows]


def requeue(db: Session, delivery: MesDelivery) -> MesDelivery:
    """Put a RETRY/DEAD event back in the queue with a fresh attempt budget."""
    delivery.status = "PENDING"
    delivery.attempts = 0
    delivery.next_attempt_at = utcnow()
    delivery.last_error = None
    delivery.max_attempts = max(1, settings.MES_MAX_ATTEMPTS)
    db.commit()
    db.refresh(delivery)
    return delivery


def list_alerts(db: Session, severity: Optional[str] = None, acknowledged: Optional[bool] = None,
                limit: int = 50) -> List[Alert]:
    query = db.query(Alert)
    if severity and severity.upper() in ALERT_SEVERITIES:
        query = query.filter(Alert.severity == severity.upper())
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged.is_(acknowledged))
    return query.order_by(desc(Alert.created_at)).limit(max(1, min(limit, 200))).all()


def alert_summary(db: Session) -> Dict[str, Any]:
    rows = db.query(Alert.severity, Alert.acknowledged, func.count(Alert.id)).group_by(
        Alert.severity, Alert.acknowledged).all()
    out = {s: {"total": 0, "unacknowledged": 0} for s in ALERT_SEVERITIES}
    for severity, acknowledged, n in rows:
        key = severity if severity in out else "WARNING"
        out[key]["total"] += n
        if not acknowledged:
            out[key]["unacknowledged"] += n
    return {
        "by_severity": out,
        "total": sum(v["total"] for v in out.values()),
        "unacknowledged": sum(v["unacknowledged"] for v in out.values()),
    }


def acknowledge(db: Session, alert: Alert, by: Optional[str] = None) -> Alert:
    alert.acknowledged = True
    alert.acknowledged_at = utcnow()
    alert.acknowledged_by = (by or "dashboard")[:64]
    db.commit()
    db.refresh(alert)
    return alert


def status(db: Session) -> Dict[str, Any]:
    """Truthful MES integration state for /api/mes/status and the dashboard header."""
    return {
        "enabled": bool(settings.MES_ENABLED),
        "configured": settings.mes_ready,
        "endpoint": redact_url(settings.MES_WEBHOOK_URL),
        "auth_configured": bool(settings.MES_API_KEY.strip()),
        "target": {"site": settings.MES_SITE_ID or None, "line": settings.MES_LINE_ID or None,
                   "station": settings.MES_STATION_ID or None},
        "retry": {"max_attempts": settings.MES_MAX_ATTEMPTS,
                  "base_seconds": settings.MES_RETRY_BASE_SECONDS,
                  "max_seconds": settings.MES_RETRY_MAX_SECONDS},
        "dispatch_interval_seconds": settings.MES_DISPATCH_INTERVAL_SECONDS,
        "queue": queue_counts(db),
        "alerts": alert_summary(db),
        "rules": {
            "alert_on_each_defect": settings.ALERT_ON_EACH_DEFECT,
            "consecutive_defects": settings.ALERT_CONSECUTIVE_DEFECTS,
            "rate_window_minutes": settings.ALERT_RATE_WINDOW_MINUTES,
            "rate_min_samples": settings.ALERT_RATE_MIN_SAMPLES,
            "rate_threshold_pct": settings.ALERT_RATE_THRESHOLD_PCT,
            "dedup_minutes": settings.ALERT_DEDUP_MINUTES,
        },
    }
