"""MES integration and alerting endpoints.

All values are read from the database: the outbox of events actually queued and the alerts the
rule engine actually created. Nothing here invents a delivery or an alert.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.models.mes_models import ALERT_SEVERITIES, DELIVERY_STATUSES, MesDelivery
from backend.services import mes

router = APIRouter(prefix="/mes", tags=["MES Integration & Alerts"])
alerts_router = APIRouter(prefix="/alerts", tags=["MES Integration & Alerts"])


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": code, "message": message})


# ------------------------------------------------------------------------------- MES
@router.get("/status")
def mes_status(db: Session = Depends(get_db)):
    """Integration state (no secrets), outbox queue depth, alert counts and the active rules."""
    return mes.status(db)


@router.get("/outbox")
def mes_outbox(
    status: Optional[str] = Query(None, description="PENDING | RETRY | SENT | DEAD"),
    limit: int = Query(50, ge=1, le=200),
    include_payload: bool = Query(False, description="Include the JSON body that was sent"),
    db: Session = Depends(get_db),
):
    """Queued / delivered MES events with their attempt counts and last error."""
    return {
        "status_allowed": list(DELIVERY_STATUSES),
        "items": mes.list_deliveries(db, status, limit, include_payload),
        "queue": mes.queue_counts(db),
    }


@router.post("/outbox/{delivery_id}/retry")
def mes_retry(delivery_id: str, db: Session = Depends(get_db)):
    """Re-queue a failed or abandoned event with a fresh attempt budget."""
    row = db.query(MesDelivery).filter(MesDelivery.id == delivery_id).first()
    if row is None:
        raise _not_found("DELIVERY_NOT_FOUND", "No such MES delivery.")
    return mes.requeue(db, row).to_dict()


@router.post("/dispatch")
def mes_dispatch(
    force: bool = Query(True, description="Ignore the back-off clock and retry everything queued"),
    db: Session = Depends(get_db),
):
    """Deliver queued events to the MES immediately instead of waiting for the background worker."""
    return mes.dispatch_pending(db, force=force)


@router.get("/payload/{scan_id}")
def mes_payload(scan_id: str, db: Session = Depends(get_db)):
    """Preview the exact JSON body an inspection is sent to the MES as (contract check)."""
    from backend.database import crud

    scan = crud.get_scan_by_id(db, scan_id)
    if scan is None:
        raise _not_found("SCAN_NOT_FOUND", "Scan not found.")
    return mes.build_inspection_event(scan, mes._json_field(scan.probabilities_json) or None)


# ---------------------------------------------------------------------------- alerts
@alerts_router.get("")
def list_alerts(
    severity: Optional[str] = Query(None, description="INFO | WARNING | CRITICAL"),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Generated quality alerts, newest first."""
    return {
        "severity_allowed": list(ALERT_SEVERITIES),
        "items": [a.to_dict() for a in mes.list_alerts(db, severity, acknowledged, limit)],
        "summary": mes.alert_summary(db),
    }


@alerts_router.get("/summary")
def alerts_summary(db: Session = Depends(get_db)):
    """Counts per severity and how many are still unacknowledged."""
    return mes.alert_summary(db)


@alerts_router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, by: Optional[str] = Query(None, max_length=64),
                      db: Session = Depends(get_db)):
    """Acknowledge an alert (records who and when)."""
    from backend.models.mes_models import Alert

    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise _not_found("ALERT_NOT_FOUND", "No such alert.")
    return mes.acknowledge(db, alert, by).to_dict()
