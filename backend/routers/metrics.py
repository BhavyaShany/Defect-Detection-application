import csv
import io
import json

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.database import get_db

router = APIRouter(prefix="/metrics", tags=["Analytics & MES Metrics"])


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    """KPI totals and rates computed from stored records (``null`` when there is no data)."""
    return crud.get_summary_metrics(db)


@router.get("/pareto")
def get_pareto(db: Session = Depends(get_db)):
    """Defect categories of rejected scans with counts and cumulative percentages."""
    return crud.get_pareto_metrics(db)


@router.get("/shifts")
def get_shifts(db: Session = Depends(get_db)):
    """Breakdown by the shift and part-type values present in the records."""
    return crud.get_shift_metrics(db)


@router.get("/activity")
def get_activity(hours: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)):
    """Hourly inspection counts (UTC) for recent-activity charts."""
    return crud.get_activity(db, hours)


@router.get("/performance")
def get_performance(db: Session = Depends(get_db)):
    """Measured per-stage processing times (mean / p50 / p95 / max) over recent inspections."""
    return crud.get_performance_metrics(db)


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection from user-supplied text."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


@router.get("/export")
def export_reports(format: str = Query("csv", pattern="^(csv|json)$"), db: Session = Depends(get_db)):
    """Download inspection logs (latest 1000) as CSV or JSON for MES integration."""
    scans = crud.get_scans(db, skip=0, limit=1000)
    if format == "json":
        return Response(
            content=json.dumps([s.to_summary() for s in scans], indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=cast_defect_inspection_report.json"},
        )
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Scan ID", "Part ID", "Part Type", "Timestamp (UTC)", "Status", "Predicted Class",
                "Confidence", "Processing (ms)", "Batch ID", "Shift", "Operator ID", "Filename", "Model"])
    for s in scans:
        sm = s.to_summary()
        w.writerow([
            s.id, _csv_safe(s.part_id), _csv_safe(s.part_type), sm["timestamp"], s.status,
            _csv_safe(s.defect_type), "" if s.confidence is None else f"{s.confidence:.4f}",
            f"{(s.latency_ms or 0):.2f}", _csv_safe(s.batch_id), _csv_safe(s.shift),
            _csv_safe(s.operator_id), _csv_safe(s.filename), _csv_safe(s.model_name),
        ])
    return Response(
        content=out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cast_defect_inspection_report.csv"},
    )
