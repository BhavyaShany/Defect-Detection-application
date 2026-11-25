"""Data access and statistics.

Every statistic is derived from stored inspection records. Metrics with no underlying
data are returned as ``None`` (rendered as "—" in the UI), never as a placeholder number.
"""
import json
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from backend.models.scan_model import STATUSES, ScanRecord, iso_utc, utcnow

CLASSIFIED = ("PASS", "FAIL", "REVIEW")  # scans that received a model decision


def create_scan(db: Session, data: Dict[str, Any]) -> ScanRecord:
    scan = ScanRecord(**data)
    db.add(scan)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(scan)
    return scan


def get_scan_by_id(db: Session, scan_id: str) -> Optional[ScanRecord]:
    return db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()


def find_by_hash(db: Session, sha256: str, exclude_id: Optional[str] = None) -> Optional[ScanRecord]:
    q = db.query(ScanRecord).filter(ScanRecord.file_sha256 == sha256)
    if exclude_id:
        q = q.filter(ScanRecord.id != exclude_id)
    return q.order_by(desc(ScanRecord.timestamp)).first()


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filtered(db, entity, status, defect_type, part_type, search):
    q = db.query(entity)
    if status and status.upper() in STATUSES:
        q = q.filter(ScanRecord.status == status.upper())
    if defect_type and defect_type.lower() != "all":
        q = q.filter(ScanRecord.defect_type == defect_type)
    if part_type and part_type.lower() != "all":
        q = q.filter(ScanRecord.part_type == part_type)
    if search:
        like = f"%{_escape_like(search)}%"
        q = q.filter(or_(
            ScanRecord.part_id.ilike(like, escape="\\"),
            ScanRecord.batch_id.ilike(like, escape="\\"),
            ScanRecord.defect_type.ilike(like, escape="\\"),
            ScanRecord.filename.ilike(like, escape="\\"),
        ))
    return q


def get_scans(db: Session, skip: int = 0, limit: int = 50, status: Optional[str] = None,
              defect_type: Optional[str] = None, part_type: Optional[str] = None,
              search: Optional[str] = None) -> List[ScanRecord]:
    q = _filtered(db, ScanRecord, status, defect_type, part_type, search)
    return q.order_by(desc(ScanRecord.timestamp)).offset(skip).limit(limit).all()


def count_scans(db: Session, status: Optional[str] = None, defect_type: Optional[str] = None,
                part_type: Optional[str] = None, search: Optional[str] = None) -> int:
    q = _filtered(db, func.count(ScanRecord.id), status, defect_type, part_type, search)
    return q.scalar() or 0


def _pct(num: int, den: int) -> Optional[float]:
    return round(num / den * 100, 2) if den else None


def status_counts(db: Session) -> Dict[str, int]:
    rows = db.query(ScanRecord.status, func.count(ScanRecord.id)).group_by(ScanRecord.status).all()
    counts = {s: 0 for s in STATUSES}
    counts.update({s: c for s, c in rows})
    return counts


def get_summary_metrics(db: Session) -> Dict[str, Any]:
    c = status_counts(db)
    total = sum(c.values())
    classified = c["PASS"] + c["FAIL"] + c["REVIEW"]
    avg_latency = db.query(func.avg(ScanRecord.latency_ms)).scalar()
    avg_conf = db.query(func.avg(ScanRecord.confidence)).filter(ScanRecord.status.in_(CLASSIFIED)).scalar()
    return {
        "total_scanned": total,
        "classified": classified,
        "passed": c["PASS"],
        "failed": c["FAIL"],
        "needs_review": c["REVIEW"],
        "unclassified": c["UNCLASSIFIED"],
        # rates are over classified scans only (unclassified scans have no model decision)
        "pass_rate_pct": _pct(c["PASS"], classified),
        "defect_rate_pct": _pct(c["FAIL"], classified),
        "review_rate_pct": _pct(c["REVIEW"], classified),
        "avg_latency_ms": None if avg_latency is None else round(float(avg_latency), 2),
        "avg_confidence_pct": None if avg_conf is None else round(float(avg_conf) * 100, 1),
    }


def get_pareto_metrics(db: Session) -> List[Dict[str, Any]]:
    """Defect-category distribution (rejected scans only) with cumulative percentages."""
    rows = (
        db.query(ScanRecord.defect_type, func.count(ScanRecord.id).label("n"))
        .filter(ScanRecord.status == "FAIL", ScanRecord.defect_type.isnot(None))
        .group_by(ScanRecord.defect_type)
        .order_by(func.count(ScanRecord.id).desc())
        .all()
    )
    total = sum(r.n for r in rows)
    out, cum = [], 0
    for r in rows:
        cum += r.n
        out.append({
            "defect_type": r.defect_type,
            "count": r.n,
            "percentage": round(r.n / total * 100, 1),
            "cumulative_pct": round(cum / total * 100, 1),
        })
    return out


def _group_breakdown(db: Session, column, key: str) -> List[Dict[str, Any]]:
    rows = db.query(column, ScanRecord.status, func.count(ScanRecord.id)).group_by(column, ScanRecord.status).all()
    acc: Dict[str, Dict[str, int]] = {}
    for name, status, n in rows:
        acc.setdefault(name or "UNSPECIFIED", {s: 0 for s in STATUSES})[status] = n
    out = []
    for name, c in sorted(acc.items()):
        classified = c["PASS"] + c["FAIL"] + c["REVIEW"]
        out.append({
            key: name, "total": sum(c.values()), "passed": c["PASS"], "failed": c["FAIL"],
            "needs_review": c["REVIEW"], "unclassified": c["UNCLASSIFIED"],
            "defect_rate_pct": _pct(c["FAIL"], classified),
        })
    return out


def get_shift_metrics(db: Session) -> Dict[str, Any]:
    """Breakdown by the shift / part-type values that actually occur in the records."""
    return {
        "by_shift": _group_breakdown(db, ScanRecord.shift, "shift"),
        "by_part_type": _group_breakdown(db, ScanRecord.part_type, "part_type"),
    }


def get_activity(db: Session, hours: int = 24) -> List[Dict[str, Any]]:
    """Hourly inspection counts for the last ``hours`` hours (UTC buckets, including empty ones)."""
    now = utcnow()
    start = (now - timedelta(hours=hours - 1)).replace(minute=0, second=0, microsecond=0)
    rows = db.query(ScanRecord.timestamp, ScanRecord.status).filter(ScanRecord.timestamp >= start).all()
    buckets = {start + timedelta(hours=i): {"total": 0, "failed": 0} for i in range(hours)}
    for ts, status in rows:
        key = ts.replace(minute=0, second=0, microsecond=0)
        if key in buckets:
            buckets[key]["total"] += 1
            buckets[key]["failed"] += status == "FAIL"
    return [{"hour_start": iso_utc(k), **v} for k, v in sorted(buckets.items())]


def get_performance_metrics(db: Session, sample: int = 200) -> Dict[str, Any]:
    """Measured processing-time statistics over the most recent ``sample`` inspections."""
    rows = db.query(ScanRecord.timings_json).order_by(desc(ScanRecord.timestamp)).limit(sample).all()
    series: Dict[str, List[float]] = {}
    for (raw,) in rows:
        if not raw:
            continue
        try:
            for k, v in json.loads(raw).items():
                if isinstance(v, (int, float)):
                    series.setdefault(k, []).append(float(v))
        except ValueError:
            continue
    stats = {
        k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.percentile(v, 50)), 2),
            "p95": round(float(np.percentile(v, 95)), 2), "max": round(float(np.max(v)), 2), "n": len(v)}
        for k, v in series.items()
    }
    return {"sample_size": len(rows), "stages_ms": stats}
