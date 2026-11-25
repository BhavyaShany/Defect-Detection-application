"""Database engine / session management.

The engine is created at application start-up (``init_db``), not at import time, so a
misconfigured database produces a clear start-up message instead of an import crash.
No sample data is ever inserted.
"""
import logging
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from backend.config import settings
from backend.models.scan_model import Base, ScanRecord

log = logging.getLogger("defect.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False)
engine: Optional[Engine] = None
active_url: str = ""
fallback_reason: Optional[str] = None


def _make_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    eng = create_engine(url, **kwargs)
    with eng.connect() as conn:  # fail fast if unreachable
        conn.execute(text("SELECT 1"))
    return eng


def init_db() -> None:
    global engine, active_url, fallback_reason
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.geometry_dir.mkdir(parents=True, exist_ok=True)

    url = settings.database_url
    try:
        engine = _make_engine(url)
        active_url, fallback_reason = url, None
    except Exception as exc:
        if not settings.DB_FALLBACK_SQLITE or url.startswith("sqlite"):
            raise RuntimeError(f"Cannot connect to database ({_redact(url)}): {exc}") from exc
        fallback_reason = f"{type(exc).__name__}: {exc}".splitlines()[0]
        log.error("Database %s unreachable (%s); falling back to SQLite.", _redact(url), fallback_reason)
        engine = _make_engine(settings.sqlite_fallback_url)
        active_url = settings.sqlite_fallback_url

    SessionLocal.configure(bind=engine)
    Base.metadata.create_all(bind=engine)
    _verify_schema()


def _verify_schema() -> None:
    """Fail clearly if an old (v2) database file is present instead of crashing mid-request."""
    cols = {c["name"] for c in inspect(engine).get_columns(ScanRecord.__tablename__)}
    missing = {c.name for c in ScanRecord.__table__.columns} - cols
    if missing:
        raise RuntimeError(
            "The existing 'scans' table was created by an older version and lacks columns "
            f"{sorted(missing)}. v2 databases contained simulated records and are not migrated: "
            "move/delete the old database file (see docs/CHANGELOG.md), then restart."
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_info() -> dict:
    dialect = engine.dialect.name if engine is not None else None
    return {"backend": dialect, "url": _redact(active_url), "fallback_reason": fallback_reason}


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url
