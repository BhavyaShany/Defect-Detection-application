import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import crud
from backend.database.database import get_db
from backend.models.scan_model import utcnow
from backend.services.geometry import encode_png_data_url
from backend.services.image_io import image_summary, image_to_data_url, load_image
from backend.services.geometry_store import delete_geometry, load_geometry_payload, save_geometry
from backend.services.model_inference import InferenceError, ModelUnavailableError
from backend.services.preprocessing import run_preprocessing
from backend.services.runtime import inference_engine
from backend.services.scan_io import ScanError, load_scan
from backend.services.ws_manager import manager

router = APIRouter(prefix="/scans", tags=["Scans & Inspection"])
log = logging.getLogger("defect.scans")

# Bounds memory/CPU use when several scans are uploaded at once.
_slots = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_INSPECTIONS))

_HTTP_STATUS = {
    "UNSUPPORTED_FORMAT": 415,
    "EMPTY_FILE": 400,
    "FILE_TOO_LARGE": 413,
    "IMAGE_TOO_LARGE": 413,
    "CONFIG_ERROR": 500,
}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _clean(value: Optional[str], max_len: int, default: str) -> str:
    text = _CTRL.sub("", (value or "")).strip()[:max_len]
    return text or default


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks, size = [], 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise ScanError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit.", "FILE_TOO_LARGE")
        chunks.append(chunk)
    return b"".join(chunks)


def accepted_extensions() -> list:
    """3-D scan formats always; photo formats only while an image-mode model is loaded."""
    exts = list(settings.allowed_extensions)
    if inference_engine.input_kind == "image":
        exts += settings.image_extensions
    return exts


@router.post("/upload", status_code=201)
async def upload_scan(
    file: UploadFile = File(...),
    part_type: str = Form("unspecified"),
    batch_id: str = Form("UNSPECIFIED"),
    shift: str = Form("UNSPECIFIED"),
    operator_id: str = Form("UNSPECIFIED"),
    part_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a 3D scan (.ply, .pcd, .stl, .obj) or, with an image-mode model, a photograph
    (.jpg, .png, .bmp) and run the inspection pipeline.

    3D stages: validate -> parse -> preprocess -> classify -> store. Photo stages: validate -> decode -> classify -> store. Progress is broadcast on
    the WebSocket; the final inspection record is returned (HTTP 201).
    """
    scan_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    filename = _clean(file.filename, 200, "scan")
    timings: Dict[str, float] = {}

    async def progress(stage: str, message: str) -> None:
        await manager.broadcast({"type": "progress", "scan_id": scan_id, "stage": stage,
                                 "message": message, "filename": filename})

    async def fail(status: int, code: str, message: str) -> HTTPException:
        await manager.broadcast({"type": "failed", "scan_id": scan_id, "code": code,
                                 "message": message, "filename": filename})
        return _error(status, code, message)

    geometry_saved = False
    try:
        await progress("received", "Upload received")
        t = time.perf_counter()
        raw = await _read_limited(file, settings.max_upload_bytes)
        timings["read_ms"] = (time.perf_counter() - t) * 1000
        sha256 = hashlib.sha256(raw).hexdigest()

        ext = os.path.splitext(filename)[1].lower()
        if ext in settings.image_extensions:
            if inference_engine.input_kind != "image":
                raise ScanError(
                    "Photographs can only be inspected when an image-mode model is loaded; this server "
                    f"accepts: {', '.join(accepted_extensions())}.", "UNSUPPORTED_FORMAT")
            return await _inspect_image(db, scan_id, t_start, timings, raw, sha256, filename,
                                        part_type, batch_id, shift, operator_id, part_id, progress)

        async with _slots:
            await progress("parsing", "Validating and parsing scan")
            t = time.perf_counter()
            scan = await run_in_threadpool(load_scan, raw, filename, settings.MIN_POINTS, settings.allowed_extensions)
            timings["parse_ms"] = (time.perf_counter() - t) * 1000

            await progress("preprocessing", "Filtering noise, aligning, analysing surface")
            pre = await run_in_threadpool(run_preprocessing, scan)
            timings.update(pre.timings_ms)
            warnings = list(pre.warnings)

            classification: Optional[Dict[str, Any]] = None
            if inference_engine.ready and inference_engine.input_kind == "image":
                warnings.append("The loaded model classifies photographs, not 3D scans; "
                                "this scan was preprocessed but not classified.")
            elif inference_engine.ready:
                await progress("inference", "Running CNN inference")
                try:
                    classification = await run_in_threadpool(inference_engine.predict, pre.depth_map, pre.normal_map)
                    timings["inference_ms"] = classification["inference_ms"]
                except (InferenceError, ModelUnavailableError) as exc:
                    log.exception("Inference failed for %s", filename)
                    warnings.append(f"Model inference failed ({exc}); stored as UNCLASSIFIED.")
            else:
                warnings.append(
                    f"No trained model is loaded ({inference_engine.error}); "
                    "the scan was preprocessed and stored as UNCLASSIFIED."
                )

            await progress("saving", "Storing results")
            t = time.perf_counter()
            await run_in_threadpool(save_geometry, scan_id, pre.view_positions, pre.view_scores,
                                    pre.view_outlier, pre.view_faces)
            geometry_saved = True
            timings["geometry_save_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            depth_url = encode_png_data_url(pre.depth_map)
            normal_url = encode_png_data_url(pre.normal_map)
            timings["encode_ms"] = (time.perf_counter() - t) * 1000

        timings["server_total_ms"] = (time.perf_counter() - t_start) * 1000
        timings = {k: round(float(v), 2) for k, v in timings.items()}

        duplicate = await run_in_threadpool(crud.find_by_hash, db, sha256)
        if duplicate:
            warnings.append(f"This exact file was already inspected as {duplicate.part_id}.")

        now = utcnow()
        record = await run_in_threadpool(crud.create_scan, db, {
            "id": scan_id,
            "part_id": _clean(part_id, 64, f"SCAN-{now:%Y%m%d}-{scan_id[:6].upper()}"),
            "part_type": _clean(part_type, 64, "unspecified"),
            "timestamp": now,
            "status": classification["status"] if classification else "UNCLASSIFIED",
            "defect_type": classification["predicted_class"] if classification else None,
            "confidence": classification["confidence"] if classification else None,
            "latency_ms": timings["server_total_ms"],
            "batch_id": _clean(batch_id, 64, "UNSPECIFIED"),
            "shift": _clean(shift, 32, "UNSPECIFIED"),
            "operator_id": _clean(operator_id, 32, "UNSPECIFIED"),
            "filename": filename,
            "file_size_bytes": len(raw),
            "file_sha256": sha256,
            "model_name": classification["model_name"] if classification else None,
            "model_version": classification["model_version"] if classification else None,
            "depth_map_base64": depth_url,
            "normal_map_base64": normal_url,
            "metrics_json": json.dumps(pre.metrics),
            "timings_json": json.dumps(timings),
            "probabilities_json": json.dumps(classification["probabilities"]) if classification else None,
            "warnings_json": json.dumps(warnings),
            "has_geometry": True,
        })
        result = record.to_dict()
        result["duplicate_of"] = duplicate.id if duplicate else None
        await manager.broadcast({"type": "result", "scan_id": scan_id, "scan": record.to_summary()})
        return result

    except ScanError as exc:
        raise await fail(_HTTP_STATUS.get(exc.code, 422), exc.code, exc.message)
    except SQLAlchemyError:
        log.exception("Database error while storing scan %s", scan_id)
        if geometry_saved:
            await run_in_threadpool(delete_geometry, scan_id)
        raise await fail(503, "DATABASE_ERROR", "The inspection result could not be saved to the database.")
    except HTTPException:
        raise
    except Exception:
        log.exception("Unexpected failure while inspecting %s", filename)
        if geometry_saved:
            await run_in_threadpool(delete_geometry, scan_id)
        raise await fail(500, "INTERNAL_ERROR", "Unexpected server error while processing the scan.")


async def _inspect_image(db, scan_id, t_start, timings, raw, sha256, filename,
                         part_type, batch_id, shift, operator_id, part_id, progress):
    """Photo flow (image-mode model): decode -> classify -> store. Returns the response dict."""
    async with _slots:
        await progress("parsing", "Validating and decoding image")
        t = time.perf_counter()
        img = await run_in_threadpool(load_image, raw, filename, settings.MAX_IMAGE_PIXELS)
        timings["decode_ms"] = (time.perf_counter() - t) * 1000
        warnings = list(img.notes)

        await progress("inference", "Running CNN inference")
        classification: Optional[Dict[str, Any]] = None
        try:
            classification = await run_in_threadpool(inference_engine.predict_image, img.rgb)
            timings["inference_ms"] = classification["inference_ms"]
            ood = classification.get("ood")
            if ood and ood["flagged"]:
                warnings.append(ood["message"])
            elif ood is None and inference_engine.input_kind == "image":
                warnings.append("Input check is not active: a photo that is not a casting could still receive a confident class. "
                                "Run ml.fit_ood to enable it.")
        except (InferenceError, ModelUnavailableError) as exc:
            log.exception("Image inference failed for %s", filename)
            warnings.append(f"Model inference failed ({exc}); stored as UNCLASSIFIED.")

        await progress("saving", "Storing results")
        t = time.perf_counter()
        preview = await run_in_threadpool(image_to_data_url, img.rgb)
        timings["encode_ms"] = (time.perf_counter() - t) * 1000

    timings["server_total_ms"] = (time.perf_counter() - t_start) * 1000
    timings = {k: round(float(v), 2) for k, v in timings.items()}

    duplicate = await run_in_threadpool(crud.find_by_hash, db, sha256)
    if duplicate:
        warnings.append(f"This exact file was already inspected as {duplicate.part_id}.")

    now = utcnow()
    try:
        record = await run_in_threadpool(crud.create_scan, db, {
            "id": scan_id,
            "part_id": _clean(part_id, 64, f"IMG-{now:%Y%m%d}-{scan_id[:6].upper()}"),
            "part_type": _clean(part_type, 64, "unspecified"),
            "timestamp": now,
            "status": classification["status"] if classification else "UNCLASSIFIED",
            "defect_type": classification["predicted_class"] if classification else None,
            "confidence": classification["confidence"] if classification else None,
            "latency_ms": timings["server_total_ms"],
            "batch_id": _clean(batch_id, 64, "UNSPECIFIED"),
            "shift": _clean(shift, 32, "UNSPECIFIED"),
            "operator_id": _clean(operator_id, 32, "UNSPECIFIED"),
            "filename": filename,
            "file_size_bytes": len(raw),
            "file_sha256": sha256,
            "model_name": classification["model_name"] if classification else None,
            "model_version": classification["model_version"] if classification else None,
            "input_type": "image",
            "image_base64": preview,
            "metrics_json": json.dumps({**image_summary(img), "ood": classification.get("ood") if classification else None}),
            "timings_json": json.dumps(timings),
            "probabilities_json": json.dumps(classification["probabilities"]) if classification else None,
            "warnings_json": json.dumps(warnings),
            "has_geometry": False,
        })
    except SQLAlchemyError:
        log.exception("Database error while storing image scan %s", scan_id)
        raise HTTPException(status_code=503, detail={"code": "DATABASE_ERROR",
                            "message": "The inspection result could not be saved to the database."})
    result = record.to_dict()
    result["duplicate_of"] = duplicate.id if duplicate else None
    await manager.broadcast({"type": "result", "scan_id": scan_id, "scan": record.to_summary()})
    return result


@router.get("")
def list_scans(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None, description="PASS | FAIL | REVIEW | UNCLASSIFIED"),
    defect_type: Optional[str] = Query(None),
    part_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None, max_length=100),
    db: Session = Depends(get_db),
):
    """Paginated scan history (summary rows; no images)."""
    items = crud.get_scans(db, skip, limit, status, defect_type, part_type, search)
    total = crud.count_scans(db, status, defect_type, part_type, search)
    return {"total": total, "skip": skip, "limit": limit, "items": [i.to_summary() for i in items]}


@router.get("/{scan_id}")
def get_scan(scan_id: str, db: Session = Depends(get_db)):
    """Full inspection record including depth/normal maps, metrics and timings."""
    scan = crud.get_scan_by_id(db, scan_id)
    if not scan:
        raise _error(404, "SCAN_NOT_FOUND", "Scan not found.")
    return scan.to_dict()


@router.get("/{scan_id}/geometry")
async def get_scan_geometry(scan_id: str, db: Session = Depends(get_db)):
    """Processed geometry for the 3D viewer (typed arrays encoded as base64)."""
    scan = await run_in_threadpool(crud.get_scan_by_id, db, scan_id)
    if not scan:
        raise _error(404, "SCAN_NOT_FOUND", "Scan not found.")
    payload = await run_in_threadpool(load_geometry_payload, scan_id)
    if payload is None:
        raise _error(404, "GEOMETRY_NOT_FOUND", "No 3D geometry is stored for this scan.")
    return payload
