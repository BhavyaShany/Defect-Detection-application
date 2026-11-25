# API reference

Base URL `http://localhost:8000` · interactive docs at `/docs` · all JSON errors use
`{"detail": {"code": "<CODE>", "message": "<human text>"}}`.

## Error codes

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `EMPTY_FILE` | Uploaded file has zero bytes |
| 413 | `FILE_TOO_LARGE` | Larger than `MAX_UPLOAD_MB` |
| 415 | `UNSUPPORTED_FORMAT` | Extension not accepted (3D: `.ply .pcd .stl .obj`; photos `.jpg .jpeg .png .bmp` only while an image-mode model is loaded), or unsupported PCD compression |
| 413 | `IMAGE_TOO_LARGE` | Photo exceeds `MAX_IMAGE_PIXELS` |
| 422 | `IMAGE_TOO_SMALL` | Photo smaller than 32 px per side |
| 422 | `CORRUPT_SCAN` / `TOO_FEW_POINTS` / `DEGENERATE_SCAN` / `SPARSE_SCAN` | File unreadable or unusable |
| 404 | `SCAN_NOT_FOUND` / `GEOMETRY_NOT_FOUND` | Unknown id / no stored geometry |
| 503 | `DATABASE_ERROR` | Result could not be stored |
| 500 | `INTERNAL_ERROR` / `CONFIG_ERROR` | Unexpected failure / bad configuration |

(Plain FastAPI request-validation errors, e.g. bad query parameters, return the standard `422` list in `detail`.)

## `POST /api/scans/upload` → 201

Accepts a 3D scan, or a photograph when an image-mode model is loaded (`input_type` in the response tells which). `multipart/form-data`: `file` (required); optional `part_type`, `batch_id`, `shift`, `operator_id`, `part_id` (max 64 chars).
`/scan/upload` is kept as an alias of the v2 route.

```jsonc
{
  "id": "6b1c…",                       "part_id": "SCAN-20261103-6B1C0A",
  "status": "PASS|FAIL|REVIEW|UNCLASSIFIED",
  "defect_type": "defective",          // predicted class name; null if UNCLASSIFIED
  "defect_class": "defective",         // alias of defect_type (v2 compatibility)
  "confidence": 0.9312,                // null if UNCLASSIFIED
  "is_defective": true,                // status == "FAIL"
  "latency_ms": 412.7,                 // server-side processing total
  "timestamp": "2026-11-03T10:15:02.114Z",
  "filename": "bracket.ply", "file_path": "bracket.ply", "file_size_bytes": 1920044, "file_sha256": "…",
  "part_type": "…", "batch_id": "…", "shift": "…", "operator_id": "…",
  "model_name": "EfficientNetB0-casting-v1", "model_version": "a1b2c3d4e5f6",
  "probabilities": { "defective": 0.93, "ok": 0.07 },
  "metrics": { "input_format": "ply", "points_in_file": 200000, "points_processed": 200000,
               "outliers_removed": 310, "alignment": "pca", "projection_coverage": 0.41,
               "mean_point_spacing": 0.0021, "original_extents": [..],
               "surface_variation": { "p50": …, "p95": …, "p99": …, "max": …, "reference": 0.05,
                                      "fraction_above_reference": 0.012 } },
  "timings": { "read_ms":…, "parse_ms":…, "align_ms":…, "outlier_filter_ms":…, "surface_analysis_ms":…,
               "projection_ms":…, "view_prep_ms":…, "preprocessing_total_ms":…, "inference_ms":…,
               "geometry_save_ms":…, "encode_ms":…, "server_total_ms":… },   // inference_ms absent if not run
  "warnings": ["…"],
  "localization": { "available": false, "reason": "…" },
  "depth_map_base64": "data:image/png;base64,…", "normal_map_base64": "data:image/png;base64,…",
  "has_geometry": true,
  "input_type": "scan3d",              // or "image" (photo inspection)
  "image_base64": null,               // JPEG preview data URL for photos (maps/metrics differ: width, height, color_mode)
  "duplicate_of": null                 // id of an earlier identical upload, if any
}
```

Upload transfer time and result-transmission time are measured by the browser (the dashboard shows them).

## `GET /api/scans`
Query: `skip`≥0, `limit` 1–200, `status` (`PASS|FAIL|REVIEW|UNCLASSIFIED`), `defect_type`, `part_type`, `search` (part id, file, batch, class).
Returns `{ "total", "skip", "limit", "items": [summary…] }`. Summary rows omit images/metrics/timings.

## `GET /api/scans/{id}` → full record (as above, without `duplicate_of`)

## `GET /api/scans/{id}/geometry`
```jsonc
{ "point_count": 60000, "face_count": 0,
  "bounds": { "min": [x,y,z], "max": [x,y,z] },
  "positions_b64": "<little-endian float32 xyz…>", "scores_b64": "<float32, 0..1 surface variation / 0.05>",
  "outlier_b64": "<uint8 flags>", "faces_b64": "<little-endian uint32 triples>|null",
  "defect_mask_b64": null,             // reserved for models with spatial output; always null today
  "score_reference": "surface_variation" }
```
Coordinates are normalised (part ≈ 1 unit across, z = depth axis).

## Metrics
| Path | Returns |
|---|---|
| `/api/metrics/summary` | `total_scanned, classified, passed, failed, needs_review, unclassified, pass_rate_pct, defect_rate_pct, review_rate_pct, avg_latency_ms, avg_confidence_pct` – rates/averages are `null` when there is no data |
| `/api/metrics/pareto` | Rejected scans by predicted class: `count, percentage, cumulative_pct` |
| `/api/metrics/shifts` | `by_shift`, `by_part_type` – groups are the values present in the data |
| `/api/metrics/activity?hours=24` | Hourly buckets (UTC) `{hour_start, total, failed}` |
| `/api/metrics/performance` | Per-stage `mean/p50/p95/max/n` over the latest 200 scans |
| `/api/metrics/export?format=csv\|json` | Latest 1000 records |

## `GET /api/system/status` and `GET /`
`limits.allowed_extensions` reflects what the loaded model can inspect and `limits.image_inspection` is `true` for image-mode models.
Model state (`ready`, `error`, `model_name`, `model_version`, `class_names`, `pass_classes`, `input_mode`, `confidence_threshold`, `warmup_ms`),
database (`backend`, redacted `url`, `fallback_reason`), `open3d_available`, `websocket_clients`, `limits`.

## WebSocket `/ws/live` (also `/api/ws/live`)
Origin must be in `CORS_ORIGINS` (otherwise closed with 1008). All messages are JSON with `type` and `ts` (UTC ISO).

| `type` | Fields | When |
|---|---|---|
| `hello` | `server_version` | on connect |
| `pong` | – | reply to client `{"type":"ping"}` (the dashboard pings every 20 s) |
| `progress` | `scan_id, stage, message, filename` | `stage`: `received`, `parsing`, `preprocessing`, `inference`, `saving` |
| `result` | `scan_id, scan` (summary row) | inspection stored |
| `failed` | `scan_id, code, message, filename` | rejected or failed (the HTTP response carries the same error) |

## Changes vs the v2 contract
* Removed: `hotspots`, `heatmap_base64`, `obj_data`, `/api/scans/simulate`, `/api/simulation/*`, `GET /api/scans/{id}/3d` (all were synthetic); `status` gains `REVIEW`/`UNCLASSIFIED`.
* `confidence`, `defect_type` may be `null`; `part_id` is no longer random; default `batch_id/shift/operator_id` are `"UNSPECIFIED"` (not invented values).
* WebSocket messages are now typed events instead of bare scan objects.
* Depth/normal maps are PNG (were JPEG); timestamps carry a `Z` suffix.
* Added: photo inspection (`input_type`, `image_base64`, `decode_ms`), `/api/system/status`, `/api/metrics/activity`, `/api/metrics/performance`, `/api/scans/{id}/geometry`, `timings`, `warnings`, `localization`, `model_*`, `file_*`, `duplicate_of`.
