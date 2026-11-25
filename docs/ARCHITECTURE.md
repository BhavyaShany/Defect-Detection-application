# Architecture (v3.0)

## Components

| Layer | Module | Responsibility |
|---|---|---|
| API | `backend/app.py` | App factory, CORS (explicit origins), routers, WebSocket endpoint, lifespan (DB init, model load) |
| API | `routers/scans.py` | Upload orchestration, history, detail, geometry |
| API | `routers/metrics.py` | Statistics computed from stored rows, export (CSV is formula-injection safe) |
| API | `routers/system.py` | Truthful runtime status (model, DB, limits) |
| Service | `services/scan_io.py` | Strict parsers: PLY (ascii/binary LE/BE), PCD (ascii/binary), STL (ascii/binary), OBJ. Raises `ScanError(code)` – never fabricates points |
| Service | `services/geometry.py` | PCA pose normalisation, statistical outlier mask, ICP (Open3D or NumPy/Kabsch), surface variation, depth/normal projection, model-input builder |
| Service | `services/image_io.py` | Photograph decode/validation (Pillow): corrupt/tiny/oversized images rejected, grayscale → RGB, JPEG preview for history |
| Service | `services/preprocessing.py` | Orchestrates the stages, returns viewer geometry, 2D maps, metrics, **per-stage timings**, warnings |
| Service | `services/model_inference.py` | Loads Keras/SavedModel + `labels.json`; `predict()` or explicit "unavailable"; pure decision logic (`interpret_prediction`) |
| Service | `services/geometry_store.py` | Atomic `.npz` persistence of viewer geometry; UUID-only file names |
| Service | `services/ws_manager.py` | Connection manager, event helpers |
| Data | `database/`, `models/scan_model.py` | SQLAlchemy model/queries; no seed data; schema check at start-up |
| UI | `frontend/src/*` | Dashboard, upload with progress, live feed hook, 3D viewer |

## Request flow (`POST /api/scans/upload`)

```
client XHR ──► read body (≤ MAX_UPLOAD_MB, 413 beyond) ──► sha256
   │            └─ progress: received
   ├─ [thread pool, ≤ MAX_CONCURRENT_INSPECTIONS at once]
   │    parse+validate (progress: parsing)
   │    run_preprocessing (progress: preprocessing)
   │    engine.predict  (progress: inference)      ← skipped if no model → UNCLASSIFIED
   │    save geometry .npz, encode PNG maps (progress: saving)
   ├─ duplicate check (sha256) → warning, not a block
   ├─ DB insert  (failure → 503, geometry file removed)
   └─ broadcast result ──► 201 + full record
```

CPU-bound work runs in the thread pool so the event loop (and therefore WebSockets) stays responsive – the
original code ran it directly in the async handler.

## Photo inspection (image-mode model)
If `labels.json` has `input_mode: "image"`, photo uploads (`.jpg .jpeg .png .bmp`) are decoded → resized (bilinear, identical to the TF training resize) → classified → stored with a preview; no geometry is produced. 3D uploads still run the full 3D pipeline but stay `UNCLASSIFIED` (mode mismatch is stated in the warnings). With a depth-mode model, photos are rejected (415).

## Data model (`scans` table)

`status ∈ {PASS, FAIL, REVIEW, UNCLASSIFIED}`; `defect_type` (predicted class) and `confidence` are **NULL** when
no model decision exists. Timestamps are naive UTC serialised with a `Z` suffix (the original emitted naive
strings that browsers parsed as local time, shifting every timestamp by the user's UTC offset). Per-stage timings,
preprocessing metrics, probabilities and warnings are JSON text columns; depth/normal maps are PNG data URLs.
Geometry lives outside the DB in `data/geometry/<uuid>.npz`.

## Decisions & definitions

* **Classification vs localisation.** The CNN gives a whole-part verdict. `localization.available` is always `false`
  today and the API says why. The viewer’s *surface variation* colouring is an unsupervised geometric
  descriptor λ₀/(λ₀+λ₁+λ₂) of k-nearest-neighbour covariance (k=16) on a unit-normalised part; it is labelled as
  geometry-only everywhere. A `defect_mask_b64` field and a *Defect overlay* viewer mode are reserved for a future
  model that outputs per-point masks.
* **Status semantics.** `FAIL` = predicted class not in `pass_classes` with confidence ≥ threshold; `PASS` = predicted
  pass class with confidence ≥ threshold; `REVIEW` = confidence below threshold (never auto-accepted);
  `UNCLASSIFIED` = no model result. Rates are computed over classified scans (PASS+FAIL+REVIEW).
* **Alignment.** PCA canonical orientation by default (centre, principal axes, skew-based sign, unit scale);
  optional ICP against `REFERENCE_SCAN_PATH`. The original ICP-to-synthetic-cylinder was removed because it distorted real scans.
* **Depth map.** Top-down z-buffer scaled by a *fixed physical reference* (`max(z-range, 0.2·width)`), not min–max
  stretched, so sensor noise on a flat part does not become a high-contrast image. Value 0 = no data; holes inside
  the silhouette are inpainted, the background is not. The hole-closing kernel adapts to point spacing.
* **Concurrency.** `asyncio.Semaphore(MAX_CONCURRENT_INSPECTIONS)` bounds parallel inspections; Keras calls are serialised with a lock.
* **Security.** Extension allow-list, streamed size cap, no user-controlled file paths, UUID validation on geometry access,
  control-character stripping on text fields, CORS origin allow-list (also enforced on the WebSocket), CSV formula-injection
  neutralisation, DB URL password redacted in status output. No authentication is implemented.
