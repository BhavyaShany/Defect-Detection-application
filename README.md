# Real-Time Defect Detection and Classification for Cast Metal Parts

Inspection dashboard: upload a 3D scan (point cloud / mesh) **or a photograph** of a cast part, preprocess it,
classify it with a trained EfficientNet CNN, and review the result, the 3D geometry, history
and statistics in a React dashboard with live WebSocket updates.

> **Status in one paragraph.** The full pipeline (upload → validation → preprocessing →
> classification → storage → WebSocket → dashboard → 3D viewer) is implemented and everything
> shown comes from real data. **No trained model is bundled**: until you train/provide one
> (`models/defect_model.keras` + `models/labels.json`, see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)) scans are preprocessed and viewable in 3D
> but stored as `UNCLASSIFIED` – the system never invents a prediction. A model is either an
> **image-mode** model (classifies photographs, e.g. trained on the Kaggle casting photos) or a
> **3D-mode** model (classifies depth maps from 3D scans); the app supports both. The classifier gives a
> **whole-part verdict, not defect locations**; the 3D colour map is a separate, clearly-labelled
> geometric analysis. See [Known limitations](#known-limitations).

## Contents
[Architecture](#architecture) · [Tech stack](#technology-stack) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [Model setup & training](#model-setup-and-training) ·
[Using the dashboard](#using-the-dashboard) · [API](#api-summary) · [WebSocket](#websocket-summary) ·
[Testing](#testing) · [Performance](#performance) · [Project layout](#project-layout) ·
[Screenshots](#screenshots) · [Known limitations](#known-limitations)

## Architecture

```
                 ┌────────────────────────── React dashboard (Vite) ──────────────────────────┐
                 │ Upload (XHR progress) · Result · Timings · History · Analytics · ThreeViewer │
                 └───────────────▲───────────────────────────────▲─────────────────────────────┘
                     REST /api/*  │                               │ WebSocket /ws/live
                 ┌───────────────┴───────────────────────────────┴─────────────────────────────┐
                 │                          FastAPI  (backend/app.py)                           │
                 │  routers: scans · metrics · system         ws_manager (progress/result/…)    │
                 └──┬───────────────┬──────────────────┬──────────────────┬────────────────────┘
                    │               │                  │                  │
              scan_io.py     preprocessing.py   model_inference.py   database/ + geometry_store
           PLY/PCD/STL/OBJ   align · outliers ·  EfficientNet (TF)    SQLite/PostgreSQL rows,
           parse+validate    surface analysis ·  or "unavailable"     .npz geometry per scan
                             depth/normal maps
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Audit of the original code base:
[docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md).

**Processing flow for one upload** (3D scan; photographs skip steps 2-3 and are decoded and classified directly)

1. Browser validates extension/size, uploads with progress (XHR).
2. Server streams the body (size-capped), hashes it, parses it with a strict parser – bad input is rejected, never replaced by fake data.
3. Preprocessing: sanitise → PCA pose normalisation (or ICP against a configured reference) → statistical outlier filtering → surface-variation analysis → top-down depth + normal maps.
4. Inference with the trained model (if loaded) → `PASS` / `FAIL` / `REVIEW` (below confidence threshold); otherwise `UNCLASSIFIED`.
5. Geometry + record stored; per-stage timings stored; progress and result broadcast over WebSocket.

## Technology stack

| Area | Used |
|---|---|
| Frontend | React 18, Vite 5, Three.js 0.160, plain CSS, WebSocket client |
| Backend | Python, FastAPI, Uvicorn, SQLAlchemy 2 |
| Processing | NumPy, SciPy (KD-tree), OpenCV; **Open3D optional** (used automatically if installed) |
| ML | TensorFlow/Keras, EfficientNetB0 transfer learning (training script in `ml/`) |
| Database | SQLite by default; PostgreSQL via `DATABASE_URL` |
| Not implemented | Docker/compose, TFLite export, authentication, Alembic migrations |

## Quick start

Requirements: Python 3.10–3.12 recommended (TensorFlow/Open3D wheels lag behind the newest
Python releases – check availability if you use 3.13), Node.js 18+.

### 1. Backend

```bash
# from the project root
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env   (optional; defaults work)
uvicorn backend.app:app --reload --port 8000
```

Check http://localhost:8000/api/health – `model_loaded` tells you whether a model is available, and
http://localhost:8000/docs has the interactive API docs. Always start uvicorn **from the project
root** (the code imports `backend.*`).

Optional extras: `pip install tensorflow` (to run a trained model), `pip install open3d`
(faster outlier removal / ICP; the pipeline works without it).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

If the API is not on `http://localhost:8000`, copy `frontend/.env.example` to `frontend/.env`
and set `VITE_API_BASE_URL`.

### 3. Try it without a scanner

```bash
python -m tools.generate_test_scans --out test_scans
```

creates 24 synthetic `.ply` files (4 part shapes × nominal/porosity/shrinkage/crack/inclusion/flash).
They are for exercising the pipeline and viewer only – they are **not training data** and carry no
ground-truth classification. Upload one from the dashboard.

## Configuration

Environment variables (or `.env`; see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | SQLite `data/defect_inspection.db` | e.g. `postgresql+psycopg2://user:pass@host/db` (`pip install psycopg2-binary`) |
| `DB_FALLBACK_SQLITE` | `true` | If the configured DB is unreachable, fall back to SQLite (logged + shown in `/api/system/status`). `false` = fail at start-up |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins (also checked for WebSockets) |
| `MAX_IMAGE_PIXELS` | `40000000` | Largest accepted photo (decompression-bomb guard) |
| `MODEL_PATH` / `MODEL_LABELS_PATH` | `models/defect_model.keras` / `models/labels.json` | Trained model + labels |
| `CONFIDENCE_THRESHOLD` | `0.75` | Predictions below this are `REVIEW`, never auto-pass/fail |
| `MAX_UPLOAD_MB` | `100` | Upload size limit (HTTP 413 above) |
| `MAX_CONCURRENT_INSPECTIONS` | `2` | Parallel inspections (memory/CPU guard) |
| `ALIGNMENT_MODE` | `pca` | `none` \| `pca` \| `icp` (`icp` needs `REFERENCE_SCAN_PATH`) |
| `MAX_VIEW_POINTS` | `60000` | Max points sent to the 3D viewer (random subset above this) |

More tunables (outlier filtering, projection size, …) are in `backend/config.py`. Secrets (e.g. a
PostgreSQL password) belong only in `.env`, which is git-ignored.

## Model setup and training

**Quick route (photos):** follow [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) (Colab notebook: `ml/train_colab.ipynb`):
`python -m ml.train_model --data-dir casting_data --pass-classes ok_front --input-mode image`

The model is expected at `models/defect_model.keras` with `models/labels.json`
(format: [models/README.md](models/README.md)). Class names **come from `labels.json`** – nothing is hard-coded.

**Training pipeline (3D data):**

```bash
# 1. Labelled point clouds: scans/<class_name>/*.ply|pcd|stl|obj
python -m ml.build_depth_dataset --scans scans --out dataset --mode depth
# 2. Train (EfficientNetB0, frozen-base training then fine-tuning; needs tensorflow)
python -m ml.train_model --data-dir dataset --pass-classes ok --input-mode depth
```

Step 1 uses exactly the same preprocessing as the live API, so training and inference images
match. Step 2 writes `models/defect_model.keras`, `labels.json`, and `evaluation_report.{json,md}`
(accuracy, per-class precision/recall/F1, confusion matrix on a held-out test split). Restart the
backend afterwards; the header badge shows the loaded model.

**Dataset note.** The Kaggle *casting product* dataset contains 2-D photographs (`def_front` /
`ok_front`) of one part type, not 3D scans. Train it with `--input-mode image`; the resulting model
classifies photographs only (3D scans uploaded meanwhile are viewable but `UNCLASSIFIED`). A photo model is
**not** compatible with depth maps from point clouds; for 3D use the pipeline above with real labelled scans.
`input_mode` in `labels.json` must always match the training images.

## Deploying and presenting
* [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - single-process mode, tunnel, cloud VM with Docker + HTTPS + password, GitHub publishing.
* [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) - what to show, what you can and cannot claim, how to position it for a professor vs. a company.
* The backend serves the built dashboard itself: `cd frontend && npm run build`, then `uvicorn backend.app:app --port 8000` and open `http://localhost:8000`.

## Using the dashboard

* **New inspection** – choose or drop a scan (`.ply .pcd .stl .obj`; plus `.jpg .jpeg .png .bmp` when an image-mode model is loaded), optionally fill part type / batch / shift / operator, and watch the stage indicator (upload → parse → preprocess → inference → save).
* **Result** – status (`PASS` / `FAIL` / `REVIEW` / `UNCLASSIFIED`), predicted class, confidence, probabilities, model/version, scan-processing metrics, warnings and **measured** per-stage times.
* **3D Inspection** – orbit/zoom/pan, reset view, points / surface / wireframe (surface modes only for meshes), original colour or *surface variation* colouring, outlier toggle, point-size slider. A *Defect overlay* mode appears only if a future model supplies a per-point defect mask.
* **Depth & Normal Maps** – the images fed to the CNN.
* **Analytics** – defect categories (rejected scans), hourly activity, by-shift breakdown, measured timing statistics.
* **Inspection log** – paginated, searchable, filterable by status; click a row to reload its result and 3D geometry (persisted across restarts).
* **Export** – CSV/JSON of the latest 1000 records.

## API summary

Full reference with examples: [docs/API.md](docs/API.md).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/scans/upload` | Upload + inspect (multipart: `file`, optional `part_type`, `batch_id`, `shift`, `operator_id`, `part_id`) → `201` |
| GET | `/api/scans` | Paginated history (`skip`, `limit`, `status`, `defect_type`, `part_type`, `search`) |
| GET | `/api/scans/{id}` | Full record (maps, metrics, timings, warnings) |
| GET | `/api/scans/{id}/geometry` | Processed geometry for the viewer |
| GET | `/api/metrics/{summary,pareto,shifts,activity,performance,export}` | Statistics from stored records |
| GET | `/api/system/status` | Model / database / limits status |
| WS | `/ws/live` | Live events |

Errors always look like `{"detail": {"code": "...", "message": "..."}}`.

## WebSocket summary

`ws://localhost:8000/ws/live` sends JSON events: `hello`, `progress` (`received → parsing →
preprocessing → inference → saving`), `result`, `failed`; reply `pong` to `{"type":"ping"}`. The
dashboard reconnects with exponential back-off, polls while disconnected, and re-syncs on reconnect.

## Testing

```bash
python -m unittest discover -s tests -t . -v        # 76 core tests (no web framework needed)
pip install -r requirements-dev.txt                 # adds FastAPI TestClient
python -m unittest tests.test_api -v                # 17 API/WebSocket tests (skipped if FastAPI missing)
python -m tools.check_wiring                        # static import/name check
python -m tools.benchmark                           # measure preprocessing latency on your machine
python -m ml.fit_ood --data-dir <casting_data> --ood-dir <non-casting images>   # fit the input check (photo models)
python -m tools.verify_model --data-dir <casting_data>   # installed model vs training report (see docs/ACCEPTANCE_TEST.md)
```

Results, coverage and what could *not* be run during the audit: [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md).

## Performance

Measured preprocessing time (single CPU core, synthetic scans; **excludes model inference and
network**; your numbers will differ – run `python -m tools.benchmark`):

| Points | Preprocessing total |
|---|---|
| 10 k | ≈ 0.08 s |
| 50 k | ≈ 0.4 s |
| 200 k | ≈ 2.0 s |
| 1 M | ≈ 11 s |

The dashboard shows true per-stage timings for every scan (and server-side p95 in Analytics), plus
browser-measured upload time, round trip and geometry load time. "Real-time" should be judged on
those numbers for *your* scan sizes and hardware, not on the fact that the UI updates live.

## Project layout

```
backend/   app.py · config.py · routers/ (scans, metrics, system) · services/ (scan_io, geometry,
           preprocessing, model_inference, geometry_store, ws_manager) · database/ · models/
frontend/  src/App.jsx · components.jsx · ThreeViewer.jsx · api.js · useLiveFeed.js · App.css
ml/        build_depth_dataset.py · train_model.py · fit_ood.py · train_colab.ipynb
tools/     generate_test_scans.py · synthetic_parts.py · benchmark.py · check_wiring.py
tests/     unit + API tests          models/   model + labels go here          data/  runtime data
docs/      ARCHITECTURE · API · DEPLOYMENT · DEMO_GUIDE · TRAINING_GUIDE · AUDIT_REPORT · TESTING_REPORT · CHANGELOG
```

## Screenshots

Screenshots were **not** captured during the audit (no browser was available in the audit
environment). To add them: run both servers, upload a file from `test_scans/`, then save images
into `docs/screenshots/` (dashboard overview, 3D viewer in *surface variation* mode, analytics
tab) and reference them here.

## Known limitations

* **No bundled model / dataset.** Without `models/defect_model.keras` + `labels.json` every scan is `UNCLASSIFIED`. Classification quality is whatever your trained model achieves; see its `evaluation_report`.
* **Non-casting images.** A softmax classifier cannot say "this is not a casting". The app therefore adds an *input check* (`python -m ml.fit_ood`, writes `models/ood.npz`): images whose features are far from the training photos, or that are colour images while the training photos were grayscale, are **not classified** (`UNCLASSIFIED` + explanation). It is a statistical safeguard calibrated on your data (default: ~0.5 % of genuine photos are refused); it can still pass images that closely resemble the training photos, so test it with your own negatives. Without `ood.npz` the dashboard shows a warning banner.
* **Classification ≠ localisation.** The CNN labels the whole part. The 3D colour map is *surface variation* (local curvature) – it highlights edges, pits and roughness but cannot tell a defect from a legitimate feature, and gentle depressions (e.g. shrinkage) are largely invisible to it (measured contrast ≈ 1.1× flat surface vs ≈ 4× for pits/cracks/bumps).
* **PCA alignment is a heuristic** – nearly symmetric parts can come out in an ambiguous orientation. Use `ALIGNMENT_MODE=icp` with a real reference scan for consistent poses.
* `binary_compressed` PCD files are rejected (re-export as ascii/binary).
* Large scans are subsampled (>1.5 M points for processing, >60 k for display). Processing time grows roughly linearly with point count.
* No authentication/authorisation, no data-retention/cleanup policy for stored geometry, no Alembic migrations, no Docker setup. Old v2 databases are intentionally not migrated (they contained simulated rows) – see [docs/CHANGELOG.md](docs/CHANGELOG.md).
* The server-side code paths that need FastAPI/SQLAlchemy/TensorFlow were not executed in the audit environment (see testing report); run the API test suite on your machine before a demo.
