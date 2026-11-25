# Project audit report

Subject: `defect-detection-app.zip` (backend ≈ 1,750 lines Python, frontend ≈ 2,000 lines JSX/CSS).
Method: full source read-through of every file, inspection of the shipped SQLite database,
then testing of everything that could be executed in the audit environment (see
[TESTING_REPORT.md](TESTING_REPORT.md)). Statements below marked **(verified)** were confirmed by running code or
querying the shipped database; the rest come from reading the code.

## 1. Headline finding

**The submitted system was a simulation, not an inspection system.** Almost every number on the dashboard was generated:

| Where | What actually happened |
|---|---|
| `model_inference.py` | No model file existed in the project. The "fallback" produced `random.uniform(0.87, 0.99)` confidences, random probability vectors and a latency drawn from `uniform(18.4, 32.8)` ms. |
| `scans.py /simulate` | The simulator **chose the defect class first**, generated geometry for it, then passed it to the "inference engine" as `injected_defect`, which returned it as the prediction. Classification accuracy of this loop is 100 % by construction. |
| `scans.py /upload` | Defect hotspots were `random.uniform` coordinates. `GET /scans/{id}` and `/3d` **regenerated a synthetic mesh from the part type** – an uploaded scan was never displayed; the viewer showed a made-up part. |
| `open3d_pipeline._parse_file_bytes` | If a file could not be parsed (any `.jpg`, a corrupt `.ply`, `.pcd`/`.stl` without Open3D…) it silently substituted **1,200 random points**, producing a "valid" inspection of noise. The upload dialog even advertised `image/*`. |
| `database.seed_initial_data` | 45 random historical records were inserted at start-up. **(verified)** The shipped DB held 322 rows: 274 simulated, 2 real uploads, the rest seeded. All statistics were computed from this. |
| `App.jsx` | Fallback display values when data was missing: confidence `98.4%`, latency `24.5 ms`, ICP RMSE `0.018`, fitness `0.96`, outliers `32`, batch `BATCH-402`. |
| `app.py /` | `"model_loaded": True` was hard-coded. |

This directly violates the project's own rules (no mock data in production workflows, no fabricated locations or confidence, no hard-coded statistics).
The fix was therefore not cosmetic: the simulated layers were removed and replaced with real processing or explicit "unavailable" states.

## 2. Existing architecture (as received)

React/Vite SPA (`App.jsx`, `ThreeViewer.jsx`) → FastAPI (`backend/app.py`; routers `scans`, `simulation`, `metrics`) →
services `open3d_pipeline` (denoise, ICP, projection), `model_inference` (EfficientNet wrapper + fallback), `mesh_generator` (synthetic parts) →
SQLAlchemy (`scans`, unused `defect_statistics`) on SQLite (Postgres URL supported with silent SQLite fallback). WebSocket `/ws/live` broadcast of bare scan dicts.
No `requirements.txt`, README, tests, Dockerfile, trained model, dataset, `__init__.py` files, or `.env` template. The archive contained `node_modules` (≈ 32 MB zip), `__pycache__` (Python 3.13) and the live `.db` file despite the `.gitignore`.

## 3. Feature checklist (original state)

| Feature | Status | Notes |
|---|---|---|
| Scan upload UI | **Partially working** | Button works; no progress, no validation, errors only in console; accepted images that cannot be processed |
| File-format handling | **Partially working** | `.obj` parsed; `.ply/.pcd/.stl` only if Open3D installed; failure → random points |
| Noise filtering | Working | Statistical outlier removal (Open3D or SciPy) |
| ICP alignment | **Partially working / misleading** | Aligned every scan to an invented cylinder, distorting real data |
| Depth / normal maps | **Partially working** | Per-point Python loop (slow for large clouds); min–max stretch amplified noise; odd constant gz |
| CNN classification | **Not implemented** | No model; random fallback |
| Confidence score | **Not implemented** | Random numbers |
| Defect localisation / highlighting | **Not implemented** | Random hotspot spheres; not mapped to mesh coordinates |
| 3D viewer of the uploaded scan | **Not implemented** | Synthetic mesh shown instead |
| 3D viewer controls | **Partially working** | Auto-rotate toggle had no effect (stale closure); no resource disposal; placeholder bracket shown when empty |
| Statistics | **Partially working** | Real SQL aggregates, but over simulated/seeded rows; hard-coded shift and part lists; UI fallbacks |
| Scan history | **Partially working** | Latest 30 only; filter counts reflect the 30; no pagination/search in UI |
| PostgreSQL / persistence | **Partially working** | SQLite persisted; Postgres failure silently fell back |
| WebSocket live updates | **Partially working** | Server broadcast worked; client tore down & reopened the socket on every new scan (see bug 1) |
| Reconnect handling | **Partially working** | Fixed 2.5 s retry, reconnect timers leaked after unmount; polling only while simulation ran |
| Processing-time measurement | **Not implemented** | Latency was synthetic |
| Export CSV/JSON | Working | No formula-injection protection |
| Documentation / tests / deployment config | **Not implemented** | |

## 4. Defects found

Severity: **C** critical (wrong results / core feature), **H** high, **M** medium, **L** low.

| # | Sev | Defect | Location |
|---|---|---|---|
| 1 | H | `fetchRecentScans` depends on `current`, so the init effect re-ran on every new scan: closed the WebSocket, reopened it, scheduled extra reconnect timers (verified by reading the effect dependency graph) | `App.jsx` |
| 2 | C | Fabricated inference/confidence/latency/hotspots/class probabilities (see §1) | `model_inference.py`, `scans.py` |
| 3 | C | Ground-truth leakage: injected defect used as prediction | `scans.py`, `model_inference.py` |
| 4 | C | Uploaded scans never rendered; synthetic geometry returned instead | `scans.py` |
| 5 | C | Unparseable input replaced with random points | `open3d_pipeline.py` |
| 6 | C | Seeded/simulated rows mixed into real statistics | `database.py`, `simulation.py` |
| 7 | H | ICP to a made-up cylinder distorts real scans | `open3d_pipeline.py` |
| 8 | H | CPU-bound work (parsing, KD-tree, projection, TF) ran inside `async def` handlers → blocks event loop and WebSocket | `scans.py` |
| 9 | H | Timestamps naive-UTC without `Z` → browsers read them as local time (wrong by the UTC offset, +5:30 for India) | `scan_model.py`, `App.jsx` |
| 10 | H | `model_loaded: True` hard-coded; model errors only printed | `app.py` |
| 11 | M | Projection used a Python `for` loop over all points (very slow ≥ 10⁵ points; verified replaced version 11 s for 1 M points) | `open3d_pipeline.py` |
| 12 | M | Auto-rotate toggle ineffective; hotspot coordinates in a different space than the scaled/centred mesh; geometries/materials never disposed on swap (GPU memory leak) | `ThreeViewer.jsx` |
| 13 | M | Hard-coded API URL; hard-coded fallback display numbers (§1) | `App.jsx` |
| 14 | M | `count_scans` ignored the defect-type search that `get_scans` applied (inconsistent totals); unescaped `LIKE` wildcards | `crud.py` |
| 15 | M | Shift/part-type breakdown hard-coded to 3 shifts / 4 part types – other values silently dropped | `crud.py` |
| 16 | M | Unbounded in-memory `mesh_cache` | `scans.py` |
| 17 | M | `CORS allow_origins=["*"]` with `allow_credentials=True`; WebSocket origin unchecked | `app.py` |
| 18 | M | No upload size cap (whole body read into memory), no content validation | `scans.py` |
| 19 | M | Relative paths (`models/…`, SQLite URL) depend on the working directory; model path pointed at a folder that did not exist | `config.py` |
| 20 | M | CSV export vulnerable to spreadsheet formula injection via user text | `metrics.py` |
| 21 | L | Hack code to sanitise `Form`/`Query` defaults when calling endpoints as functions | `scans.py`, `app.py` |
| 22 | L | `DefectStatistic` table never used; `CONFIDENCE_THRESHOLD` never used; deprecated `datetime.utcnow`, pydantic `class Config` | various |
| 23 | L | Random part IDs could collide (`randint(1000, 9999)`) | `scans.py` |
| 24 | L | Silent SQLite fallback when PostgreSQL unreachable | `database.py` |
| 25 | L | No `__init__.py`, requirements, README, tests; `node_modules`, `.pyc` and the live DB were packaged | repo |

## 5. Recommendations carried out / left open
See [CHANGELOG.md](CHANGELOG.md) for exactly what was changed and what remains.
