# Change summary: v2.4.0 (as received) → v3.1.0

## v3.1.2 – deployment kit
Backend serves the built dashboard (`FRONTEND_DIST`, same origin); `/api/health` added (root `/` is the dashboard when `frontend/dist` exists); production dashboard needs no API URL; WebSocket origin policy also accepts the page's own origin (`services/origin.py`, tested); `Dockerfile` (multi-stage), `docker-compose.yml` (app + Caddy: HTTPS, shared password), `deploy/Caddyfile`, `deploy.env.example`, `.dockerignore`, `docs/DEPLOYMENT.md`, `docs/DEMO_GUIDE.md`. **The Docker/Caddy files were not executed in the build environment.**

## v3.1.1 – input check for photo models
Trigger: a selfie was classified `ok_front` with high confidence (closed-set softmax). Added `services/ood.py` (Mahalanobis feature-distance + colour gate), engine integration (`predict_image` returns `ood`; flagged → `UNCLASSIFIED`, no class/confidence), `ml/fit_ood.py` (fits on the installed model, calibrates on held-out photos, reports rejection of synthetic and user-supplied negatives), dashboard banner when inactive and result details when active, `verify_model --no-input-check`, config `OOD_PATH`/`OOD_COLOR_CHECK`, 14 tests. **Limits:** statistical, can miss look-alike images; ~0.5 % of genuine photos refused by default; the TensorFlow parts (`build_feature_model`, `fit_ood` run) were not executable in the build environment - first real run is on the user's machine.

## v3.1.0 – photo inspection + training workflow (added after v3.0.0)
* **Image-mode models:** `labels.json` `input_mode: "image"`; backend accepts `.jpg/.jpeg/.png/.bmp` uploads only while such a model is loaded (new `services/image_io.py`, `input_type` + `image_base64` DB columns, `decode_ms` timing); dashboard shows the photo instead of the 3D viewer. Mode mismatches are refused or flagged explicitly.
* **`ml/train_model.py` rewritten for photos:** official `train/`+`test/` layout (incl. nested Kaggle folders), validation carved from train, **exact-duplicate leakage check** between train and test, class weights, **confidence-threshold analysis** (missed defects / false rejects / review rate per threshold), automatic cautions, `--smoke-test`, reproducibility info in the report.
* **Docs/tools:** `docs/TRAINING_GUIDE.md`, `ml/train_colab.ipynb`.
* **Tests:** +18 (62 core + 16 API, of which the API ones need FastAPI).
* **DB:** two new columns; a database created by v3.0.0 must be deleted (same start-up message as for v2).

## v3.0.0

## Approach
The architecture (React + Three.js ↔ FastAPI ↔ preprocessing ↔ EfficientNet ↔ SQL database, WebSocket updates) was kept.
What was removed or replaced is the *simulation layer* that fabricated results (see [AUDIT_REPORT.md](AUDIT_REPORT.md) §1),
because it could not be preserved without violating "no mock data / no fabricated confidence or locations".

## Removed (and why)
| Removed | Reason / replacement |
|---|---|
| `backend/routers/simulation.py`, `/api/scans/simulate`, conveyor-simulation UI | Wrote synthetic scans with pre-chosen defects into the production DB. Replaced by `tools/generate_test_scans.py` (files for testing only, never inserted into the DB) |
| `backend/services/mesh_generator.py` | Synthetic part generator coupled to the simulator; replaced by `tools/synthetic_parts.py` (test utility, denser sampling) |
| Fallback "model" (random confidences, injected labels, fake Grad-CAM) | Replaced by real inference or an explicit `unavailable` state → `UNCLASSIFIED` |
| `seed_initial_data()` and the shipped `defect_inspection.db` | 45 random rows seeded; DB contained 274 simulated rows |
| Random hotspots, `heatmap_base64`, `obj_data`, `/scans/{id}/3d` | Fabricated; replaced by real stored geometry + `/geometry` |
| Random-points fallback in the parser | Replaced by strict parsers with error codes |
| ICP against an invented cylinder | PCA normalisation by default; ICP only with a real reference |
| `DefectStatistic` table, `pydantic-settings` dependency, `node_modules/`, `__pycache__/` and `.db` from the package | Unused / unnecessary / not source |

## Files added
`README.md`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, `frontend/.env.example`, `docs/{ARCHITECTURE,API,AUDIT_REPORT,TESTING_REPORT,CHANGELOG}.md`,
`backend/{__init__,database/__init__,models/__init__,routers/__init__,services/__init__}.py`, `backend/routers/system.py`,
`backend/services/{scan_io,geometry,preprocessing,geometry_store,ws_manager,runtime}.py`,
`frontend/src/{api.js,useLiveFeed.js,components.jsx}`, `ml/{__init__,build_depth_dataset,train_model}.py`,
`tools/{__init__,synthetic_parts,generate_test_scans,benchmark,check_wiring}.py`,
`tests/{__init__,helpers,test_scan_io,test_geometry,test_geometry_store,test_inference,test_ml_tools,test_api}.py`,
`models/{README.md,labels.json.example}`, `data/` placeholders.

## Files modified (rewritten in place, same roles)
`backend/app.py`, `backend/config.py`, `backend/database/{database,crud}.py`, `backend/models/scan_model.py`,
`backend/routers/{scans,metrics}.py`, `backend/services/model_inference.py` (was in `open3d_pipeline.py` / `model_inference.py`; preprocessing moved to `preprocessing.py` + `geometry.py`),
`frontend/src/{App.jsx,ThreeViewer.jsx,App.css}` (CSS: existing styles kept, new ones appended), `frontend/{index.html,package.json}`, `.gitignore`.

## Bugs fixed (numbers refer to the audit table)
1 WebSocket teardown loop + leaked reconnect timers · 2–3 fabricated inference / label leakage · 4 uploaded scan never displayed ·
5 random-points fallback · 6 simulated/seeded statistics · 7 ICP distortion · 8 event-loop blocking (thread pool + concurrency cap) ·
9 timestamp time-zone bug (`Z` suffix) · 10 hard-coded `model_loaded` · 11 slow projection loop (vectorised z-buffer) ·
12 viewer: working spin toggle, correct coordinate space, full GPU resource disposal, no fake placeholder part ·
13 configurable API URL, no hard-coded display numbers · 14 consistent counts, escaped `LIKE` · 15 shift/part breakdown from real values ·
16 unbounded cache removed · 17 CORS origin allow-list incl. WebSocket · 18 streamed size cap + content validation ·
19 paths resolved from project root · 20 CSV formula-injection · 21–25 hacks, unused code, random IDs, silent DB fallback (now logged and shown in status), packaging.

## Features improved / added
Strict multi-format loader (PLY/PCD/STL/OBJ, ascii+binary) · upload progress, stage indicator, drag & drop, duplicate warning · `REVIEW` status using the confidence threshold ·
honest `UNCLASSIFIED` state and model banner · measured per-stage timings + server p95 + browser-side round trip · real 3D geometry viewer (points/surface/wireframe, surface-variation colouring with legend and caveat, outlier toggle, point size, reset view, loading/error states, cleanup) ·
paginated, searchable, status-filtered history · statistics with `null` instead of placeholder numbers · hourly activity chart · typed WebSocket events with heartbeat, back-off and re-sync ·
model/label validation, model provenance (name + fingerprint) stored per scan · training and dataset-building tools that reuse the live preprocessing · documentation and tests.

## Dependencies
Added to requirements (new file): `fastapi`, `uvicorn[standard]`, `python-multipart`, `sqlalchemy>=2`, `numpy`, `scipy`, `opencv-python-headless`, `Pillow`. Optional: `tensorflow`, `open3d`, `psycopg2-binary`. Dev: `httpx`, `pytest`.
Not used: `pydantic-settings`. Frontend: unchanged (`react`, `react-dom`, `three`, `vite`, `@vitejs/plugin-react`).

## Breaking changes for existing data/clients
* **Old v2 databases are not migrated** – they contained simulated rows. At start-up an old `scans` table triggers a clear error; move/delete the old file (default location is now `data/defect_inspection.db`).
* API contract changes: listed in [API.md](API.md#changes-vs-the-v2-contract).
* Start uvicorn from the project root: `uvicorn backend.app:app` (unchanged).

## Remaining issues / not done
* **No trained model or training data was provided** – classification accuracy is unknown until a model is trained (`ml/` tools) and evaluated.
* Server paths needing FastAPI/SQLAlchemy/TensorFlow, and the whole frontend at runtime, were **not executed** in the audit environment – run `tests/test_api.py` and the manual smoke test ([TESTING_REPORT.md](TESTING_REPORT.md)).
* Defect **localisation** does not exist (needs a segmentation/detection model with per-point or per-pixel labels).
* Gentle shrinkage-type defects are not visible to the surface-variation analysis.
* No authentication, retention policy for stored geometry/records, Alembic migrations, Docker setup, TFLite export, screenshots.

## Recommended next steps
1. Build a labelled 3D dataset and train/evaluate with `ml/` tools; record the evaluation report with the model.
2. Add multi-scale surface analysis (e.g. k = 16/64) and, when mask data exist, a segmentation head → fill `defect_mask_b64` (viewer already supports it).
3. Share one k-NN query between outlier filtering and surface analysis (≈ 30–40 % less preprocessing time); move to background workers (e.g. Celery/RQ) for very large scans.
4. Add authentication, audit logging, retention/cleanup jobs, Alembic migrations, Dockerfile + compose with PostgreSQL.
5. MES integration endpoint (push results to the MES; the export format is the starting point).
6. Browser end-to-end tests (Playwright) and screenshots for the README.
