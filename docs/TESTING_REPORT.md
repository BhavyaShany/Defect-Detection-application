# Testing report

## 1. What could and could not be executed

The audit was done in a sandbox **without network access**. Installed: Python 3.12, NumPy 2.4, SciPy 1.17, OpenCV 4.13, Node 22.
**Not installed (and not installable): FastAPI, SQLAlchemy, TensorFlow, Open3D, a browser.** Consequences:

| Area | Executed here? | How verified |
|---|---|---|
| Scan parsing/validation, geometry, preprocessing, image loading, inference decision logic, geometry store, ML tooling (splits, leakage check, threshold analysis) | **Yes** | 62 automated tests, all passing |
| Server wiring (routers, DB, WebSocket) | **No (not run)** | Static check (`tools.check_wiring`: every internal import resolves, no undefined names) + 13 API tests written but **skipped here** (need FastAPI). **Run them on your machine before the demo.** |
| Real Keras/TensorFlow model loading and prediction | **No** | Decision logic and the engine contract are tested with a stub model; `ml/train_model.py` (its TensorFlow section) and `_load_tf` are syntax-checked only; **the training run itself has never been executed** |
| Open3D acceleration path | **No** | NumPy/SciPy path tested (it is the fallback); Open3D calls are wrapped and fall back on any exception |
| Frontend in a browser (rendering, WebGL, WebSocket, XHR progress) | **No** | JSX parsed with Babel (all files valid); JS decoder verified against bytes written by the Python backend; imports/exports cross-checked. Visual/interactive behaviour **unverified** |

## 2. Automated results (this environment)

`python -m unittest discover -s tests -t .` → **78 tests: 62 passed, 16 skipped (API suite), 0 failed.**

| Module | Tests | What is covered |
|---|---|---|
| `test_scan_io` | 16 | PLY ascii / binary LE / binary BE / mesh, PCD ascii+binary, STL ascii+binary (vertex dedup), OBJ (`v/vt/vn`); NaN removal reporting; **rejection** of wrong extension, empty file, garbage with valid extension (all 4 formats), truncated PLY/PCD, too few points, degenerate geometry, unsupported PCD compression, out-of-range faces |
| `test_geometry` | 17 | PCA invariance + unit scale + preserved original extents; outlier removal; ICP recovers a known rigid transform; surface variation (plane ≈ 0, noisy/curved ≫ plane, pits/cracks/bumps/ridges ≥ 2.5× flat baseline, defective > nominal); projection shape/range/no-data semantics, noise not amplified, bump visible; all 4 parts × 6 defect types run end-to-end; large-cloud cap; mesh faces kept; sparse/thin scan rejected; timings recorded |
| `test_inference` | 11 | PASS/FAIL/**REVIEW** logic; class names from labels; probability normalisation (softmax/sigmoid); labels validation (7 invalid variants); missing model/labels/invalid labels → `unavailable`, never a prediction; stub-model contract (input shape 1×224×224×3, 0–255 range); exceptions → `InferenceError`; wrong output size |
| `test_geometry_store` | 1 | Round trip, optional faces, path-traversal guard, delete, no temp-file leaks |
| `test_ml_tools` | 9 | 3D dataset builder; metrics; single-folder and official (nested Kaggle) layouts; train/test class mismatch; **duplicate leakage detection**; tiny-class rejection; threshold analysis (missed defects / review %); cautions |
| `test_image_mode` | 8 | Photo decode (PNG/JPEG/BMP, grayscale→RGB), rejection of empty/corrupt/truncated/tiny/oversized images, preview, image-mode engine contract, mode-mismatch refusal both ways |
| `test_api` | 16 (skipped) | Health/model truthfulness, empty-DB metrics are `null`, UNCLASSIFIED path without model, PASS/FAIL/REVIEW with stub model, statistics/history from real rows, duplicate flag, invalid inputs (415/400/422), oversize 413, inference failure → UNCLASSIFIED, 404s, WebSocket events (hello/pong/progress/result/failed), CSV injection safety, system status |

### Issues found while testing and fixed
| Found by | Issue | Fix |
|---|---|---|
| Geometry tests | A valid sparse mesh (900 vertices) was rejected: "coverage" counted only raw pixel hits | Coverage is now the closed silhouette fraction; hole-closing kernel adapts to point spacing |
| Geometry tests | Config import failed without `pydantic-settings` | Replaced the dependency with a small stdlib `.env` loader (fewer dependencies) |
| Measurement | Shrinkage-type gentle depressions are **not** detectable by surface variation (contrast 1.1× flat) while pits/cracks/bumps are (≈ 3.7–4.3×) | Documented as a limitation; the test asserts only what is true |
| Review | Self-review of the WebSocket API test found it would read too few events for a rejected upload | Test loops until `failed` |
| Review | Stray variable reference in the ASCII-PLY branch (`NameError` for pure point clouds) | Removed |

## 3. Requested test cases – status

| # | Test case | Status |
|---|---|---|
| 1 | Valid scan end-to-end | Pipeline stages **executed & passing** (parse → preprocess → maps → geometry store → JS decode). HTTP/DB/WebSocket/browser legs: automated tests written (`test_api`), **not executed here** |
| 2 | Defective component | Geometric analysis verified on synthetic defective parts. Model prediction/confidence: decision logic tested with a stub; **no trained model exists to test real accuracy** |
| 3 | Non-defective component | Same as above (nominal scores lower than defective; PASS path via stub) |
| 4 | Invalid files (extension, empty, corrupt, invalid cloud, oversized) | Parser-level cases **executed & passing**; HTTP status mapping (415/400/422/413) in `test_api` (not run) |
| 5 | Backend failure (backend down, model load failure, inference failure, DB failure, preprocessing failure) | Model load failure and inference failure: logic tested (engine `unavailable`; `InferenceError`) and API behaviour in `test_api` (not run). Backend-down: frontend shows "Backend unreachable" (code review only). DB failure → 503 path: code review only |
| 6 | WebSocket failure/reconnect | Client back-off/heartbeat/re-sync implemented; server events tested in `test_api` (not run). **Needs manual test:** stop/start the backend while the dashboard is open |
| 7 | 3D viewer | JSX syntax valid; decoder verified. Rendering, controls, large clouds, memory behaviour **need manual verification in a browser** (viewer disposes geometries/materials, controls and the WebGL context on unmount) |

### Suggested manual smoke test (≈ 10 min)
1. Start backend + frontend; header shows *Live updates connected* and *Model: not loaded* (if no model).
2. `python -m tools.generate_test_scans --out test_scans`; upload `engine_bracket__porosity.ply` → stage indicator advances, result shows `UNCLASSIFIED` + warning, 3D view shows the part with *surface variation* colouring, timings table filled.
3. Upload `x.jpg` and an empty `.ply` → clear error messages, nothing stored. Upload the same file twice → "already inspected" warning.
4. Stop the backend → header switches to *reconnecting*; restart → it reconnects and refreshes.
5. Reload the page → history and statistics persist; click an old row → geometry loads.
6. After training/providing a model → `PASS/FAIL/REVIEW`, confidence and probabilities appear.

## 4. Measured performance (preprocessing, single CPU core, synthetic scans, median of 3)

Excludes model inference, HTTP transfer and browser rendering. Command: `python -m tools.benchmark`.

| Points | file MB | parse | align | outlier filter | surface analysis | projection | **total** |
|---|---|---|---|---|---|---|---|
| 10 k | 0.1 | 0.8 ms | 3.6 | 23.8 | 40.4 | 10.5 | **78.7 ms** |
| 50 k | 0.5 | 4.1 | 13.9 | 132.3 | 203.0 | 15.0 | **415 ms** |
| 200 k | 1.9 | 13.3 | 66.0 | 766 | 1177 | 16.9 | **2.03 s** |
| 1 M | 9.6 | 78.8 | 346 | 4529 | 6122 | 40.4 | **11.2 s** |

Observations: KD-tree work (outlier filter + surface analysis) is ≈ 90 % of the time and scales ~linearly; SciPy uses all cores
(`workers=-1`), so multi-core machines will be faster than this single-core sandbox. A possible 30–40 % saving is to share one k-NN query between the two stages (listed under future work).
Model-inference latency, upload time and end-to-end latency on real hardware were **not measurable** here; the dashboard reports them per scan.
The original system's "18–35 ms inference" figures were random numbers and must not be quoted.
