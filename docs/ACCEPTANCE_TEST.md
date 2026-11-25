# Acceptance test (after the model is installed)

Do these in order; stop at the first failure and fix it before continuing.

## 1. Automated suite (5 min)
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
```
Expected: **no skipped tests** and no failures (the 16 API tests now run because FastAPI is installed). They use their own temporary database and stub models, so your real model and data are not touched.

## 2. Model reproduces the training report (5-15 min)
```bash
python -m tools.verify_model --data-dir <path to casting_data>
```
Runs the whole cleaned test set through the app's own image loader and model code. Expected: `RESULT: MATCH` (accuracy within 0.5 points of the report, about 3 images of 651). `MISMATCH` means the app does not reproduce training: check the TensorFlow version, `labels.json`, and report the output.
Use `--limit 50` for a quick run (no report comparison).

## 3. Server and dashboard (15 min)
Start the backend and frontend (see README). Then:

| # | Action | Expected |
|---|---|---|
| 1 | Open http://localhost:8000/ | `"model_loaded": true`, classes `def_front`, `ok_front` |
| 2 | Open the dashboard | Header: *Live updates connected* and *Model: casting-photo-v1*; no yellow banner |
| 3 | Upload 5 defective photos from `test/def_front` | `FAIL` · class `def_front` · high confidence · photo shown |
| 4 | Upload 5 good photos from `test/ok_front` | `PASS` · class `ok_front` |
| 5 | Upload a photo that is not a casting (e.g. any random picture) | Look at the result. The model has no "not a casting" class, so it will still output a class, possibly confidently. Record this; it is a known limitation |
| 6 | Upload a `.ply` from `test_scans/` | Preprocessed + 3D view, status `UNCLASSIFIED`, warning that the model classifies photographs |
| 7 | Upload an empty file, a renamed `.txt`→`.jpg`, a `.pdf` | Clear error each time, nothing added to the log |
| 8 | Upload the same photo twice | Second result carries an "already inspected" warning |
| 9 | Check *Processing time (measured)* | Decode, inference, total times filled; note inference ms on your hardware |
| 10 | Check KPI cards, Analytics tab, inspection log | Counts equal the scans you uploaded; filters/search work |
| 11 | Export CSV | Opens in Excel with your scans |
| 12 | Stop the backend, wait 10 s, restart it | Header shows *reconnecting*, then connects; log still contains all scans |
| 13 | Close and reopen the browser | History and statistics persist; clicking an old row shows its photo and result |

## 4. Record the result
Write down: date, model name + fingerprint (shown on each result), `CONFIDENCE_THRESHOLD`, the `verify_model` output, and any failures. Keep it with the model's `evaluation_report.md`.
