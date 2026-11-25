# MES integration and automated alerting

The system is the quality gate of the process: every inspection ends in a decision (`PASS` /
`FAIL` / `REVIEW` / `UNCLASSIFIED`) that has to reach the Manufacturing Execution System, and
process problems have to raise alerts without a human watching the dashboard.

Two pieces implement this:

| Piece | Module | What it does |
|---|---|---|
| Outbox + dispatcher | `backend/services/mes.py` (`enqueue`, `deliver`, `dispatch_pending`, `dispatch_loop`) | Turns each stored inspection and each alert into a JSON event, commits it to `mes_outbox`, and POSTs it to the MES with retries |
| Alert rule engine | `backend/services/mes.py` (`evaluate_scan`) | Applies the rules below to the new record plus the stored history and stores `alerts` rows |
| API | `backend/routers/mes.py` | `/api/mes/*`, `/api/alerts/*` |
| Dashboard | *Alerts & MES* tab | Queue depth, integration state, alert list, acknowledge / dispatch buttons |

Nothing is simulated: an event exists only because an inspection was stored, an alert only
because a rule matched real rows, and a delivery only after the MES actually answered.

## Why an outbox

A MES is an external system: it can be down, slow or reject an event. Writing the event in the
same transaction as the inspection (before any HTTP call) means

* a stored inspection **always** has a queued event -- no lost quality decisions on a MES outage;
* the upload response is never blocked by MES latency (the HTTP call happens in the dispatcher);
* delivery state is inspectable (`GET /api/mes/outbox`) instead of living in a log line.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MES_ENABLED` | `false` | Master switch. Off = no queueing, no dispatch (alerts are still generated) |
| `MES_WEBHOOK_URL` | – | Endpoint that receives the JSON events (`POST`) |
| `MES_API_KEY` | – | Optional credential, sent in `MES_AUTH_HEADER`. Never logged or returned by the API |
| `MES_AUTH_HEADER` / `MES_AUTH_SCHEME` | `Authorization` / `Bearer` | Auth header shape (e.g. `X-API-Key` with an empty scheme) |
| `MES_TIMEOUT_SECONDS` | `5` | Per-request timeout |
| `MES_MAX_ATTEMPTS` | `5` | Attempts before the event is marked `DEAD` and a `MES_DELIVERY_FAILED` alert is raised |
| `MES_RETRY_BASE_SECONDS` / `MES_RETRY_MAX_SECONDS` | `5` / `300` | Exponential back-off (5, 10, 20, 40 … capped) |
| `MES_DISPATCH_INTERVAL_SECONDS` | `5` | Background dispatcher period |
| `MES_BATCH_SIZE` | `20` | Events per dispatch pass |
| `MES_SITE_ID` / `MES_LINE_ID` / `MES_STATION_ID` | – | Identifiers written into every event (`source`) |

`MES_*` values belong in `.env` (git-ignored) — see `.env.example`.

## Event contract

`POST <MES_WEBHOOK_URL>` with `Content-Type: application/json`:

```jsonc
{
  "event": "inspection.completed",
  "event_id": "6b1c…",                        // == inspection id (idempotency key)
  "occurred_at": "2026-11-16T09:12:44.318Z",
  "source": { "system": "Cast Metal 3D Defect Inspection System", "version": "3.2.0",
              "site": "plant-1", "line": "L2", "station": "ST-07" },
  "part":   { "part_id": "SCAN-20261116-6B1C0A", "part_type": "bracket", "batch_id": "B-1042",
              "shift": "A", "operator_id": "OP-17", "filename": "bracket.ply",
              "file_sha256": "…", "input_type": "scan3d" },
  "result": { "status": "FAIL", "verdict": "REJECT",       // ACCEPT|REJECT|HOLD|NO_DECISION
              "predicted_class": "porosity", "confidence": 0.968,
              "confidence_threshold": 0.75,
              "probabilities": { "ok": 0.032, "porosity": 0.968 },
              "model_name": "EfficientNetB0-depth", "model_version": "a1b2c3d4e5f6",
              "localization": { "available": false } },
  "inspection": { "latency_ms": 812.4, "timings": { "…": 0 }, "metrics": { "…": 0 }, "warnings": [] }
}
```

* `event_id` is the inspection id, so the MES can de-duplicate retries.
* `verdict` is the process decision: `PASS→ACCEPT`, `FAIL→REJECT`, `REVIEW→HOLD`,
  `UNCLASSIFIED→NO_DECISION`. `UNCLASSIFIED` never carries a class or confidence.
* Alerts are pushed as a second event type:

```jsonc
{ "event": "quality.alert", "event_id": "<alert id>", "occurred_at": "…",
  "source": { "…": "…" },
  "alert": { "code": "DEFECT_RATE_SPIKE", "severity": "CRITICAL",
             "title": "Defect rate 40.0% in the last 30 min", "message": "…",
             "scan_id": "6b1c…", "part_type": "bracket", "batch_id": "B-1042",
             "details": { "window_minutes": 30, "classified": 10, "failed": 4,
                          "defect_rate_pct": 40.0 } } }
```

Any `2xx` answer counts as delivered; the response body is not interpreted. `4xx`/`5xx` and
network errors are retried.

Preview the exact body for a stored inspection without sending anything:

```bash
curl http://localhost:8000/api/mes/payload/<scan_id>
```

## Alert rules

| Code | Severity | Trigger | De-duplicated |
|---|---|---|---|
| `DEFECT_DETECTED` | WARNING | stored as `FAIL` (`ALERT_ON_EACH_DEFECT`) | no (one per rejected part) |
| `REVIEW_REQUIRED` | INFO | stored as `REVIEW` (confidence below `CONFIDENCE_THRESHOLD`) | no |
| `CONSECUTIVE_DEFECTS` | CRITICAL | the last `ALERT_CONSECUTIVE_DEFECTS` inspections (default 3) are all `FAIL` | no (fires on each new link in the streak) |
| `DEFECT_RATE_SPIKE` | CRITICAL | defect rate in the last `ALERT_RATE_WINDOW_MINUTES` (30) ≥ `ALERT_RATE_THRESHOLD_PCT` (25 %), with at least `ALERT_RATE_MIN_SAMPLES` (10) classified scans | yes, `ALERT_DEDUP_MINUTES` |
| `MODEL_UNAVAILABLE` | WARNING | stored as `UNCLASSIFIED` because no model is loaded | yes |
| `MES_DELIVERY_FAILED` | WARNING | an event exhausted `MES_MAX_ATTEMPTS` | no |

Alerts appear on the WebSocket as `{"type": "alert", "alert": {…}}`, in `GET /api/alerts`, and as
`alerts[]` in the upload response. Acknowledge with
`POST /api/alerts/{id}/acknowledge?by=<name>` (records who and when).

## Testing it end to end

No real MES is needed — `tools/mes_stub_server.py` is a stand-in that prints events and answers
`202`:

```bash
# 1. stand-in MES (prints every event; --log writes them to JSONL for inspection)
python -m tools.mes_stub_server --port 8099 --log mes_events.jsonl

# 2. the app, with the integration enabled
MES_ENABLED=true MES_WEBHOOK_URL=http://127.0.0.1:8099/api/quality/inspections \
  MES_API_KEY=local-test python -m uvicorn backend.app:app --port 8000

# 3. produce inspections (or upload from the dashboard)
python -m tools.generate_test_scans --out test_scans
curl -F "file=@test_scans/bracket_porosity.ply" -F "part_type=bracket" -F "batch_id=B-1" \
     http://localhost:8000/api/scans/upload

# 4. inspect the result
curl http://localhost:8000/api/mes/status
curl http://localhost:8000/api/mes/outbox?include_payload=true
curl http://localhost:8000/api/alerts
```

Failure path: run the stub with `--fail`, upload, then watch `attempts` grow in
`/api/mes/outbox`, the event end as `DEAD`, and `MES_DELIVERY_FAILED` appear in `/api/alerts`.
`POST /api/mes/outbox/{id}/retry` re-queues it with a fresh attempt budget, and
`POST /api/mes/dispatch` flushes the queue immediately instead of waiting for the background worker.

## Limitations

* The HTTP transport (`send_http`) was exercised against `tools/mes_stub_server.py`; **no real
  MES/ERP product has been integrated or certified**. Field names are documented above and in
  `docs/API.md`, so an adapter on the MES side is normally a small mapping job.
* Delivery is at-least-once: a timeout after the MES accepted the event causes a retry. Use
  `event_id` for idempotency on the MES side.
* The outbox is *not* pruned automatically (same policy as inspection records): rows are kept for
  traceability. A retention job is not implemented.
* Alerts reflect the classifier's decisions, so their quality is bounded by the model's
  precision/recall (see `models/evaluation_report.md`). They are not a substitute for a human
  first-article inspection.
