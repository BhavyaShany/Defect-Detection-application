# Deploying to Google Cloud Run

Cloud Run runs the same `Dockerfile` used by `docker-compose.yml`; the image serves the dashboard, the
REST API and the WebSocket from one origin. It is the cheapest hosted option for this app because the
container needs ~500 MB RAM with the model loaded, which the free monthly quota covers for a demo.

## Current deployment

| | |
|---|---|
| URL | <https://defect-inspect-fjvxk6bkqq-uc.a.run.app> |
| Project | `defect-inspection-11251940` |
| Region | `us-central1` |
| Service / revision | `defect-inspect` / `defect-inspect-00001-lcr` |
| Settings | 2 GiB, 1 vCPU, concurrency 8, max 1 instance, min 0, timeout 3600 s, unauthenticated |

Billing account `011535-2B4BC2-D99452` is linked to the project; the APIs below are enabled. To update
the running service, re-run the deploy command with the same flags.

## What you get, and what you do not
* A public HTTPS URL (`https://<service>-<hash>.<region>.run.app`) with no server to administer.
* **No persistent disk.** Cloud Run's filesystem is in-memory: the SQLite database and stored geometry
  under `/app/data` are lost whenever the instance stops (after the idle timeout or on redeploy). The
  dashboard works normally, it just starts empty again. Use the compose/VM track in
  [DEPLOYMENT.md](DEPLOYMENT.md) if inspection history must survive.
* **Cold starts.** With `--min-instances 0` the container sleeps when idle; the first request afterwards
  waits ~20-40 s while TensorFlow loads the model. Add `--min-instances 1` to keep it warm (billable).
* **One instance only.** Live WebSocket updates are held in process memory, so the service must run a
  single instance; Cloud Run is capped at `--max-instances 1` below.

## One-time setup
> **A billing account is mandatory.** Google refuses to deploy to Cloud Run on a project without billing
> enabled, even though the usage itself stays inside the free quota. Attach one at
> <https://console.cloud.google.com/billing> first; the $300 / 90-day new-account credit covers this app.
>
> Already linked for this project: `011535-2B4BC2-D99452` → `defect-inspection-11251940`
> (`gcloud billing projects link defect-inspection-11251940 --billing-account=011535-2B4BC2-D99452`).

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>          # must have billing enabled
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```
`gcloud run deploy --source .` uploads the folder, builds it with Cloud Build and deploys it. The upload
is filtered by `.gcloudignore` (the trained model in `models/` is included on purpose; `data/`,
`datasets`, `node_modules` and the local venv are not).

## Deploy / update
```bash
gcloud run deploy defect-inspect \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 \
  --port 8080 \
  --concurrency 8 --max-instances 1 --min-instances 0 \
  --timeout 3600 \
  --set-env-vars CONFIDENCE_THRESHOLD=0.75
```
* `--port 8080` matches the `PORT` Cloud Run injects; the image already honours it.
* `--timeout 3600` is required for the live feed: Cloud Run's default 300 s would drop `/ws/live`
  connections every five minutes.
* `--memory 2Gi` gives TensorFlow headroom (measured ≈ 490 MB with the model loaded, more while
  preprocessing a large scan). 1Gi works but leaves little room for concurrent uploads.

Redeploy after a code or model change with the same command; a new revision replaces the old one.

### If `gcloud` fails to load on Windows
This machine's SDK install needs its bundled libraries on the path and an interpreter it can use:
```bash
export CLOUDSDK_PYTHON="<repo>/.venv/Scripts/python.exe"
export PYTHONPATH="$HOME/AppData/Local/google-cloud-sdk/lib;$HOME/AppData/Local/google-cloud-sdk/lib/third_party"
"$CLOUDSDK_PYTHON" "$HOME/AppData/Local/google-cloud-sdk/lib/gcloud.py" run deploy defect-inspect --source . ...
```

## Verify after deploying
```bash
URL=$(gcloud run services describe defect-inspect --region us-central1 --format 'value(status.url)')
curl -s "$URL/api/health"      # expect "model_loaded": true and the two class names
```
Then open `$URL` and upload a **grayscale casting photo** in the style of the training set. Note that the
input check (`models/ood.npz`) refuses photos whose features are far from the training photos — colour
photos and phone snapshots are reported as `UNCLASSIFIED` with an explanation rather than being guessed
at. That is intended behaviour, not a deployment failure; see "Known limitations" in the README.
