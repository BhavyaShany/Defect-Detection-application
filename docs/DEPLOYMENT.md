# Deployment guide

> **Status:** the Docker/Caddy kit below was written but **could not be executed** in the build environment (no Docker).
> Track 0 (single-process mode) uses only code that is covered by tests. Do Track 0 first; if it works, Track 2 only adds packaging.
> The app has **no user accounts**. The kit protects it with one shared password at the proxy. Use sample data only.

## How the pieces fit
The backend can serve the built dashboard itself, so **one process = one URL** (dashboard, REST API, WebSocket). Everything below is a
different way of making that one process reachable.

| Track | What you get | Effort | Use it for |
|---|---|---|---|
| 0. Single-process mode | `http://localhost:8000`, whole product on one port | 5 min | Rehearsal, the backup plan on demo day |
| 1. Laptop + tunnel | A public `https://…` link to your laptop, free | 10 min | Showing someone remotely today; not "deployed" |
| **2. Cloud VM (Docker + Caddy)** | **A real server, HTTPS, password, survives restarts** | **1-2 h first time** | **The deployment your professor asks for** |
| 3. Free container hosts | e.g. Hugging Face Spaces (Docker) | untested | Zero-cost alternative; see notes |

## Track 0 - single-process mode (do this first)
```powershell
cd frontend
npm run build            # creates frontend/dist
cd ..
uvicorn backend.app:app --port 8000
```
Open **http://localhost:8000** (not 5173). The dashboard now loads from the backend; `/docs` and `/api/health` still work.
If `frontend/dist` does not exist, the backend falls back to development mode (`/` returns JSON).

## Track 1 - laptop + tunnel (quick sharing)
With Track 0 running, in a second terminal (install once: `winget install Cloudflare.cloudflared`):
```powershell
cloudflared tunnel --url http://localhost:8000
```
It prints a `https://<random>.trycloudflare.com` link. Anyone with the link can use your app **without a password** while your laptop and server are running.
Send the link only to the people you present to and stop the tunnel (Ctrl+C) afterwards. The link changes on every run.
(ngrok works the same way: `ngrok http 8000`, but needs an account.)

## Track 2 - cloud VM with Docker + Caddy (the real deployment)

### What you need
* A Linux VM (Ubuntu 22.04/24.04), **2 vCPU and 4 GB RAM or more** (TensorFlow needs ~2 GB), a public IP, ports **22, 80, 443** open in the provider's firewall.
  Any provider works. Check student credits first (GitHub Student Developer Pack, Azure for Students, etc. - offers change, verify current terms).
  Otherwise a small VM costs roughly the price of a few coffees per month; **delete the VM after the presentation** to stop charges.
* Your trained model bundle on your PC: `models/defect_model.keras`, `models/labels.json`, `models/ood.npz`.
* The exact TensorFlow version from `models/evaluation_report.json` -> `environment.tensorflow`.

### Steps
1. **Create the VM**, note its public IP (example: `203.0.113.10`), and connect: `ssh <user>@203.0.113.10`.
2. **Install Docker** on the VM:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER      # then log out and back in
   ```
3. **Get the code** onto the VM. Either `git clone <your private repo>` (needs a GitHub personal access token as password) or, from your PC,
   zip the project without `.venv`, `node_modules`, `data` and `models`, `scp` it up and unzip.
4. **Upload the model bundle** (from your PC; adjust user/IP/path):
   ```powershell
   scp models\defect_model.keras models\labels.json models\ood.npz <user>@203.0.113.10:~/defect-detection-app/models/
   ```
   All three files must come from the same training run (`ood.npz` is tied to the model; a mismatch disables the input check and shows a banner).
5. **Configure** on the VM, inside the project folder:
   ```bash
   cp deploy.env.example deploy.env
   nano deploy.env
   ```
   Set `TENSORFLOW_VERSION` (exact) and `SITE_ADDRESS` (`203-0-113-10.sslip.io` for IP `203.0.113.10`, or your own domain pointing at the IP).
6. **Set the demo password.** Generate a hash and paste it into `deploy/Caddyfile` in place of `REPLACE_WITH_BCRYPT_HASH`:
   ```bash
   docker run --rm caddy:2-alpine caddy hash-password --plaintext 'choose-a-password'
   nano deploy/Caddyfile
   ```
   The login will be user `demo` and your password. Share it only with the people you present to.
7. **Start it** (first build takes several minutes because TensorFlow is installed):
   ```bash
   docker compose --env-file deploy.env up -d --build
   docker compose logs -f app        # wait for: "Model ready: casting-photo-v1 (...)"  (Ctrl+C to leave the log)
   ```
8. **Open** `https://203-0-113-10.sslip.io`, log in, and run the checks in `docs/ACCEPTANCE_TEST.md` (section 3) against the live URL.
   The header must show *Model: casting-photo-v1* and no yellow banner (input check active).

### Operating it
| Task | Command (in the project folder on the VM) |
|---|---|
| See status / logs | `docker compose ps` · `docker compose logs --tail 100 app` |
| Update the code | `git pull && docker compose --env-file deploy.env up -d --build` |
| Install a new model | copy the three files into `models/`, then `docker compose restart app` |
| Stop (keeps data) | `docker compose down` |
| Wipe all inspection data | `docker compose down -v` (also deletes certificates) |

Inspection history and stored geometry live in the Docker volume `appdata`, so they survive restarts and rebuilds.

### Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Browser certificate error | Ports 80/443 not open at the provider, or `SITE_ADDRESS` does not point at this IP. `docker compose logs caddy` shows the reason |
| Caddy exits immediately | The password hash in `deploy/Caddyfile` is still the placeholder or was pasted wrongly |
| `model_loaded: false` | `models/` files missing on the VM, or `TENSORFLOW_VERSION` differs from training. `docker compose logs app` prints the exact reason |
| Out-of-memory / container restarts | VM has less than ~3 GB RAM; use a bigger size |
| Live updates show "reconnecting" | Proxy blocks WebSockets (not an issue with Caddy) or the app restarted |
| Build fails at `npm ci` | Network problem on the VM; retry |

### Security reality check
Shared password + HTTPS is acceptable for a supervised demo with sample images. It is **not** acceptable for customer data: there are no user accounts,
roles, audit logs, rate limiting, backups or retention policy yet. Do not let a company upload their own production images to this instance.

## Track 3 - free container hosts (untested)
Hugging Face Spaces (Docker SDK) and similar hosts can run the same `Dockerfile` (they set `PORT`, which the image honours). Caveats: free tiers sleep when idle
(first request is slow), storage is usually not persistent, and they have no volumes, so the model must be baked into the image: delete the `models` line in `.dockerignore`
and keep the repository **private**. Use the Spaces build logs to see failures. I could not verify the current limits; read the host's documentation first.

## Publishing the code on GitHub
```powershell
git init
git add .
git status          # CHECK: no .env, deploy.env, .venv, node_modules, data/, dataset/, test_scans/, *.keras
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Create the repository **Private** and empty (no README/licence) on github.com first. Never commit secrets (if you do, change them: Git keeps history).
GitHub rejects files over 100 MB; keep `*.keras` out of the repository (the `.gitignore` already does) and move the model with `scp` as in step 4, or use Git LFS.
Do not add an open-source licence if you intend to sell the product; check the Kaggle dataset licence before commercial use.
