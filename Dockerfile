# One image = dashboard + API + inference. Build:  docker compose --env-file deploy.env up -d --build
# --- stage 1: build the React dashboard ------------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- stage 2: Python backend (serves the dashboard from the same origin) ---------------------
FROM python:3.11-slim
# Must be compatible with the version the model was trained with (models/evaluation_report.json -> environment).
# Defaults to a version verified to load the bundled model, so hosts that build without build args still
# get a working image; override with --build-arg TENSORFLOW_VERSION=x.y.z (compose does this from deploy.env).
ARG TENSORFLOW_VERSION=2.20.0
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -n "$TENSORFLOW_VERSION" ]; then pip install --no-cache-dir "tensorflow-cpu==$TENSORFLOW_VERSION"; fi
COPY backend ./backend
COPY models ./models
COPY --from=web /fe/dist ./frontend/dist
RUN useradd -m app && mkdir -p /app/data /app/models && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://localhost:%s/api/health'%os.environ.get('PORT','8000'))"
# Single worker on purpose: live-update connections are kept in process memory.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
