"""FastAPI application entry point.

Run from the project root:   uvicorn backend.app:app --reload --port 8000
"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.database import init_db
from backend.routers import metrics, scans, system
from backend.services.origin import origin_allowed
from backend.services.runtime import inference_engine
from backend.services.ws_manager import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("defect.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s v%s", settings.PROJECT_NAME, settings.VERSION)
    init_db()  # raises with a clear message if the database is unusable
    await run_in_threadpool(inference_engine.load)  # never raises; sets .ready / .error
    if inference_engine.ready:
        log.info("Model ready: %s (%s)", inference_engine.meta["model_name"], inference_engine.model_version)
    else:
        log.warning("Model NOT loaded: %s -- scans will be stored as UNCLASSIFIED.", inference_engine.error)
    yield
    log.info("Shutting down")


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan,
              docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(scans.router, prefix=settings.API_PREFIX)
app.include_router(metrics.router, prefix=settings.API_PREFIX)
app.include_router(system.router, prefix=settings.API_PREFIX)

# Backward-compatible upload route from v2
app.add_api_route("/scan/upload", scans.upload_scan, methods=["POST"], status_code=201,
                  include_in_schema=False)


@app.exception_handler(Exception)
async def unhandled_exception(request, exc):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": {"code": "INTERNAL_ERROR",
                                                              "message": "Unexpected server error."}})


@app.websocket("/ws/live")
@app.websocket("/api/ws/live")
async def live_feed(websocket: WebSocket):
    if not origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host"), settings.cors_origins):
        await websocket.close(code=1008)  # policy violation: origin not allowed
        return
    await manager.connect(websocket, settings.VERSION)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                if json.loads(text).get("type") == "ping":
                    await manager.pong(websocket)
            except (ValueError, AttributeError):
                pass  # ignore non-JSON chatter
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WebSocket error")
    finally:
        manager.disconnect(websocket)


def health_check():
    info = inference_engine.info()
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_docs": "/docs",
        "model_loaded": info["ready"],
        "model_error": info["error"],
        "classes": info["class_names"],
    }


app.add_api_route("/api/health", health_check, methods=["GET"], tags=["System"])

# Serve the built dashboard from the same process/origin when it exists (production, demos, tunnels).
# Otherwise "/" stays the plain JSON health check used in development (dashboard on its own dev server).
_dist = settings.frontend_dist
if (_dist / "index.html").exists():
    log.info("Serving dashboard from %s", _dist)
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="dashboard")
else:
    app.add_api_route("/", health_check, methods=["GET"], include_in_schema=False)
