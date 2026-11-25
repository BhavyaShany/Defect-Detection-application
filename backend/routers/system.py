from fastapi import APIRouter

from backend.config import settings
from backend.database.database import db_info
from backend.services.geometry import HAS_OPEN3D
from backend.services.runtime import inference_engine
from backend.services.ws_manager import manager
from backend.routers.scans import accepted_extensions

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/status")
def system_status():
    """Real runtime status: model state, database backend, optional accelerators, limits."""
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": inference_engine.info(),
        "database": db_info(),
        "open3d_available": HAS_OPEN3D,
        "websocket_clients": manager.count,
        "limits": {
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "allowed_extensions": accepted_extensions(),
            "image_inspection": inference_engine.input_kind == "image",
            "max_view_points": settings.MAX_VIEW_POINTS,
        },
        "alignment_mode": settings.ALIGNMENT_MODE,
    }
