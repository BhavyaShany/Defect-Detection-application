"""Application settings.

All values can be overridden through environment variables or a ``.env`` file in the
project root (see ``.env.example``). Relative paths are resolved against the project
root, not the current working directory, so the app behaves the same wherever it is
launched from.
"""
import os
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE, '#' comments). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw.strip())


class Settings:
    """Plain settings object populated from environment variables (see ``.env.example``)."""

    def __init__(self) -> None:
        _load_dotenv(BASE_DIR / ".env")
        e = _env
        self.PROJECT_NAME: str = "Cast Metal 3D Defect Inspection System"
        self.VERSION: str = "3.1.2"
        self.API_PREFIX: str = "/api"

        # ---- Database ----
        self.DATABASE_URL: str = e("DATABASE_URL", "")  # empty -> SQLite file under DATA_DIR
        self.DB_FALLBACK_SQLITE: bool = e("DB_FALLBACK_SQLITE", True, bool)
        self.DATA_DIR: str = e("DATA_DIR", "data")
        # Built dashboard (npm run build). If present, the backend serves it at "/" (single origin, one URL).
        self.FRONTEND_DIST: str = e("FRONTEND_DIST", "frontend/dist")

        # ---- CORS ----
        self.CORS_ORIGINS: str = e("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

        # ---- Trained model ----
        self.MODEL_PATH: str = e("MODEL_PATH", "models/defect_model.keras")
        self.MODEL_LABELS_PATH: str = e("MODEL_LABELS_PATH", "models/labels.json")
        self.CONFIDENCE_THRESHOLD: float = e("CONFIDENCE_THRESHOLD", 0.75, float)
        self.OOD_PATH: str = e("OOD_PATH", "models/ood.npz")  # input-check statistics (python -m ml.fit_ood)
        self.OOD_COLOR_CHECK: bool = e("OOD_COLOR_CHECK", True, bool)

        # ---- Uploads ----
        self.MAX_UPLOAD_MB: int = e("MAX_UPLOAD_MB", 100, int)
        self.ALLOWED_EXTENSIONS: str = e("ALLOWED_EXTENSIONS", ".ply,.pcd,.stl,.obj")
        self.IMAGE_EXTENSIONS: str = e("IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.bmp")  # only accepted by image-mode models
        self.MAX_IMAGE_PIXELS: int = e("MAX_IMAGE_PIXELS", 40_000_000, int)  # decompression-bomb guard
        self.MAX_CONCURRENT_INSPECTIONS: int = e("MAX_CONCURRENT_INSPECTIONS", 2, int)
        self.MIN_POINTS: int = e("MIN_POINTS", 200, int)

        # ---- Preprocessing ----
        self.MAX_POINTS_PROCESSING: int = e("MAX_POINTS_PROCESSING", 1_500_000, int)  # subsample above
        self.MAX_VIEW_POINTS: int = e("MAX_VIEW_POINTS", 60_000, int)  # points sent to the viewer
        self.MAX_VIEW_FACES: int = e("MAX_VIEW_FACES", 150_000, int)
        self.OUTLIER_NB_NEIGHBORS: int = e("OUTLIER_NB_NEIGHBORS", 20, int)
        self.OUTLIER_STD_RATIO: float = e("OUTLIER_STD_RATIO", 2.0, float)
        self.SURFACE_VARIATION_K: int = e("SURFACE_VARIATION_K", 16, int)
        self.ALIGNMENT_MODE: str = e("ALIGNMENT_MODE", "pca")  # none | pca | icp
        self.REFERENCE_SCAN_PATH: str = e("REFERENCE_SCAN_PATH", "")
        self.ICP_MAX_CORRESPONDENCE_DISTANCE: float = e("ICP_MAX_CORRESPONDENCE_DISTANCE", 0.05, float)
        self.ICP_MAX_ITERATIONS: int = e("ICP_MAX_ITERATIONS", 50, int)
        self.PROJECTION_RESOLUTION: int = e("PROJECTION_RESOLUTION", 224, int)
        self.MIN_PROJECTION_COVERAGE: float = e("MIN_PROJECTION_COVERAGE", 0.02, float)

    # ---- Derived helpers ----
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return _resolve(self.DATA_DIR)

    @property
    def frontend_dist(self) -> Path:
        return _resolve(self.FRONTEND_DIST)

    @property
    def geometry_dir(self) -> Path:
        return self.data_dir / "geometry"

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL.strip():
            return self.DATABASE_URL.strip()
        return f"sqlite:///{(self.data_dir / 'defect_inspection.db').as_posix()}"

    @property
    def sqlite_fallback_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'defect_inspection.db').as_posix()}"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()]

    @property
    def image_extensions(self) -> List[str]:
        return [e.strip().lower() for e in self.IMAGE_EXTENSIONS.split(",") if e.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.MAX_UPLOAD_MB) * 1024 * 1024

    @property
    def model_path(self) -> Path:
        return _resolve(self.MODEL_PATH)

    @property
    def labels_path(self) -> Path:
        return _resolve(self.MODEL_LABELS_PATH)

    @property
    def ood_path(self) -> Path:
        return _resolve(self.OOD_PATH)

    @property
    def reference_scan_path(self) -> Optional[Path]:
        return _resolve(self.REFERENCE_SCAN_PATH) if self.REFERENCE_SCAN_PATH.strip() else None


settings = Settings()
