"""Process-wide singletons shared by routers (no import-time model loading)."""
from backend.config import settings
from backend.services.model_inference import DefectInferenceEngine

inference_engine = DefectInferenceEngine(
    settings.model_path, settings.labels_path, settings.CONFIDENCE_THRESHOLD,
    ood_path=settings.ood_path, ood_color_check=settings.OOD_COLOR_CHECK,
)
