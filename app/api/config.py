from __future__ import annotations

from fastapi import APIRouter

from app.core.config import DEFAULT_PROCESSING_MODE, PROCESSING_MODES, STEM_NAMES

router = APIRouter()


@router.get("/config")
def get_config() -> dict:
    return {
        "stem_names": list(STEM_NAMES),
        "processing_modes": list(PROCESSING_MODES),
        "default_mode": DEFAULT_PROCESSING_MODE,
    }
