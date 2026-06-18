"""Singleton ML models — avoid reloading Demucs / basic-pitch on every request."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_demucs_model: Any = None
_basic_pitch_model: Any = None


def get_demucs_model():
    global _demucs_model
    if _demucs_model is not None:
        return _demucs_model
    with _lock:
        if _demucs_model is None:
            from demucs.pretrained import get_model

            logger.info("Loading Demucs htdemucs model (cached for process lifetime)")
            model = get_model("htdemucs")
            model.eval()
            _demucs_model = model
        return _demucs_model


def get_basic_pitch_model():
    global _basic_pitch_model
    if _basic_pitch_model is not None:
        return _basic_pitch_model
    with _lock:
        if _basic_pitch_model is None:
            from basic_pitch.inference import ICASSP_2022_MODEL_PATH, Model

            logger.info("Loading basic-pitch model (cached for process lifetime)")
            _basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
        return _basic_pitch_model


def preload_ml_models() -> None:
    """Warm models at startup when deps are present (best-effort)."""
    try:
        from stem_separation import demucs_available

        if demucs_available():
            get_demucs_model()
    except Exception as exc:
        logger.debug("Demucs preload skipped: %s", exc)
    try:
        get_basic_pitch_model()
    except Exception as exc:
        logger.debug("basic-pitch preload skipped: %s", exc)
