"""Pitch normalization for chord detection (YouTube / streaming detune)."""
from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Ignore shifts smaller than ~5 cents; cap at ±1 semitone.
MIN_SHIFT_SEMITONES = 0.05
MAX_SHIFT_SEMITONES = 1.0


def estimate_pitch_shift_semitones(y: np.ndarray, sr: int) -> float:
    """
    Estimate global tuning offset in semitones (librosa chroma bin units).

    Positive values mean the recording is sharp vs equal temperament.
    """
    if y is None or len(y) < sr // 4:
        return 0.0

    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception as exc:
        logger.debug("Tuning estimation failed: %s", exc)
        return 0.0

    if not np.isfinite(tuning):
        return 0.0

    return float(np.clip(tuning, -MAX_SHIFT_SEMITONES, MAX_SHIFT_SEMITONES))


def pitch_shift_audio(y: np.ndarray, sr: int, shift: float) -> np.ndarray | None:
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-shift)
    except Exception:
        return None


def normalize_pitch(
    y: np.ndarray,
    sr: int,
    *,
    max_shift: float = MAX_SHIFT_SEMITONES,
) -> tuple[np.ndarray, float]:
    """
    Pitch-shift audio toward concert tuning when detune is detected.

    Returns (audio, applied_shift_semitones). Shift is 0 when negligible.
    ponytail: no reliability gate here — the ML path measures both branches
    (chord_engine_ml) instead of predicting; this blind path only serves the
    classic fallback engine.
    """
    shift = estimate_pitch_shift_semitones(y, sr)
    shift = float(np.clip(shift, -max_shift, max_shift))

    if abs(shift) < MIN_SHIFT_SEMITONES:
        return y, 0.0

    corrected = pitch_shift_audio(y, sr, shift)
    if corrected is None:
        return y, 0.0

    logger.info("Applied pitch correction: %.2f semitones", -shift)
    return corrected, -shift
