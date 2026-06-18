"""Harmonic source extraction for chord analysis."""
from __future__ import annotations

import logging

import numpy as np

from stem_separation import separate_stems

logger = logging.getLogger(__name__)


def extract_harmonic(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Return a harmony-heavy mono signal for chroma extraction.

    Uses stem separation (other + bass) — Chord AI-style prep.
    Set CHORD_USE_DEMUCS=1 for Demucs stems when installed.
    """
    return separate_stems(y, sr).chord_signal
