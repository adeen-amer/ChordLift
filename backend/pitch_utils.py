"""Pitch normalization for chord detection (YouTube / streaming detune)."""
from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# --- Gate thresholds (tuned on gold v2; see PHASE13.md) ---
# Ignore shifts smaller than ~5 cents; cap at ±1 semitone.
MIN_SHIFT_SEMITONES = 0.05
MAX_SHIFT_SEMITONES = 1.0
# Chroma key-fit must improve by at least this much to apply correction.
MIN_KEY_SCORE_GAIN = 0.01
# Harmonic vs full-mix estimate disagreement → skip (unreliable on bass-heavy tracks).
MAX_STEM_ESTIMATE_DELTA = 0.18
# Bass-line driven mixes: weak key-fit gains are often misleading.
BASS_HEAVY_RATIO = 8.0
BASS_HEAVY_SHIFT_MIN = 0.28
BASS_HEAVY_MIN_GAIN = 0.012


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


def _chroma_key_fit_score(y: np.ndarray, sr: int) -> float:
    """Best major-key template correlation (higher = clearer tonal center)."""
    if y is None or len(y) < sr // 8:
        return 0.0
    try:
        from analyzer import MAJOR_KEY_PROFILE

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mean = np.mean(chroma, axis=1)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            return 0.0
        mean = mean / norm
        profile = MAJOR_KEY_PROFILE / (np.linalg.norm(MAJOR_KEY_PROFILE) + 1e-8)
        return max(float(np.dot(np.roll(mean, -shift), profile)) for shift in range(12))
    except Exception as exc:
        logger.debug("Chroma key-fit failed: %s", exc)
        return 0.0


def _bass_energy_ratio(y: np.ndarray, sr: int) -> float:
    """Low-band STFT energy vs mean — high on bass-line driven tracks."""
    if y is None or len(y) < sr // 8:
        return 0.0
    try:
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass = spec[freqs < 150].mean()
        return float(bass / (spec.mean() + 1e-8))
    except Exception:
        return 0.0


def _pitch_shift(y: np.ndarray, sr: int, shift: float) -> np.ndarray | None:
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-shift)
    except Exception:
        return None


def _estimate_reliable(shift: float, y: np.ndarray, sr: int, corrected: np.ndarray) -> bool:
    """Return False when correction likely hurts (e.g. another-one-bites-the-dust)."""
    if abs(shift) < MIN_SHIFT_SEMITONES:
        return False

    try:
        y_harm, _y_perc = librosa.effects.hpss(y)
        shift_harm = estimate_pitch_shift_semitones(y_harm, sr)
        if abs(shift - shift_harm) > MAX_STEM_ESTIMATE_DELTA:
            logger.info(
                "Pitch gate: full-mix vs harmonic disagree (%.2f vs %.2f st)",
                shift, shift_harm,
            )
            return False
    except Exception:
        pass

    before = _chroma_key_fit_score(y, sr)
    after = _chroma_key_fit_score(corrected, sr)
    gain = after - before
    if gain < MIN_KEY_SCORE_GAIN:
        logger.info(
            "Pitch gate: chroma key-fit did not improve (%.3f → %.3f, gain %.4f)",
            before, after, gain,
        )
        return False
    bass_ratio = _bass_energy_ratio(y, sr)
    if (
        bass_ratio > BASS_HEAVY_RATIO
        and abs(shift) > BASS_HEAVY_SHIFT_MIN
        and gain < BASS_HEAVY_MIN_GAIN
    ):
        logger.info(
            "Pitch gate: bass-heavy (ratio %.1f) weak gain %.4f",
            bass_ratio, gain,
        )
        return False
    return True


def normalize_pitch(
    y: np.ndarray,
    sr: int,
    *,
    max_shift: float = MAX_SHIFT_SEMITONES,
) -> tuple[np.ndarray, float]:
    """
    Pitch-shift audio toward concert tuning when detune is detected and reliable.

    Returns (audio, applied_shift_semitones). Shift is 0 when correction
    is negligible or unreliable.
    """
    shift = estimate_pitch_shift_semitones(y, sr)
    shift = float(np.clip(shift, -max_shift, max_shift))

    if abs(shift) < MIN_SHIFT_SEMITONES:
        return y, 0.0

    corrected = _pitch_shift(y, sr, shift)
    if corrected is None:
        return y, 0.0

    if not _estimate_reliable(shift, y, sr, corrected):
        return y, 0.0

    logger.info("Applied pitch correction: %.2f semitones", -shift)
    return corrected, -shift
