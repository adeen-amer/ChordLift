"""
Chord extraction dispatcher.

Set CHORD_ENGINE=classic to force the template-based engine.
ML (autochord) is the default; failures fall back to classic automatically.

Pitch normalization runs before either engine (see pitch_utils).
"""
import logging
import os

from pitch_utils import normalize_pitch

logger = logging.getLogger(__name__)

CHORD_ENGINE = os.getenv("CHORD_ENGINE", "ml").lower().strip()
PITCH_CORRECT = os.getenv("CHORD_PITCH_CORRECT", "1").lower() not in ("0", "false", "no")


def extract_chords(y, sr):
    pitch_shift = 0.0
    if PITCH_CORRECT:
        y, pitch_shift = normalize_pitch(y, sr)
        if pitch_shift:
            logger.info("Pitch-corrected audio by %.2f semitones before chord analysis", pitch_shift)

    if CHORD_ENGINE == "ml":
        try:
            from chord_engine_ml import extract_chords_ml
            logger.info("Using ML chord engine (autochord)")
            segments, key_info = extract_chords_ml(y, sr)
            if pitch_shift:
                key_info = dict(key_info or {})
                key_info["pitch_correction_semitones"] = round(pitch_shift, 3)
            return segments, key_info
        except Exception as exc:
            logger.warning("ML chord engine unavailable, using classic: %s", exc)

    from analyzer import extract_chords as extract_chords_classic
    segments, key_info = extract_chords_classic(y, sr)
    if pitch_shift:
        key_info = dict(key_info or {})
        key_info["pitch_correction_semitones"] = round(pitch_shift, 3)
    return segments, key_info
