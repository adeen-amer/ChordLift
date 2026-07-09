"""
Chord extraction dispatcher.

Set CHORD_ENGINE=classic to force the template-based engine.
ML default: lv-chordia on HPSS chord stem + decode-both pitch selection.

Classic engine: pitch normalization runs up front. ML engine: pitch is
selected per-branch inside chord_engine_ml (see pitch_utils).
"""
import logging
import os

from engine_verify import strict_engine_checks_enabled
from ml_env import summarize_ml_readiness
from pitch_utils import normalize_pitch

logger = logging.getLogger(__name__)

CHORD_ENGINE = os.getenv("CHORD_ENGINE", "ml").lower().strip()
PITCH_CORRECT = os.getenv("CHORD_PITCH_CORRECT", "1").lower() not in ("0", "false", "no")
_ml_status_logged = False


def log_chord_engine_status():
    """Log ML readiness once at startup (see ML_SETUP.md if deps missing)."""
    global _ml_status_logged
    if _ml_status_logged:
        return
    _ml_status_logged = True
    status = summarize_ml_readiness()
    if status["engine_requested"] != "ml":
        logger.info("Chord engine: classic (CHORD_ENGINE=%s)", status["engine_requested"])
        return
    if status["ml_deps_ok"]:
        logger.info("Chord engine: ML lv-chordia — dependencies OK")
        return
    logger.error(
        "Chord engine: ML requested (%s) but missing packages %s. %s",
        status["ml_model"],
        ", ".join(status["missing_packages"]),
        status["setup_hint"],
    )


def extract_chords(y, sr, pipeline=None, *, return_pipeline=False):
    log_chord_engine_status()
    if not PITCH_CORRECT:
        os.environ["CHORD_PITCH_SELECT"] = "off"
    pitch_shift = 0.0
    if PITCH_CORRECT and CHORD_ENGINE != "ml":
        # ML path selects pitch per-branch inside extract_chords_ml (v50);
        # only the classic engine still uses whole-mix pre-correction.
        y, pitch_shift = normalize_pitch(y, sr)
        if pitch_shift:
            logger.info("Pitch-corrected audio by %.2f semitones before chord analysis", pitch_shift)

    from chord_pipeline import build_chord_pipeline_context
    pipeline = pipeline or build_chord_pipeline_context(y, sr)

    def _annotate_key_info(key_info):
        key_info = dict(key_info or {})
        key_info["stem_method"] = pipeline.stems.method
        key_info["chord_pipeline"] = "stems+beats+bars"
        if pitch_shift and "pitch_correction_semitones" not in key_info:
            key_info["pitch_correction_semitones"] = round(pitch_shift, 3)
        return key_info

    if CHORD_ENGINE == "ml":
        ml_fallback_reason = None
        try:
            from chord_engine_ml import extract_chords_ml, get_chord_ml_model
            ml_model = get_chord_ml_model()
            logger.info(
                "Using ML chord engine (%s) | stems=%s",
                ml_model,
                pipeline.stems.method,
            )
            segments, key_info = extract_chords_ml(y, sr, pipeline)
            key_info = dict(key_info or {})
            key_info["chord_engine_actual"] = "ml"
            key_info["ml_model"] = ml_model
            key_info = _annotate_key_info(key_info)
            if return_pipeline:
                return segments, key_info, pipeline
            return segments, key_info
        except Exception as exc:
            ml_fallback_reason = f"{exc.__class__.__name__}: {exc}"
            logger.error(
                "ML chord engine FAILED (CHORD_ML_MODEL=%s); falling back to classic. "
                "%s: %s",
                os.getenv("CHORD_ML_MODEL", "ensemble"),
                exc.__class__.__module__,
                exc,
                exc_info=True,
            )
            if strict_engine_checks_enabled():
                raise
            if PITCH_CORRECT:
                # ponytail: rebuild is only reached on the rare ML-failure
                # fallback path; the direct-classic path above already
                # pitch-corrected before the pipeline was first built.
                y, pitch_shift = normalize_pitch(y, sr)
                if pitch_shift:
                    logger.info("Pitch-corrected audio by %.2f semitones before chord analysis", pitch_shift)
                pipeline = build_chord_pipeline_context(y, sr)

    from analyzer import extract_chords as extract_chords_classic
    segments, key_info = extract_chords_classic(y, sr, pipeline)
    key_info = dict(key_info or {})
    key_info["chord_engine_actual"] = "classic"
    if CHORD_ENGINE == "ml":
        key_info["ml_fallback"] = True
        key_info["ml_fallback_reason"] = ml_fallback_reason or "unknown"
    key_info = _annotate_key_info(key_info)
    if return_pipeline:
        return segments, key_info, pipeline
    return segments, key_info
