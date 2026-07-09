"""
ML chord engine — lv-chordia raw segments + key metadata (pitch applied upstream).
"""
import logging
import os

import numpy as np

from analyzer import (
    _build_key_info,
    _extract_chroma_stack,
    _resolve_song_key,
)
from chord_engine_chordia import (
    chordia_confidence,
    decode_chordia_probs,
    recognize_chordia_probs,
)
from pitch_utils import (
    MIN_SHIFT_SEMITONES,
    estimate_pitch_shift_semitones,
    pitch_shift_audio,
)

logger = logging.getLogger(__name__)

from chord_model_registry import get_chord_model, validate_chord_model

# Indirection points for tests; production uses the real functions.
_recognize_probs = recognize_chordia_probs
_decode_probs = decode_chordia_probs
_estimate_shift = estimate_pitch_shift_semitones
_shift_audio = pitch_shift_audio


def get_chord_ml_model() -> str:
    """Return ML model id — always chordia for serving."""
    return validate_chord_model(get_chord_model())


def _chordia_segments_pitch_selected(y_chord, sr):
    """
    Decode-both pitch handling (v50): measure which branch decodes better
    instead of predicting reliability (the v49 gate this replaces cost 1.3pp).

    Returns (raw_segments, applied_shift_semitones).
    """
    mode = os.getenv("CHORD_PITCH_SELECT", "confidence").lower().strip()
    max_candidate = float(os.getenv("CHORD_PITCH_MAX_CANDIDATE_SHIFT", "0.40"))
    shift = _estimate_shift(y_chord, sr)
    probs_raw, hmm_raw, entry_raw = _recognize_probs(y_chord, sr)

    # ponytail: cap tuned on 3 measured cases across both gold splits — real
    # varispeed detunes at 0.37-0.38st (corrections worth +13/+20pp),
    # estimator error at 0.45st (bass-only track, correction cost -11pp).
    # 0.05st of headroom to the known-bad case; raise/lower via env as new
    # cases appear.
    if (
        mode == "off"
        or abs(shift) < MIN_SHIFT_SEMITONES
        or abs(shift) > max_candidate
    ):
        return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0

    corrected = _shift_audio(y_chord, sr, shift)
    if corrected is None:
        return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0
    probs_cor, hmm_cor, entry_cor = _recognize_probs(corrected, sr)

    if mode == "tta":
        n = min(p.shape[0] for p in (probs_raw[0], probs_cor[0]))
        avg = [
            np.mean([a[:n], b[:n]], axis=0)
            for a, b in zip(probs_raw, probs_cor)
        ]
        return _decode_probs(avg, hmm_raw, entry_raw), 0.0

    margin = float(os.getenv("CHORD_PITCH_CONF_MARGIN", "0.0"))
    conf_raw = chordia_confidence(probs_raw)
    conf_cor = chordia_confidence(probs_cor)
    if conf_cor > conf_raw + margin:
        logger.info(
            "Pitch select: corrected wins (%.4f vs %.4f, shift %.2f st)",
            conf_cor, conf_raw, -shift,
        )
        return _decode_probs(probs_cor, hmm_cor, entry_cor), -shift
    logger.info(
        "Pitch select: raw wins (%.4f vs %.4f, candidate shift %.2f st)",
        conf_raw, conf_cor, -shift,
    )
    return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0


def _attach_key_info(segments, y, sr, pipeline, y_chord=None):
    from chord_pipeline import build_chord_pipeline_context

    if pipeline is None:
        pipeline = build_chord_pipeline_context(y, sr)

    y_chord = y_chord if y_chord is not None else pipeline.y_chord
    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(
        y, sr, y_harmonic=y_chord,
    )
    chroma_mean = chroma.mean(axis=1)
    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    key_info = _build_key_info(
        key_root,
        is_major,
        mode=mode if mode not in ("major", "minor") else None,
    )
    return segments, key_info


def extract_chords_ml(y, sr, pipeline=None):
    """ML chord extraction: lv-chordia on HPSS chord stem, raw segments."""
    get_chord_ml_model()

    if pipeline is None:
        from chord_pipeline import build_chord_pipeline_context
        pipeline = build_chord_pipeline_context(y, sr)

    from analyzer import extract_chords as extract_chords_classic
    from chord_engine_chordia import chordia_to_segments

    raw_segments, applied_shift = _chordia_segments_pitch_selected(
        pipeline.y_chord, sr
    )
    raw = chordia_to_segments(raw_segments)
    if not raw:
        logger.warning("lv-chordia returned no segments; falling back to classic stem HMM")
        return extract_chords_classic(y, sr, pipeline, bar_finalize=True)

    y_chord_for_key = pipeline.y_chord
    if applied_shift:
        # key estimation should see the same (corrected) stem the labels came from
        shifted = _shift_audio(pipeline.y_chord, sr, -applied_shift)
        if shifted is not None:
            y_chord_for_key = shifted

    segments, key_info = _attach_key_info(raw, y, sr, pipeline, y_chord=y_chord_for_key)
    key_info = dict(key_info or {})
    key_info["pitch_select_mode"] = os.getenv("CHORD_PITCH_SELECT", "confidence").lower().strip()
    if applied_shift:
        key_info["pitch_correction_semitones"] = round(applied_shift, 3)
    return segments, key_info
