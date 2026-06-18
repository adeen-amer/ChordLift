"""
ML chord engine — lv-chordia raw segments + key metadata (pitch applied upstream).
"""
import logging

from analyzer import (
    _build_key_info,
    _extract_chroma_stack,
    _resolve_song_key,
)

logger = logging.getLogger(__name__)

from chord_model_registry import get_chord_model, validate_chord_model


def get_chord_ml_model() -> str:
    """Return ML model id — always chordia for serving."""
    return validate_chord_model(get_chord_model())


def _attach_key_info(segments, y, sr, pipeline):
    from chord_pipeline import build_chord_pipeline_context

    if pipeline is None:
        pipeline = build_chord_pipeline_context(y, sr)

    y_chord = pipeline.y_chord
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
    from chord_engine_chordia import chordia_to_segments, recognize_chordia

    raw = chordia_to_segments(recognize_chordia(pipeline.y_chord, sr))
    if not raw:
        logger.warning("lv-chordia returned no segments; falling back to classic stem HMM")
        return extract_chords_classic(y, sr, pipeline, bar_finalize=True)
    return _attach_key_info(raw, y, sr, pipeline)
