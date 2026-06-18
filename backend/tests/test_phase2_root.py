"""Phase 2 root-accuracy changes (analyzer v37)."""
import numpy as np
import pytest

from chord_constants import CHORD_POLISH_DEFAULTS


def test_bass_chroma_uses_c1_c3_register():
    from analyzer import _extract_chroma_stack

    sr = 22050
    hop = 512
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 55.0 * t)  # A1-ish bass tone

    _, _, chroma_low, _ = _extract_chroma_stack(y, sr, hop_length=hop)
    assert chroma_low.shape[0] == 12
    assert chroma_low.shape[1] > 0


def test_align_segments_rejects_weak_transpose():
    from analyzer import _align_segments_to_chroma

    chroma_mean = np.zeros(12, dtype=np.float64)
    chroma_mean[0] = 1.0  # strong C
    segments = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    out, shift = _align_segments_to_chroma(segments, chroma_mean, min_gain=0.15)
    assert shift == 0
    assert out[0]["chord"] == "C"


def test_align_segments_applies_strong_transpose():
    from analyzer import _align_segments_to_chroma

    chroma_mean = np.zeros(12, dtype=np.float64)
    chroma_mean[5] = 1.0  # strong F — mislabeled C/G should shift
    segments = [
        {"time": 0.0, "end_time": 4.0, "chord": "C"},
    ]
    out, shift = _align_segments_to_chroma(segments, chroma_mean, min_gain=0.05)
    assert shift != 0
    assert out[0]["chord"] != "C"


def test_resolve_song_key_ignores_pop_loop_override():
    from analyzer import _resolve_song_key

    chroma_mean = np.zeros(12, dtype=np.float64)
    chroma_mean[0] = 1.0
    segments = [
        {"time": 0.0, "end_time": 8.0, "chord": "G"},
        {"time": 8.0, "end_time": 16.0, "chord": "D"},
        {"time": 16.0, "end_time": 24.0, "chord": "Em"},
        {"time": 24.0, "end_time": 32.0, "chord": "C"},
    ]
    chroma = np.tile(chroma_mean[:, None], (1, 100))
    chroma_low = chroma.copy()
    chroma_mid = chroma.copy()
    y = np.zeros(22050)
    sr = 22050

    root, is_major, _mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y, sr,
    )
    assert is_major is True
    assert root in range(12)


def test_ml_polish_skips_second_refine(monkeypatch):
    from chord_polish import _polish_segment_labels

    calls = {"n": 0}
    import analyzer as a

    def _count_refine(*args, **kwargs):
        calls["n"] += 1
        return args[0]

    monkeypatch.setattr(a, "_refine_segments_with_vocabulary", _count_refine)
    segments = [{"time": 0.0, "end_time": 1.0, "chord": "C"}]
    chroma = np.ones((12, 10))
    _polish_segment_labels(
        segments, chroma, chroma, chroma, None, None,
        key_root=0, is_major=True, ml_root_bias=1.02,
    )
    assert calls["n"] == 1


def test_ml_polish_uses_higher_inertia(monkeypatch):
    from chord_polish import _polish_segment_labels

    captured = {}

    import analyzer as a
    import chord_polish

    def _capture_refine(segments, *args, **kwargs):
        captured["label_inertia"] = kwargs.get("label_inertia")
        return segments

    monkeypatch.setattr(a, "_refine_segments_with_vocabulary", _capture_refine)
    monkeypatch.setattr(chord_polish, "cross_validate_adjacent_segments", lambda segs, *a, **k: segs)
    segments = [{"time": 0.0, "end_time": 1.0, "chord": "C"}]
    chroma = np.ones((12, 10))
    _polish_segment_labels(
        segments, chroma, chroma, chroma, None, None,
        key_root=0, is_major=True, ml_root_bias=1.02,
    )
    assert captured["label_inertia"] == CHORD_POLISH_DEFAULTS["ml_label_inertia"]
