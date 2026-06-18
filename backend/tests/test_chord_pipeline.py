"""Tests for Chord AI-style pipeline (stems, beats, bar decode)."""
import numpy as np

from bar_decode import (
    apply_bar_decode,
    collapse_timeline_to_bars,
    quantize_segment_starts_to_downbeats,
    should_use_strict_collapse,
)
from beat_tracking import track_beats
from chord_pipeline import build_chord_pipeline_context
from stem_separation import separate_stems


def _sine_chord(sr=22050, duration=2.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # A major-ish: 110, 138, 165 Hz
    y = 0.3 * np.sin(2 * np.pi * 110 * t)
    y += 0.25 * np.sin(2 * np.pi * 138 * t)
    y += 0.2 * np.sin(2 * np.pi * 165 * t)
    return y.astype(np.float32)


def test_hpss_stems_shape():
    y = _sine_chord()
    stems = separate_stems(y, 22050)
    assert stems.full.shape == y.shape
    assert stems.method in ("hpss", "demucs")
    assert stems.chord_signal.shape == y.shape


def test_beat_grid_has_downbeats():
    y = _sine_chord(duration=4.0)
    stems = separate_stems(y, 22050)
    grid = track_beats(stems.chord_signal, stems.bass, 22050)
    assert len(grid.beat_times) >= 4
    assert len(grid.downbeat_times) >= 1
    assert grid.beat_duration > 0


def test_build_pipeline_context():
    y = _sine_chord(duration=3.0)
    ctx = build_chord_pipeline_context(y, 22050)
    assert ctx.y_chord.shape == y.shape
    assert len(ctx.beat_times) >= 2
    assert len(ctx.downbeat_times) >= 1


def test_bar_quantize_snaps_to_downbeats():
    downbeats = np.array([0.0, 2.0, 4.0])
    segments = [
        {"time": 0.2, "end_time": 1.0, "chord": "A"},
        {"time": 2.1, "end_time": 4.0, "chord": "E"},
    ]
    out = quantize_segment_starts_to_downbeats(segments, downbeats, beat_duration=0.5)
    assert out[0]["time"] == 0.0
    assert out[1]["time"] == 2.0


def test_apply_bar_decode_merges_flicker():
    import bar_decode as bd

    downbeats = np.array([0.0, 2.0, 4.0])

    def score_fn(start, end, candidates):
        return candidates[0], 1.0, candidates

    segments = [
        {"time": 0.0, "end_time": 0.5, "chord": "A", "confidence": 0.7},
        {"time": 0.5, "end_time": 2.0, "chord": "A", "confidence": 0.6},
        {"time": 2.0, "end_time": 4.0, "chord": "E", "confidence": 0.8},
    ]
    old = bd.BAR_DECODE_STRICT
    try:
        bd.BAR_DECODE_STRICT = False
        out = apply_bar_decode(segments, downbeats, beat_duration=0.5, score_fn=score_fn)
        assert len(out) <= 2
    finally:
        bd.BAR_DECODE_STRICT = old


def test_strict_collapse_only_when_oversegmented():
    downbeats = np.array([0.0, 2.0, 4.0, 6.0])
    few_segments = [
        {"time": 0.0, "end_time": 2.0, "chord": "A"},
        {"time": 2.0, "end_time": 6.0, "chord": "E"},
    ]
    many_segments = [
        {"time": i * 0.5, "end_time": (i + 1) * 0.5, "chord": "A" if i % 2 == 0 else "E"}
        for i in range(10)
    ]
    assert not should_use_strict_collapse(few_segments, downbeats)
    assert should_use_strict_collapse(many_segments, downbeats)


def test_ensemble_alternate_adds_bar_candidates():
    downbeats = np.array([0.0, 2.0, 4.0])

    def score_fn(start, end, candidates):
        if "E" in candidates:
            return "E", 1.0, candidates
        return candidates[0], 0.5, candidates

    primary = [
        {"time": 0.0, "end_time": 2.0, "chord": "A", "confidence": 0.8},
        {"time": 2.0, "end_time": 4.0, "chord": "A", "confidence": 0.8},
    ]
    alternate = [
        {"time": 0.0, "end_time": 2.0, "chord": "E", "confidence": 0.7},
        {"time": 2.0, "end_time": 4.0, "chord": "A", "confidence": 0.7},
    ]
    out = collapse_timeline_to_bars(
        primary, downbeats, beat_duration=0.5, score_fn=score_fn,
        alternate_segments=alternate,
    )
    assert out[0]["chord"] == "E"
    assert out[1]["chord"] == "A"


def test_strict_bar_decode_one_chord_per_bar():
    import bar_decode as bd

    downbeats = np.array([0.0, 2.0, 4.0])

    def score_fn(start, end, candidates):
        if "E" in candidates and "A" in candidates:
            return "A", 0.80, candidates
        return candidates[0], 1.0, candidates

    segments = [
        {"time": 0.0, "end_time": 0.5, "chord": "A", "confidence": 0.7},
        {"time": 0.5, "end_time": 1.0, "chord": "E", "confidence": 0.9},
        {"time": 1.0, "end_time": 2.0, "chord": "A", "confidence": 0.6},
        {"time": 2.0, "end_time": 4.0, "chord": "E", "confidence": 0.8},
    ]
    old = bd.BAR_DECODE_STRICT
    try:
        bd.BAR_DECODE_STRICT = True
        out = bd.collapse_timeline_to_bars(
            segments, downbeats, beat_duration=0.5, score_fn=score_fn,
        )
        assert len(out) == 2
        assert out[0]["chord"] == "A"
        assert out[1]["chord"] == "E"
    finally:
        bd.BAR_DECODE_STRICT = old
