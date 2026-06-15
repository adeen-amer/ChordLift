"""Tests for ML bar quantize + within-bar merge."""
import numpy as np

from chord_engine_ml import (
    _merge_ml_segments_within_bar,
    _quantize_ml_segments_to_bars,
    _should_use_bar_quantize,
    _smooth_ml_segment_boundaries,
    _snap_ml_segments_to_beats,
)


def test_quantize_snaps_to_bar_starts():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    segments = [
        {"time": 0.1, "end_time": 1.0, "chord": "C", "confidence": 0.7},
        {"time": 1.2, "end_time": 2.0, "chord": "C", "confidence": 0.6},
        {"time": 2.1, "end_time": 4.0, "chord": "G", "confidence": 0.8},
    ]
    out = _quantize_ml_segments_to_bars(segments, beats, beats_per_bar=4)
    assert out[0]["time"] == 0.0
    assert out[-1]["time"] == 2.0
    assert sorted(set(s["time"] for s in out)) == [0.0, 2.0]


def test_merge_within_bar_collapses_flicker():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    chroma = np.zeros((12, 100))
    chroma_low = np.zeros((12, 100))
    chroma[0, :] = 1.0
    chroma[4, :] = 0.8
    chroma[7, :] = 0.9
    chroma_low[0, :] = 1.0

    segments = [
        {"time": 0.0, "end_time": 0.5, "chord": "C", "confidence": 0.7},
        {"time": 0.0, "end_time": 1.0, "chord": "C5", "confidence": 0.6},
        {"time": 0.0, "end_time": 2.0, "chord": "C", "confidence": 0.65},
        {"time": 2.0, "end_time": 4.0, "chord": "G", "confidence": 0.8},
    ]
    out = _merge_ml_segments_within_bar(
        segments, beats, chroma, chroma_low, None, None, 22050, beat_duration=0.5,
    )
    assert len(out) <= 2
    assert out[0]["chord"] in ("C", "C5", "")


def test_skip_bar_merge_for_fast_changes():
    beats = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    segments = [
        {"time": 0.0, "end_time": 0.25, "chord": "F5", "confidence": 0.7},
        {"time": 0.25, "end_time": 0.5, "chord": "F5", "confidence": 0.7},
        {"time": 0.5, "end_time": 0.75, "chord": "C5", "confidence": 0.7},
        {"time": 0.75, "end_time": 1.0, "chord": "C5", "confidence": 0.7},
    ]
    assert not _should_use_bar_quantize(segments, beat_duration=0.25)
    chroma = np.zeros((12, 40))
    chroma_low = np.zeros((12, 40))
    beat_only = _snap_ml_segments_to_beats(segments, beats)
    out = _smooth_ml_segment_boundaries(
        segments, beats, chroma, chroma_low, None, None, 22050, beat_duration=0.25,
    )
    assert len(out) == len(beat_only)
