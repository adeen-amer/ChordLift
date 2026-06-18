"""Bar quantize tests — canonical implementation is bar_decode.py (Phase 4)."""
import numpy as np

from bar_decode import (
    merge_segments_within_bar,
    quantize_segment_starts_to_downbeats,
    should_apply_bar_decode,
)
from presentation_timeline import snap_timeline_to_beats


def test_quantize_snaps_to_bar_starts():
    downbeats = np.array([0.0, 2.0, 4.0])
    segments = [
        {"time": 0.1, "end_time": 1.0, "chord": "C", "confidence": 0.7},
        {"time": 1.2, "end_time": 2.0, "chord": "C", "confidence": 0.6},
        {"time": 2.1, "end_time": 4.0, "chord": "G", "confidence": 0.8},
    ]
    out = quantize_segment_starts_to_downbeats(segments, downbeats, beat_duration=0.5)
    assert out[0]["time"] == 0.0
    assert out[-1]["time"] == 2.0
    assert sorted(set(s["time"] for s in out)) == [0.0, 2.0]


def test_merge_within_bar_collapses_flicker():
    downbeats = np.array([0.0, 2.0, 4.0, 6.0])

    def score_fn(bar_start, bar_end, candidates):
        return candidates[0], 1.0, candidates

    segments = [
        {"time": 0.0, "end_time": 0.5, "chord": "C", "confidence": 0.7},
        {"time": 0.0, "end_time": 1.0, "chord": "C5", "confidence": 0.6},
        {"time": 0.0, "end_time": 2.0, "chord": "C", "confidence": 0.65},
        {"time": 2.0, "end_time": 4.0, "chord": "G", "confidence": 0.8},
    ]
    out = merge_segments_within_bar(
        segments, downbeats, beat_duration=0.5, score_fn=score_fn,
    )
    assert len(out) <= 2
    assert out[0]["chord"] in ("C", "C5", "")


def test_skip_bar_merge_for_fast_changes():
    segments = [
        {"time": 0.0, "end_time": 0.25, "chord": "F5", "confidence": 0.7},
        {"time": 0.25, "end_time": 0.5, "chord": "F5", "confidence": 0.7},
        {"time": 0.5, "end_time": 0.75, "chord": "C5", "confidence": 0.7},
        {"time": 0.75, "end_time": 1.0, "chord": "C5", "confidence": 0.7},
    ]
    assert not should_apply_bar_decode(segments, beat_duration=0.25)
    beats = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    out = snap_timeline_to_beats(segments, beats)
    assert len(out) >= 1
