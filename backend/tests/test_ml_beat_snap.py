"""Tests for ML beat-grid boundary snapping."""
import numpy as np

from presentation_timeline import snap_timeline_to_beats


def test_snap_collapses_subbeat_boundaries():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    segments = [
        {"time": 0.05, "end_time": 0.48, "chord": "C", "confidence": 0.7},
        {"time": 0.52, "end_time": 0.98, "chord": "C", "confidence": 0.6},
        {"time": 1.02, "end_time": 2.5, "chord": "G", "confidence": 0.8},
    ]
    out = snap_timeline_to_beats(segments, beats)
    assert len(out) <= 2
    assert out[0]["chord"] in ("C", "G")
