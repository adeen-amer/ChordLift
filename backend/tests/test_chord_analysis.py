"""Tests for chord engine diff analysis."""
from eval_chord_utils import (
    analyze_chord_prediction,
    segments_with_end_times,
    timeline_mismatch_windows,
)


def test_segments_with_end_times():
    segs = [{"time": 0.0, "chord": "C"}, {"time": 2.0, "chord": "G"}]
    out = segments_with_end_times(segs, duration=5.0)
    assert out[0]["end_time"] == 2.0
    assert out[1]["end_time"] == 5.0


def test_analyze_perfect_match():
    segs = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    changes = [{"time": 2.0, "chord": "G"}]
    report = analyze_chord_prediction(
        segs, segs, changes,
        duration=4.0,
        expected_symbols=["C", "G"],
    )
    assert report["timeline_score"] == 1.0
    assert report["boundary_score"] == 1.0
    assert report["symbol_recall"] == 1.0
    assert report["error_breakdown_pct"]["match"] == 100.0


def test_analyze_root_vs_quality_errors():
    pred = [
        {"time": 0.0, "end_time": 3.0, "chord": "Am"},
        {"time": 3.0, "end_time": 6.0, "chord": "F"},
    ]
    ref = [
        {"time": 0.0, "end_time": 3.0, "chord": "A"},
        {"time": 3.0, "end_time": 6.0, "chord": "G"},
    ]
    changes = [{"time": 3.0, "chord": "G"}]
    report = analyze_chord_prediction(
        pred, ref, changes,
        duration=6.0,
        tolerant=False,
    )
    assert report["error_breakdown_pct"]["quality_mismatch"] > 0
    assert report["error_breakdown_pct"]["root_mismatch"] > 0
