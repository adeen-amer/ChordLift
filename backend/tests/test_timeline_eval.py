"""Unit tests for timeline agreement metrics."""
from eval_chord_utils import (
    chord_at_time,
    chords_match,
    timeline_agreement_score,
)


def test_chord_at_time():
    segs = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    assert chord_at_time(segs, 0.5) == "C"
    assert chord_at_time(segs, 2.5) == "G"


def test_timeline_perfect_agreement():
    segs = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    ref = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    score = timeline_agreement_score(segs, ref, step_sec=0.5, max_offset_sec=0.0)
    assert score == 1.0


def test_chords_match_tolerant_power():
    assert chords_match("C5", "C", tolerant=True)
    assert chords_match("Am", "Am7", tolerant=True)
