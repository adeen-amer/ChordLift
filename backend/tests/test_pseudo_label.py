"""backend/tests/test_pseudo_label.py

Task 3 adds a module-level `pytest.importorskip("lv_chordia")` once
label_track needs it, same convention as test_chord_training_finetune.py:
the whole file skips together when lv_chordia/torch aren't installed,
rather than gating individual tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "chord_training"))


def test_frame_confidences_is_max_over_classes():
    from pseudo_label import frame_confidences

    probs = [np.array([[0.1, 0.9], [0.6, 0.4]])]
    assert list(frame_confidences(probs)) == pytest.approx([0.9, 0.6])


def test_segment_mean_confidence_averages_covered_frames():
    from pseudo_label import segment_mean_confidence

    frame_conf = np.array([0.2, 0.8, 0.8, 0.2])
    # hop=100, sr=1000 -> 0.1s/frame; segment [0.1s, 0.3s) covers frames 1,2
    seg = {"start_time": 0.1, "end_time": 0.3}
    assert segment_mean_confidence(seg, frame_conf, sr=1000, hop=100) == pytest.approx(0.8)


def test_filter_low_confidence_segments_drops_below_threshold():
    from pseudo_label import filter_low_confidence_segments

    frame_conf = np.array([0.9, 0.9, 0.1, 0.1])
    segs = [
        {"start_time": 0.0, "end_time": 0.2, "chord": "C:maj"},
        {"start_time": 0.2, "end_time": 0.4, "chord": "X"},
    ]
    kept = filter_low_confidence_segments(segs, frame_conf, sr=1000, hop=100, threshold=0.5)
    assert kept == [segs[0]]
