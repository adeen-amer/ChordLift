"""Tests for dynamic boundary reference resolution."""
import json
import numpy as np
from pathlib import Path

from eval_fixture_utils import merge_track_enrichment, load_all_overlays
from eval_chord_utils import (
    filter_reference_changes,
    filter_reference_timeline,
    resolve_boundary_references,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "chord_refs.json"


def test_resolve_boundary_beat_loop_fallback():
    sr = 22050
    t = np.linspace(0, 8, sr * 8)
    y = 0.5 * np.sin(2 * np.pi * 2 * t)
    track = {
        "progression": ["C", "G"],
        "beats_per_chord": 4,
        "boundary_cycles": 2,
        "boundary_method": "beat_loop",
    }
    changes, timeline, method = resolve_boundary_references(y, sr, track)
    assert method == "beat_loop"
    assert changes
    assert timeline


def test_merge_includes_eval_windows():
    tracks = json.loads(FIXTURES.read_text())["tracks"]
    let_it_be = next(t for t in tracks if t["id"] == "let-it-be")
    merged = merge_track_enrichment(let_it_be, load_all_overlays())
    assert merged.get("boundary_eval_start") == 12.0
    assert merged.get("progression") == ["C", "G", "Am", "F"]
    assert "reference_changes" not in merged


def test_filter_reference_window():
    changes = [
        {"time": 1.0, "chord": "C"},
        {"time": 5.0, "chord": "G"},
        {"time": 20.0, "chord": "Am"},
    ]
    filtered = filter_reference_changes(changes, eval_start=4.0, eval_end=15.0)
    assert len(filtered) == 1
    assert filtered[0]["chord"] == "G"

    timeline = [
        {"time": 0.0, "end_time": 5.0, "chord": "C"},
        {"time": 5.0, "end_time": 20.0, "chord": "G"},
    ]
    clipped = filter_reference_timeline(timeline, eval_start=4.0, eval_end=10.0)
    assert len(clipped) == 2
    assert clipped[0]["time"] == 4.0
    assert clipped[1]["end_time"] == 10.0
