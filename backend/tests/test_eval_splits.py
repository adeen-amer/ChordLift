"""Tests for eval train/benchmark/held-out splits."""
from eval_split_utils import classify_track, load_benchmark_ids, load_held_out_ids, split_summary


def test_splits_disjoint():
    benchmark = load_benchmark_ids()
    held_out = load_held_out_ids()
    assert benchmark.isdisjoint(held_out)


def test_classify_track():
    assert classify_track("let-it-be") == "gold"
    assert classify_track("ho-hey") == "held_out"
    assert classify_track("creep") == "benchmark"
    assert classify_track("farda") == "held_out"
    assert classify_track("aisay-kaisay") == "tune"


def test_split_summary():
    rows = [
        {"id": "let-it-be", "timeline_score": 0.4, "symbol_recall": 0.9},
        {"id": "ho-hey", "timeline_score": 0.3, "symbol_recall": 0.8},
        {"id": "creep", "timeline_score": 0.35, "boundary_score": 0.2},
    ]
    summary = split_summary(rows)
    assert "gold" in summary
    assert summary["gold"]["track_count"] == 1
    assert "held_out" in summary
    assert "benchmark" in summary
