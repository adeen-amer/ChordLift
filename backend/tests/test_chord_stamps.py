"""Tests for chord stamp eval fixtures."""
import json
from pathlib import Path

from eval_fixture_utils import (
    DEFAULT_STAMP_REFS,
    apply_chord_stamps,
    load_chord_stamps,
    load_gold_labels,
    prepare_eval_track,
    load_all_overlays,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "chord_refs.json"


def test_load_chord_stamps_structure():
    stamps = load_chord_stamps()
    if not DEFAULT_STAMP_REFS.exists():
        assert stamps == {}
        return

    assert isinstance(stamps, dict)
    for tid, stamp in stamps.items():
        assert "reference_changes" in stamp, tid
        assert "reference_timeline" in stamp, tid
        changes = stamp["reference_changes"]
        assert changes, tid
        for change in changes:
            assert "time" in change and "chord" in change


def test_prepare_eval_track_uses_stamps():
    tracks = json.loads(FIXTURES.read_text())["tracks"]
    let_it_be = next(t for t in tracks if t["id"] == "let-it-be")
    overlays = load_all_overlays()
    stamps = load_chord_stamps()
    gold = load_gold_labels()
    if not stamps.get("let-it-be") and not gold.get("let-it-be"):
        return

    merged = prepare_eval_track(let_it_be, overlays, stamps, gold)
    assert merged.get("reference_changes")
    assert merged.get("reference_timeline")
    assert merged.get("reference_source") == "gold"


def test_prepare_eval_track_live_strips_stamps():
    fake_stamp = {
        "reference_changes": [{"time": 1.0, "chord": "C"}],
        "reference_timeline": [{"time": 1.0, "end_time": 2.0, "chord": "C"}],
    }
    track = {"id": "test", "progression": ["C", "G"]}
    merged = prepare_eval_track(
        track,
        stamps={"test": fake_stamp},
        live_boundaries=True,
    )
    assert "reference_changes" not in merged


def test_apply_chord_stamps_preserves_progression():
    track = {"id": "x", "progression": ["Am", "G"]}
    stamp = {
        "reference_changes": [{"time": 0.0, "chord": "Am"}],
        "reference_timeline": [],
        "boundary_method": "phase_aligned",
    }
    merged = apply_chord_stamps(track, stamp)
    assert merged["progression"] == ["Am", "G"]
    assert merged["reference_changes"][0]["chord"] == "Am"
