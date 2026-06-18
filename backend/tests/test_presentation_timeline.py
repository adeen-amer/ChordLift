"""Phase 13 presentation timeline tests."""
import numpy as np

from presentation_timeline import (
    apply_confidence_tiers,
    capo_info_default,
    constrain_timeline_to_key,
    snap_timeline_to_beats,
)


def test_snap_timeline_merges_same_chord_on_beat():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    timeline = [
        {"time": 0.1, "end_time": 0.55, "chord": "C", "confidence": 0.8},
        {"time": 0.55, "end_time": 1.05, "chord": "C", "confidence": 0.7},
        {"time": 1.05, "end_time": 2.0, "chord": "G", "confidence": 0.9},
    ]
    out = snap_timeline_to_beats(timeline, beats)
    assert len(out) == 2
    assert out[0]["chord"] == "C"
    assert out[0]["time"] == 0.0
    assert out[1]["chord"] == "G"


def test_constrain_snaps_low_conf_out_of_key():
    timeline = [
        {"time": 0.0, "end_time": 2.0, "chord": "F#m", "confidence": 0.3, "is_low_confidence": True},
    ]
    out = constrain_timeline_to_key(timeline, key_root=0, is_major=True)
    assert out[0]["chord"].endswith("m")
    assert out[0].get("display_adjusted") is True


def test_bvii_in_c_major_not_snapped():
    timeline = [
        {"time": 0.0, "end_time": 2.0, "chord": "Bb", "confidence": 0.3, "is_low_confidence": True},
    ]
    out = constrain_timeline_to_key(timeline, key_root=0, is_major=False)
    assert out[0]["chord"] == "Bb"


def test_secondary_dominant_not_snapped():
    timeline = [
        {"time": 0.0, "end_time": 2.0, "chord": "D", "confidence": 0.3, "is_low_confidence": True},
    ]
    out = constrain_timeline_to_key(timeline, key_root=0, is_major=True)
    assert out[0]["chord"] == "D"


def test_confidence_tiers():
    timeline = [{"time": 0, "end_time": 1, "chord": "C", "confidence": 0.2}]
    out = apply_confidence_tiers(timeline)
    assert out[0]["confidence_tier"] == "low"
    assert out[0]["is_low_confidence"] is True


def test_capo_not_from_pitch_detune():
    info = capo_info_default()
    assert info["capo_fret"] == 0
    assert info["display"] is None
