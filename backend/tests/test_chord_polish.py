"""Tests for flux-guided boundary detection and frame-voting label scoring."""
import numpy as np

from analyzer import (
    _apply_intro_guard,
    _bar_aware_min_duration,
    _boundary_candidate_times,
    _chroma_flux_series,
    _flux_peak_frames,
    _merge_adjacent_same_root,
    _pick_chord_for_region,
    _third_quality_bias,
)


def test_chroma_flux_detects_step_change():
    chroma = np.zeros((12, 80))
    chroma[0, :40] = 1.0
    chroma[7, 40:] = 1.0
    flux = _chroma_flux_series(chroma)
    peaks = _flux_peak_frames(flux, sr=22050, hop=512, beat_duration=0.5)
    assert len(peaks) >= 1
    assert float(flux[peaks[0]]) > 0.5


def test_boundary_candidates_include_flux_peak():
    approx = 2.0
    flux_times = [1.85, 4.0]
    onsets = [1.9, 3.0]
    cands = _boundary_candidate_times(approx, flux_times, onsets, beat_duration=0.5, search_sec=0.3)
    assert approx in cands
    assert 1.85 in cands or 1.9 in cands


def test_bar_aware_min_duration_scales_with_tempo():
    assert _bar_aware_min_duration(0.5) == 1.25
    assert _bar_aware_min_duration(0.25) == 0.625
    assert _bar_aware_min_duration(0.8) == 1.4


def test_merge_adjacent_same_root():
    segments = [
        {"time": 0.0, "end_time": 1.0, "chord": "Am", "confidence": 0.8},
        {"time": 1.0, "end_time": 2.0, "chord": "A5", "confidence": 0.7},
        {"time": 2.0, "end_time": 4.0, "chord": "G", "confidence": 0.9},
    ]
    merged = _merge_adjacent_same_root(segments)
    assert len(merged) == 2
    assert merged[0]["chord"] == "Am"
    assert merged[0]["end_time"] == 2.0


def test_intro_guard_collapses_early_flicker():
    sr, hop = 22050, 512
    n = 80
    chroma = np.zeros((12, n))
    chroma_low = np.zeros((12, n))
    chroma[0, :] = 1.0
    chroma[4, :] = 0.7
    chroma[7, :] = 0.8
    chroma_low[0, :] = 1.0

    segments = [
        {"time": 0.0, "end_time": 0.4, "chord": "G5", "confidence": 0.5},
        {"time": 0.4, "end_time": 0.8, "chord": "C5", "confidence": 0.5},
        {"time": 0.8, "end_time": 1.2, "chord": "Am", "confidence": 0.5},
        {"time": 1.2, "end_time": 5.0, "chord": "C", "confidence": 0.9},
    ]
    out = _apply_intro_guard(
        segments, 0.5, chroma, chroma_low, None,
        None, None, sr, hop, stable_beats=4,
    )
    assert len(out) < len(segments)
    assert out[0]["time"] == 0.0


def test_third_quality_bias_detects_minor():
    chroma = np.zeros(12)
    chroma[0] = 1.0   # C root
    chroma[3] = 0.9   # Eb minor third
    chroma[4] = 0.2   # E major third
    bias = _third_quality_bias(chroma, "C")
    assert bias.get("m", 0) > 1.0
    assert bias.get("", 1.0) < 1.0


def test_pick_chord_for_region_major_vs_minor():
    sr, hop = 22050, 512
    n = 60
    chroma = np.zeros((12, n))
    chroma_low = np.zeros((12, n))
    # C major triad: strong C, E, G
    for f in range(n):
        chroma[0, f] = 1.0
        chroma[4, f] = 0.8
        chroma[7, f] = 0.9
        chroma_low[0, f] = 1.0

    name, conf = _pick_chord_for_region(
        chroma, chroma_low, None,
        0.0, n * hop / sr, sr, hop,
        key_root=0, is_major=True,
    )
    assert name in ("C", "C5", "")
    assert conf > 0
