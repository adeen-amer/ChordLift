"""backend/tests/test_beat_engine.py"""
import numpy as np
import pytest

import beat_tracking
from beat_tracking import BeatGrid, track_beats_auto
from stem_separation import separate_stems


def _sine_chord(sr=22050, duration=6.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 110 * t)
    y += 0.25 * np.sin(2 * np.pi * 138 * t)
    y += 0.2 * np.sin(2 * np.pi * 165 * t)
    return y.astype(np.float32)


def test_librosa_engine_still_works(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "librosa")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 4


def test_madmom_engine_returns_valid_beat_grid(monkeypatch):
    pytest.importorskip("madmom")
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "madmom")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 1
    assert len(grid.downbeat_times) >= 1
    assert np.all(np.diff(grid.beat_times) >= 0)


def test_auto_falls_back_to_librosa_on_madmom_failure(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("madmom unavailable")

    monkeypatch.setattr(beat_tracking, "_track_beats_madmom", _boom)
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 4
