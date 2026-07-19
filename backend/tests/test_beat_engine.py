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


# --- Beat-Transformer engine ------------------------------------------------
from pathlib import Path  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "beat_transformer.pth"


def test_beat_transformer_forward_pass_matches_contract():
    """Real forward pass on the actual checkpoint (no madmom needed).

    Exercises FINDINGS.md's spectrogram construction + model contract for
    real: build_model_input -> (1, 5, T, 128), forward -> pred (1, T, 2) +
    tempo (1, 300). This is the part of the transformer engine that CAN run
    on this Windows dev box (torch/librosa only); madmom post-processing is
    covered separately below with importorskip.
    """
    pytest.importorskip("torch")
    if not CHECKPOINT.exists():
        pytest.skip("Beat-Transformer checkpoint not present")
    import torch

    from beat_transformer.infer import build_model_input
    from model_cache import get_beat_transformer_model

    stems = separate_stems(_sine_chord(sr=44100), 44100)
    x = build_model_input(stems, 44100)
    assert x.shape[0] == 1 and x.shape[1] == 5 and x.shape[3] == 128
    assert x.dtype == torch.float32

    model = get_beat_transformer_model()
    with torch.no_grad():
        pred, tempo = model(x)

    T = x.shape[2]
    assert pred.shape == (1, T, 2)
    assert tempo.shape == (1, 300)


def test_beat_transformer_checkpoint_sha256_matches_findings():
    """The download helper's hash check against the checkpoint FINDINGS.md verified."""
    if not CHECKPOINT.exists():
        pytest.skip("Beat-Transformer checkpoint not present")
    from model_cache import _sha256

    assert _sha256(CHECKPOINT) == (
        "1940a5034bb2bb7860c1a3219d359906913be88c915f6135a5ce8382c239d738"
    )


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Beat-Transformer checkpoint not present")
def test_transformer_engine_returns_valid_beat_grid(monkeypatch):
    pytest.importorskip("madmom")
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "transformer")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 1
    assert np.all(np.diff(grid.beat_times) >= 0)


def test_auto_prefers_transformer_for_demucs_stems(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")
    calls = []
    monkeypatch.setattr(
        beat_tracking,
        "_track_beats_transformer",
        lambda *a, **k: calls.append("transformer")
        or BeatGrid(np.array([0.0, 0.5]), np.array([0.0]), 120.0),
    )
    monkeypatch.setattr(
        beat_tracking,
        "_track_beats_madmom",
        lambda *a, **k: calls.append("madmom")
        or BeatGrid(np.array([0.0, 0.5]), np.array([0.0]), 120.0),
    )
    stems = separate_stems(_sine_chord(), 22050)
    object.__setattr__(stems, "method", "demucs")  # force demucs-shaped call for this test
    track_beats_auto(stems, 22050)
    assert calls == ["transformer"]


def test_auto_falls_back_to_madmom_when_transformer_fails(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")
    calls = []

    def _boom(*_a, **_k):
        calls.append("transformer")
        raise RuntimeError("transformer unavailable")

    monkeypatch.setattr(beat_tracking, "_track_beats_transformer", _boom)
    monkeypatch.setattr(
        beat_tracking,
        "_track_beats_madmom",
        lambda *a, **k: calls.append("madmom")
        or BeatGrid(np.array([0.0, 0.5]), np.array([0.0]), 120.0),
    )
    stems = separate_stems(_sine_chord(), 22050)
    object.__setattr__(stems, "method", "demucs")
    track_beats_auto(stems, 22050)
    assert calls == ["transformer", "madmom"]
