"""Tests for pitch normalization."""
import numpy as np
import librosa
import pytest

from pitch_utils import estimate_pitch_shift_semitones, normalize_pitch


def test_normalize_pitch_leaves_in_tune_audio_unchanged():
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    out, shift = normalize_pitch(y.astype(np.float32), sr)
    assert abs(shift) < 0.05
    np.testing.assert_allclose(out, y, atol=1e-5)


def test_estimate_pitch_shift_detects_sharp_signal():
    sr = 22050
    duration = 3.0
    y = librosa.tone(440, sr=sr, duration=duration)
    y_sharp = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
    shift = estimate_pitch_shift_semitones(y_sharp, sr)
    assert shift > 0.1


def test_normalize_pitch_applies_detected_shift(monkeypatch):
    import numpy as np
    import pitch_utils

    monkeypatch.setattr(pitch_utils, "estimate_pitch_shift_semitones", lambda y, sr: 0.3)
    y = np.random.RandomState(0).randn(22050).astype(np.float32)
    corrected, shift = pitch_utils.normalize_pitch(y, 22050)
    assert shift == pytest.approx(-0.3)
    assert not np.array_equal(corrected, y)


def test_pitch_shift_audio_is_public():
    import numpy as np
    from pitch_utils import pitch_shift_audio

    y = np.random.RandomState(0).randn(22050).astype(np.float32)
    out = pitch_shift_audio(y, 22050, 0.5)
    assert out is not None and len(out) == len(y)

