"""Tests for pitch normalization."""
import numpy as np
import librosa

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
