"""Lever 0 self-check: posterior wrapper reproduces the packaged decode path."""
import numpy as np
import pytest

lv_chordia = pytest.importorskip("lv_chordia")


def _sine_chord(sr=22050, duration=5.0):
    """C major triad, same recipe as tests/test_chord_pipeline.py."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [261.63, 329.63, 392.00]
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return y.astype(np.float32)


def test_probs_wrapper_matches_packaged_path(tmp_path):
    """True parity: wrapper decode must equal lv_chordia's own function."""
    import soundfile as sf
    from lv_chordia.chord_recognition import chord_recognition

    from chord_engine_chordia import decode_chordia_probs, recognize_chordia_probs

    y = _sine_chord()
    probs, hmm, entry = recognize_chordia_probs(y, 22050)

    assert isinstance(probs, list) and len(probs) == 6
    assert all(p.ndim == 2 for p in probs)
    # softmax heads: rows sum to ~1
    assert np.allclose(probs[0].sum(axis=1), 1.0, atol=1e-3)

    via_probs = decode_chordia_probs(probs, hmm, entry)
    wav = tmp_path / "sine.wav"
    sf.write(str(wav), y, 22050)
    packaged = chord_recognition(audio_path=str(wav))
    assert via_probs == packaged


def test_chordia_confidence_in_unit_interval():
    from chord_engine_chordia import chordia_confidence

    fake = [np.full((10, 4), 0.25), np.zeros((10, 13))]
    assert chordia_confidence(fake) == pytest.approx(0.25)


def test_model_list_env_override(monkeypatch):
    import chord_engine_chordia as cec

    monkeypatch.setenv("CHORD_CHORDIA_MODELS", "name_a.best,name_b.best")
    assert cec._model_names() == ["name_a.best", "name_b.best"]
    monkeypatch.delenv("CHORD_CHORDIA_MODELS")
    from lv_chordia.chord_recognition import MODEL_NAMES
    assert cec._model_names() == list(MODEL_NAMES)
