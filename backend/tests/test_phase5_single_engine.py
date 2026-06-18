"""Phase 5: single ML engine (chordia)."""
import pytest

from chord_model_registry import ML_MODEL_CHORDIA, ML_MODEL_DEFAULT
from chord_engine_ml import get_chord_ml_model


def test_get_chord_ml_model_defaults_to_chordia(monkeypatch):
    monkeypatch.delenv("CHORD_ML_MODEL", raising=False)
    monkeypatch.delenv("CHORD_MODEL", raising=False)
    assert get_chord_ml_model() == ML_MODEL_DEFAULT == ML_MODEL_CHORDIA


def test_deprecated_ml_model_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("CHORD_ML_MODEL", "conformer")
    monkeypatch.setenv("CHORD_ENGINE_STRICT", "1")
    with pytest.raises(ValueError, match="removed|invalid|unsupported"):
        get_chord_ml_model()


def test_deprecated_ml_model_warns_outside_strict_mode(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("CHORD_ML_MODEL", "autochord")
    monkeypatch.delenv("CHORD_ENGINE_STRICT", raising=False)
    assert get_chord_ml_model() == ML_MODEL_CHORDIA
    assert "removed" in caplog.text.lower() or "invalid" in caplog.text.lower()
