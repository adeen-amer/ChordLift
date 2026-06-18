"""Tests for ML environment checks."""
from ml_env import check_ml_chord_deps, summarize_ml_readiness


def test_summarize_ml_readiness_shape():
    status = summarize_ml_readiness()
    assert "engine_requested" in status
    assert "ml_deps_ok" in status
    assert isinstance(status["missing_packages"], list)


def test_check_ml_deps_returns_list():
    assert isinstance(check_ml_chord_deps(), list)
