"""Phase 0 observability: cache keys and engine honesty."""
import json
import os
from unittest.mock import patch

import pytest

from engine_verify import EngineMismatchError, assert_engine_honest, clear_analysis_cache


def test_cache_metadata_includes_ml_model(monkeypatch):
    from analyzer import ANALYZER_VERSION, cache_metadata

    monkeypatch.setenv("CHORD_ENGINE", "ml")
    monkeypatch.setenv("CHORD_ML_MODEL", "chordia")
    monkeypatch.setenv("CHORD_STEM_MODE", "hpss")
    monkeypatch.setenv("CHORD_CHORDIA_DICT", "submission")
    meta = cache_metadata()
    assert meta["analyzer_version"] == ANALYZER_VERSION
    assert meta["chord_engine"] == "ml"
    assert meta["chord_ml_model"] == "chordia"
    assert meta["chord_stem_mode"] == "hpss"
    assert meta["chord_chordia_dict"] == "submission"
    assert meta["chord_pitch_correct"] is True


def test_cache_invalidates_when_ml_model_changes(monkeypatch):
    from analyzer import cache_metadata, is_cache_valid

    monkeypatch.setenv("CHORD_ENGINE", "ml")
    monkeypatch.setenv("CHORD_ML_MODEL", "ensemble")
    cached = cache_metadata()
    assert is_cache_valid(cached)

    monkeypatch.setenv("CHORD_ML_MODEL", "chordia")
    assert not is_cache_valid(cached)


def test_cache_invalidates_when_stem_mode_changes(monkeypatch):
    from analyzer import cache_metadata, is_cache_valid

    monkeypatch.setenv("CHORD_ENGINE", "ml")
    monkeypatch.setenv("CHORD_STEM_MODE", "auto")
    cached = cache_metadata()
    monkeypatch.setenv("CHORD_STEM_MODE", "hpss")
    assert not is_cache_valid(cached)


def test_assert_engine_honest_fails_on_ml_fallback(monkeypatch):
    monkeypatch.setenv("CHORD_ENGINE", "ml")
    monkeypatch.setenv("CHORD_ML_MODEL", "ensemble")
    monkeypatch.setenv("CHORD_ENGINE_STRICT", "1")
    with pytest.raises(EngineMismatchError, match="ran classic"):
        assert_engine_honest(
            {
                "chord_engine_actual": "classic",
                "ml_fallback_reason": "ImportError: no lv_chordia",
            }
        )


def test_assert_engine_honest_passes_when_ml_ran(monkeypatch):
    monkeypatch.setenv("CHORD_ENGINE", "ml")
    monkeypatch.setenv("CHORD_ML_MODEL", "chordia")
    monkeypatch.setenv("CHORD_ENGINE_STRICT", "1")
    assert_engine_honest({"chord_engine_actual": "ml", "ml_model": "chordia"})


def test_clear_analysis_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "abc.json").write_text("{}")
    removed = clear_analysis_cache(str(cache))
    assert removed == 1
    assert not list(cache.glob("*.json"))


def test_analyze_audio_writes_full_cache_metadata(tmp_path, monkeypatch):
    """Fresh analysis JSON must carry all cache key fields."""
    from analyzer import analyze_audio, cache_metadata, is_cache_valid

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHORD_ENGINE", "classic")
    (tmp_path / "cache").mkdir()

    fake_audio = tmp_path / "test.wav"
    import numpy as np
    import soundfile as sf

    sf.write(str(fake_audio), np.zeros(22050, dtype=np.float32), 22050)

    minimal_chords = [{"start": 0.0, "end": 1.0, "chord": "C"}]
    minimal_key = {"display": "C major", "chord_engine_actual": "classic"}

    display = {
        "timeline": minimal_chords,
        "model_timeline": minimal_chords,
        "beats": {},
        "capo": {"capo_fret": 0, "display": None, "transpose_semitones": 0},
        "presentation": "v13",
    }

    with patch("chord_engine.extract_chords", return_value=(minimal_chords, minimal_key, None)):
        with patch("presentation_timeline.build_display_timeline", return_value=display):
            with patch("analyzer._basic_pitch_predict", return_value=([], [], [])):
                with patch("analyzer._map_notes_to_fretboard", return_value=[]):
                    with patch("lyrics.attach_lyrics_to_timeline", side_effect=lambda c, *a, **k: (c, {})):
                        result = analyze_audio(str(fake_audio), "vid-test", force_reanalyze=True)

    meta = cache_metadata()
    for key, value in meta.items():
        assert result.get(key) == value

    cache_file = tmp_path / "cache" / "vid-test.json"
    assert cache_file.is_file()
    cached = json.loads(cache_file.read_text())
    assert is_cache_valid(cached)
