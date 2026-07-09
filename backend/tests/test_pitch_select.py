"""Decode-both pitch selection (v50 levers 1+2). Chordia calls are mocked."""
import numpy as np
import pytest

import chord_engine_ml


def _mk_probs(peak):
    """One triad head with constant max-prob `peak`, plus 5 dummy heads."""
    triad = np.full((8, 4), (1 - peak) / 3)
    triad[:, 0] = peak
    return [triad] + [np.zeros((8, 2))] * 5


def _patch(monkeypatch, conf_raw, conf_cor, shift=0.4):
    calls = {"n": 0, "decoded": []}
    branches = [
        (_mk_probs(conf_raw), "hmm_raw", "entry_raw"),
        (_mk_probs(conf_cor), "hmm_cor", "entry_cor"),
    ]

    def fake_probs(y, sr):
        out = branches[calls["n"]]
        calls["n"] += 1
        return out

    def fake_decode(probs, hmm, entry):
        calls["decoded"].append((float(np.asarray(probs[0]).max(axis=1).mean()), hmm))
        return [{"start_time": 0.0, "end_time": 1.0, "chord": f"via_{hmm}"}]

    monkeypatch.setattr(chord_engine_ml, "_recognize_probs", fake_probs)
    monkeypatch.setattr(chord_engine_ml, "_decode_probs", fake_decode)
    monkeypatch.setattr(
        chord_engine_ml, "_estimate_shift", lambda y, sr: shift
    )
    monkeypatch.setattr(
        chord_engine_ml, "_shift_audio", lambda y, sr, s: y * 0.99
    )
    return calls


def test_corrected_branch_wins_on_higher_confidence(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "confidence")
    monkeypatch.setenv("CHORD_PITCH_CONF_MARGIN", "0.0")
    _patch(monkeypatch, conf_raw=0.5, conf_cor=0.7, shift=0.4)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert segs[0]["chord"] == "via_hmm_cor"
    assert applied == pytest.approx(-0.4)


def test_raw_wins_on_tie(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "confidence")
    monkeypatch.setenv("CHORD_PITCH_CONF_MARGIN", "0.0")
    _patch(monkeypatch, conf_raw=0.6, conf_cor=0.6)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert segs[0]["chord"] == "via_hmm_raw"
    assert applied == 0.0


def test_margin_blocks_marginal_correction(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "confidence")
    monkeypatch.setenv("CHORD_PITCH_CONF_MARGIN", "0.05")
    _patch(monkeypatch, conf_raw=0.60, conf_cor=0.63)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert segs[0]["chord"] == "via_hmm_raw"


def test_small_shift_skips_second_inference(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "confidence")
    calls = _patch(monkeypatch, conf_raw=0.5, conf_cor=0.9, shift=0.01)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert calls["n"] == 1 and applied == 0.0


def test_off_mode_never_corrects(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "off")
    calls = _patch(monkeypatch, conf_raw=0.5, conf_cor=0.9, shift=0.8)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert calls["n"] == 1 and applied == 0.0


def test_tta_decodes_averaged_probs(monkeypatch):
    monkeypatch.setenv("CHORD_PITCH_SELECT", "tta")
    calls = _patch(monkeypatch, conf_raw=0.5, conf_cor=0.9, shift=0.4)
    segs, applied = chord_engine_ml._chordia_segments_pitch_selected(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert applied == 0.0  # TTA blends labels; no shift is "applied"
    assert calls["decoded"][0][0] == pytest.approx(0.7)  # mean of 0.5 and 0.9
    assert calls["decoded"][0][1] == "hmm_raw"  # decoded with raw-branch decoder


def test_ml_path_does_not_precorrect_full_mix(monkeypatch):
    import chord_engine

    monkeypatch.setattr(chord_engine, "CHORD_ENGINE", "ml")

    def boom(y, sr):
        raise AssertionError("normalize_pitch must not run on the ML path")

    monkeypatch.setattr(chord_engine, "normalize_pitch", boom)
    monkeypatch.setattr(
        "chord_engine_ml.extract_chords_ml",
        lambda y, sr, pipeline: ([{"chord": "C", "start": 0.0, "end": 1.0}], {}),
    )
    monkeypatch.setattr(
        "chord_pipeline.build_chord_pipeline_context",
        lambda y, sr: __import__("types").SimpleNamespace(
            stems=__import__("types").SimpleNamespace(method="hpss")
        ),
    )
    segments, key_info = chord_engine.extract_chords(
        np.zeros(22050, dtype=np.float32), 22050
    )
    assert segments[0]["chord"] == "C"
    assert key_info["chord_engine_actual"] == "ml"


def test_classic_path_still_normalizes_pitch(monkeypatch):
    import chord_engine

    monkeypatch.setattr(chord_engine, "CHORD_ENGINE", "classic")
    called = {}

    def fake_normalize(y, sr):
        called["yes"] = True
        return y, 0.0

    monkeypatch.setattr(chord_engine, "normalize_pitch", fake_normalize)
    monkeypatch.setattr(
        "analyzer.extract_chords",
        lambda y, sr, pipeline: ([{"chord": "C", "start": 0.0, "end": 1.0}], {}),
    )
    monkeypatch.setattr(
        "chord_pipeline.build_chord_pipeline_context",
        lambda y, sr: __import__("types").SimpleNamespace(
            stems=__import__("types").SimpleNamespace(method="hpss")
        ),
    )
    chord_engine.extract_chords(np.zeros(22050, dtype=np.float32), 22050)
    assert called.get("yes")
