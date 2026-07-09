# Raw Accuracy v50 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Dispatch code-execution subagents with `model: sonnet` (Sonnet 5) per user request.**

**Goal:** Lift gold-v2 TEST majmin WCSR from 73.0% (v49) to ≥76% by exposing lv-chordia frame posteriors and replacing the predictive pitch gate with measured decode-both selection.

**Architecture:** All levers operate on the frame posteriors lv_chordia already computes internally (5-checkpoint ensemble → averaged softmax heads → XHMM decode). We add a posterior-returning wrapper in `chord_engine_chordia.py`, move pitch handling from a full-mix pre-correction (`chord_engine.py`) into a decode-both selection inside the ML path (`chord_engine_ml.py`), and delete the four-threshold reliability gate in `pitch_utils.py`.

**Tech Stack:** Python 3.11 (backend/.venv), lv_chordia (PyTorch), librosa, pytest, existing `eval_gold_mir.py` harness.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-raw-accuracy-v50-design.md`
- Eval runs always set: `CHORD_ENGINE=ml`, `CHORD_ENGINE_STRICT=1`, `CHORD_ML_POSTPROCESS=raw`, `CHORD_ML_MODEL=chordia`, flags `--no-cache --require-audio-identity` (copy the `rebaseline_v49.py` pattern exactly).
- DEV picks candidates; TEST runs **once** at the end. Both splits reported.
- `another-one-bites-the-dust` TEST majmin must stay ≥ 0.128 (v49 level).
- All backend commands run from `backend/` using `.venv`: `cd backend && .venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures`
- Working tree is the ChordLift repo: `/Users/adeenamer/Documents/Test/ChordLift`
- Do not edit anything under `backend/.venv/` (site-packages is read-only reference).

---

### Task 1: Posterior plumbing in chord_engine_chordia.py (Lever 0)

**Files:**
- Modify: `backend/chord_engine_chordia.py`
- Test: `backend/tests/test_chordia_probs.py` (create)

**Interfaces:**
- Consumes: `lv_chordia` internals (`MODEL_NAMES`, `ChordNet`, `NetworkInterface`, `CQTV2`, `XHMMDecoder`, `DataEntry`, `io`, `DEFAULT_SR`, `DEFAULT_HOP_LENGTH`) — mirrors `lv_chordia/chord_recognition.py:49-77`.
- Produces (later tasks rely on these exact names):
  - `recognize_chordia_probs(y_chord, sr: int) -> tuple[list, XHMMDecoder, DataEntry]` — probs is a list of 6 np.ndarrays `[n_frames, n_classes]` (triad, bass, 7th, 9th, 11th, 13th heads), averaged over the 5 checkpoints.
  - `decode_chordia_probs(probs, hmm, entry) -> list[dict]` — raw segments `[{start_time, end_time, chord}]`.
  - `chordia_confidence(probs) -> float` — mean over frames of max triad posterior.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_chordia_probs.py`:

```python
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


def test_probs_wrapper_matches_packaged_path():
    from chord_engine_chordia import (
        decode_chordia_probs,
        recognize_chordia,
        recognize_chordia_probs,
    )

    y = _sine_chord()
    probs, hmm, entry = recognize_chordia_probs(y, 22050)

    assert isinstance(probs, list) and len(probs) == 6
    assert all(p.ndim == 2 for p in probs)
    # softmax heads: rows sum to ~1
    assert np.allclose(probs[0].sum(axis=1), 1.0, atol=1e-3)

    via_probs = decode_chordia_probs(probs, hmm, entry)
    via_labels = recognize_chordia(y, 22050)
    assert via_probs == via_labels


def test_chordia_confidence_in_unit_interval():
    from chord_engine_chordia import chordia_confidence

    fake = [np.full((10, 4), 0.25), np.zeros((10, 13))]
    assert chordia_confidence(fake) == pytest.approx(0.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chordia_probs.py -v`
Expected: FAIL with `ImportError: cannot import name 'recognize_chordia_probs'`

- [ ] **Step 3: Implement the posterior wrapper**

Replace the body of `backend/chord_engine_chordia.py` with:

```python
"""ISMIR 2019 large-vocabulary chord recognition (lv-chordia / PyTorch)."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

from chord_label_utils import raw_labels_to_segments

logger = logging.getLogger(__name__)

CHORDIA_DICT = os.getenv("CHORD_CHORDIA_DICT", "submission").lower().strip()


def recognize_chordia_probs(y_chord, sr: int):
    """
    Run lv-chordia inference and return frame posteriors instead of labels.

    Returns (probs, hmm, entry):
      probs — list of 6 arrays [n_frames, n_classes] (triad, bass, 7, 9, 11, 13
              softmax heads), averaged over the 5 packaged checkpoints.
      hmm   — XHMMDecoder ready to decode these probs.
      entry — DataEntry carrying sr/hop metadata the decoder needs.

    Mirrors lv_chordia.chord_recognition.chord_recognition() internals so the
    posteriors can be reused (confidence scoring, TTA averaging) before decode.
    """
    import importlib.resources

    from lv_chordia.chord_recognition import MODEL_NAMES
    from lv_chordia.chordnet_ismir_naive import ChordNet
    from lv_chordia.extractors.cqt import CQTV2
    from lv_chordia.extractors.xhmm_ismir import XHMMDecoder
    from lv_chordia.mir import DataEntry, io
    from lv_chordia.mir.nn.train import NetworkInterface
    from lv_chordia.settings import DEFAULT_HOP_LENGTH, DEFAULT_SR

    with importlib.resources.path(
        "lv_chordia.data", f"{CHORDIA_DICT}_chord_list.txt"
    ) as data_file:
        hmm = XHMMDecoder(template_file=str(data_file))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y_chord, sr)
    try:
        entry = DataEntry()
        entry.prop.set("sr", DEFAULT_SR)
        entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
        entry.append_file(tmp_path, io.MusicIO, "music")
        entry.append_extractor(CQTV2, "cqt")
        cqt = entry.cqt  # materialize before the temp file is unlinked
        per_model = []
        for model_name in MODEL_NAMES:
            net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)
            logger.info("Inference: %s", model_name)
            per_model.append(net.inference(cqt))
        probs = [
            np.mean([p[i] for p in per_model], axis=0)
            for i in range(len(per_model[0]))
        ]
        return probs, hmm, entry
    finally:
        os.unlink(tmp_path)


def decode_chordia_probs(probs, hmm, entry) -> list[dict]:
    """XHMM-decode posteriors to raw chordia JSON segments."""
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    return [
        {
            "start_time": float(f"{seg[0]:.2f}"),
            "end_time": float(f"{seg[1]:.2f}"),
            "chord": str(seg[2]),
        }
        for seg in chordlab
    ]


def chordia_confidence(probs) -> float:
    """Decoder confidence: mean over frames of the max triad posterior."""
    return float(np.asarray(probs[0]).max(axis=1).mean())


def recognize_chordia(y_chord, sr: int) -> list[dict]:
    """
    Run lv-chordia on a mono waveform (prefer stem chord_signal).

    Returns raw chordia JSON segments: [{start_time, end_time, chord}, ...]
    """
    probs, hmm, entry = recognize_chordia_probs(y_chord, sr)
    return decode_chordia_probs(probs, hmm, entry)


def chordia_to_segments(raw) -> list[dict]:
    return raw_labels_to_segments(raw, default_confidence=0.78)
```

Note: `recognize_chordia` no longer calls `lv_chordia.chord_recognition` — the
test compares against the wrapper-decoded output *and* asserts head shapes, so
parity means "identical to what serving produced before" (same internals, same
checkpoints, same decoder, deterministic under `torch.no_grad`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chordia_probs.py -v`
Expected: 2 PASSED (first test takes ~30-90s — 5 checkpoints on CPU; acceptable)

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `cd backend && .venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures`
Expected: all pass (no existing test imports `recognize_chordia` internals)

- [ ] **Step 6: Commit**

```bash
git add backend/chord_engine_chordia.py backend/tests/test_chordia_probs.py
git commit -m "feat: expose lv-chordia frame posteriors (v50 lever 0)"
```

---

### Task 2: Delete the pitch gate, publish pitch_shift_audio (pitch_utils.py)

**Files:**
- Modify: `backend/pitch_utils.py`
- Modify: `backend/tests/test_pitch_utils.py` (gate test at line 26 references `_estimate_reliable`)

**Interfaces:**
- Produces: `pitch_shift_audio(y, sr, shift) -> np.ndarray | None` (renamed from `_pitch_shift`; applies `-shift` semitones, returns None on failure). `estimate_pitch_shift_semitones`, `MIN_SHIFT_SEMITONES`, `MAX_SHIFT_SEMITONES`, `normalize_pitch` keep their signatures.
- Deletes: `_estimate_reliable`, `_chroma_key_fit_score`, `_bass_energy_ratio`, constants `MIN_KEY_SCORE_GAIN`, `MAX_STEM_ESTIMATE_DELTA`, `BASS_HEAVY_RATIO`, `BASS_HEAVY_SHIFT_MIN`, `BASS_HEAVY_MIN_GAIN`.

- [ ] **Step 1: Update the tests first (failing)**

In `backend/tests/test_pitch_utils.py`, delete
`test_pitch_gate_skips_unreliable_correction` (it monkeypatches
`pitch_utils._estimate_reliable`, which is going away) and append:

```python
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
```

(`import pytest` is already at the top of the file; add it if not.)

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_utils.py -v`
Expected: `test_pitch_shift_audio_is_public` FAILS with ImportError; `test_normalize_pitch_applies_detected_shift` FAILS (gate blocks the shift)

- [ ] **Step 3: Implement**

In `backend/pitch_utils.py`:
1. Delete `_estimate_reliable`, `_chroma_key_fit_score`, `_bass_energy_ratio` and the five gate-threshold constants (lines 15-22: `MIN_KEY_SCORE_GAIN` through `BASS_HEAVY_MIN_GAIN`), and the module docstring reference to the gate.
2. Rename `_pitch_shift` → `pitch_shift_audio` (same body).
3. `normalize_pitch` becomes gate-free (v48 blind behavior — now used only by the classic fallback path after Task 4):

```python
def normalize_pitch(
    y: np.ndarray,
    sr: int,
    *,
    max_shift: float = MAX_SHIFT_SEMITONES,
) -> tuple[np.ndarray, float]:
    """
    Pitch-shift audio toward concert tuning when detune is detected.

    Returns (audio, applied_shift_semitones). Shift is 0 when negligible.
    ponytail: no reliability gate here — the ML path measures both branches
    (chord_engine_ml) instead of predicting; this blind path only serves the
    classic fallback engine.
    """
    shift = estimate_pitch_shift_semitones(y, sr)
    shift = float(np.clip(shift, -max_shift, max_shift))

    if abs(shift) < MIN_SHIFT_SEMITONES:
        return y, 0.0

    corrected = pitch_shift_audio(y, sr, shift)
    if corrected is None:
        return y, 0.0

    logger.info("Applied pitch correction: %.2f semitones", -shift)
    return corrected, -shift
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_utils.py -v`
Expected: all PASS

- [ ] **Step 5: Grep for dangling references**

Run: `grep -rn "_estimate_reliable\|_chroma_key_fit_score\|_bass_energy_ratio\|_pitch_shift\b" backend --include="*.py" | grep -v .venv`
Expected: no hits outside `pitch_utils.py`'s own `pitch_shift_audio`. (`scripts/phase12_5_postprocess_ablation.py` uses only `normalize_pitch` — unaffected.)

- [ ] **Step 6: Commit**

```bash
git add backend/pitch_utils.py backend/tests/test_pitch_utils.py
git commit -m "feat: remove predictive pitch gate, publish pitch_shift_audio (v50)"
```

---

### Task 3: Decode-both / TTA pitch selection in the ML path (Levers 1 + 2)

**Files:**
- Modify: `backend/chord_engine_ml.py`
- Test: `backend/tests/test_pitch_select.py` (create)

**Interfaces:**
- Consumes: Task 1's `recognize_chordia_probs` / `decode_chordia_probs` / `chordia_confidence`; Task 2's `pitch_shift_audio`, `estimate_pitch_shift_semitones`, `MIN_SHIFT_SEMITONES`.
- Produces: `_chordia_segments_pitch_selected(y_chord, sr) -> tuple[list[dict], float]` (raw segments, applied_shift). `extract_chords_ml` keeps its signature but its returned `key_info` now includes `pitch_correction_semitones` (rounded, only when nonzero) and `pitch_select_mode`.
- Env knobs: `CHORD_PITCH_SELECT` ∈ `confidence` (default) | `tta` | `off`; `CHORD_PITCH_CONF_MARGIN` (float, default `0.0`).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_pitch_select.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_select.py -v`
Expected: FAIL with `AttributeError: module 'chord_engine_ml' has no attribute '_recognize_probs'`

- [ ] **Step 3: Implement**

In `backend/chord_engine_ml.py`, add below the imports (module-level indirection
so tests can monkeypatch without touching chord_engine_chordia):

```python
import os

import numpy as np

from chord_engine_chordia import (
    chordia_confidence,
    decode_chordia_probs,
    recognize_chordia_probs,
)
from pitch_utils import (
    MIN_SHIFT_SEMITONES,
    estimate_pitch_shift_semitones,
    pitch_shift_audio,
)

# Indirection points for tests; production uses the real functions.
_recognize_probs = recognize_chordia_probs
_decode_probs = decode_chordia_probs
_estimate_shift = estimate_pitch_shift_semitones
_shift_audio = pitch_shift_audio


def _chordia_segments_pitch_selected(y_chord, sr):
    """
    Decode-both pitch handling (v50): measure which branch decodes better
    instead of predicting reliability (the v49 gate this replaces cost 1.3pp).

    Returns (raw_segments, applied_shift_semitones).
    """
    mode = os.getenv("CHORD_PITCH_SELECT", "confidence").lower().strip()
    shift = _estimate_shift(y_chord, sr)
    probs_raw, hmm_raw, entry_raw = _recognize_probs(y_chord, sr)

    if mode == "off" or abs(shift) < MIN_SHIFT_SEMITONES:
        return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0

    corrected = _shift_audio(y_chord, sr, shift)
    if corrected is None:
        return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0
    probs_cor, hmm_cor, entry_cor = _recognize_probs(corrected, sr)

    if mode == "tta":
        n = min(p.shape[0] for p in (probs_raw[0], probs_cor[0]))
        avg = [
            np.mean([a[:n], b[:n]], axis=0)
            for a, b in zip(probs_raw, probs_cor)
        ]
        return _decode_probs(avg, hmm_raw, entry_raw), 0.0

    margin = float(os.getenv("CHORD_PITCH_CONF_MARGIN", "0.0"))
    conf_raw = chordia_confidence(probs_raw)
    conf_cor = chordia_confidence(probs_cor)
    if conf_cor > conf_raw + margin:
        logger.info(
            "Pitch select: corrected wins (%.4f vs %.4f, shift %.2f st)",
            conf_cor, conf_raw, -shift,
        )
        return _decode_probs(probs_cor, hmm_cor, entry_cor), -shift
    logger.info(
        "Pitch select: raw wins (%.4f vs %.4f, candidate shift %.2f st)",
        conf_raw, conf_cor, -shift,
    )
    return _decode_probs(probs_raw, hmm_raw, entry_raw), 0.0
```

Then rewire `extract_chords_ml` to use it and annotate key_info. Replace the
body from `raw = chordia_to_segments(...)` down:

```python
def extract_chords_ml(y, sr, pipeline=None):
    """ML chord extraction: lv-chordia on HPSS chord stem, raw segments."""
    get_chord_ml_model()

    if pipeline is None:
        from chord_pipeline import build_chord_pipeline_context
        pipeline = build_chord_pipeline_context(y, sr)

    from analyzer import extract_chords as extract_chords_classic
    from chord_engine_chordia import chordia_to_segments

    raw_segments, applied_shift = _chordia_segments_pitch_selected(
        pipeline.y_chord, sr
    )
    raw = chordia_to_segments(raw_segments)
    if not raw:
        logger.warning("lv-chordia returned no segments; falling back to classic stem HMM")
        return extract_chords_classic(y, sr, pipeline, bar_finalize=True)

    y_chord_for_key = pipeline.y_chord
    if applied_shift:
        # key estimation should see the same (corrected) stem the labels came from
        shifted = _shift_audio(pipeline.y_chord, sr, -applied_shift)
        if shifted is not None:
            y_chord_for_key = shifted

    segments, key_info = _attach_key_info(raw, y, sr, pipeline, y_chord=y_chord_for_key)
    key_info = dict(key_info or {})
    key_info["pitch_select_mode"] = os.getenv("CHORD_PITCH_SELECT", "confidence").lower().strip()
    if applied_shift:
        key_info["pitch_correction_semitones"] = round(applied_shift, 3)
    return segments, key_info
```

And extend `_attach_key_info` to accept the stem override:

```python
def _attach_key_info(segments, y, sr, pipeline, y_chord=None):
    from chord_pipeline import build_chord_pipeline_context

    if pipeline is None:
        pipeline = build_chord_pipeline_context(y, sr)

    y_chord = y_chord if y_chord is not None else pipeline.y_chord
    y_harmonic, chroma, chroma_low, chroma_mid = _extract_chroma_stack(
        y, sr, y_harmonic=y_chord,
    )
    chroma_mean = chroma.mean(axis=1)
    key_root, is_major, mode = _resolve_song_key(
        chroma_mean, segments, chroma, chroma_low, chroma_mid, y_harmonic, sr,
    )
    key_info = _build_key_info(
        key_root,
        is_major,
        mode=mode if mode not in ("major", "minor") else None,
    )
    return segments, key_info
```

Remove the now-unused `recognize_chordia` import from this module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_select.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Run the full suite**

Run: `cd backend && .venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/chord_engine_ml.py backend/tests/test_pitch_select.py
git commit -m "feat: decode-both pitch selection in ML path (v50 levers 1+2)"
```

---

### Task 4: Stop pre-correcting the full mix on the ML path (chord_engine.py)

**Files:**
- Modify: `backend/chord_engine.py:44-61`
- Test: `backend/tests/test_pitch_select.py` (append two tests)

**Interfaces:**
- Consumes: Task 3's `extract_chords_ml` (self-annotates `pitch_correction_semitones`).
- Produces: `extract_chords(y, sr, pipeline=None, *, return_pipeline=False)` unchanged signature. ML path no longer calls `normalize_pitch` (pipeline builds from raw audio; selection happens on the stem inside `extract_chords_ml`). Classic path keeps the (now gate-free) `normalize_pitch`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_pitch_select.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_select.py -v -k "precorrect or classic_path"`
Expected: `test_ml_path_does_not_precorrect_full_mix` FAILS (AssertionError from boom)

- [ ] **Step 3: Implement**

In `backend/chord_engine.py`, replace lines 44-61 (`extract_chords` head +
`_annotate_key_info`) with:

```python
def extract_chords(y, sr, pipeline=None, *, return_pipeline=False):
    log_chord_engine_status()
    pitch_shift = 0.0
    if PITCH_CORRECT and CHORD_ENGINE != "ml":
        # ML path selects pitch per-branch inside extract_chords_ml (v50);
        # only the classic engine still uses whole-mix pre-correction.
        y, pitch_shift = normalize_pitch(y, sr)
        if pitch_shift:
            logger.info("Pitch-corrected audio by %.2f semitones before chord analysis", pitch_shift)

    from chord_pipeline import build_chord_pipeline_context
    pipeline = pipeline or build_chord_pipeline_context(y, sr)

    def _annotate_key_info(key_info):
        key_info = dict(key_info or {})
        key_info["stem_method"] = pipeline.stems.method
        key_info["chord_pipeline"] = "stems+beats+bars"
        if pitch_shift and "pitch_correction_semitones" not in key_info:
            key_info["pitch_correction_semitones"] = round(pitch_shift, 3)
        return key_info
```

Everything from `if CHORD_ENGINE == "ml":` down is unchanged — the ML branch's
`key_info` now arrives with `pitch_correction_semitones` already set by
`extract_chords_ml` when a correction won, and `_annotate_key_info` no longer
overwrites it. Also update the module docstring line 5 to:
`ML default: lv-chordia on HPSS chord stem + decode-both pitch selection.`

One more caller check: `PITCH_CORRECT=0` env must fully disable correction on
both paths. On the ML path that means `CHORD_PITCH_SELECT=off` — add at the
top of `extract_chords`, right after `log_chord_engine_status()`:

```python
    if not PITCH_CORRECT:
        os.environ["CHORD_PITCH_SELECT"] = "off"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_pitch_select.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Run the full suite**

Run: `cd backend && .venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures`
Expected: all pass (phase5/phase6 tests exercise extract_chords fallback paths — if any fail, they are asserting the old pre-correction; fix the test only if it monkeypatches `normalize_pitch` on the ML path, otherwise fix the implementation)

- [ ] **Step 6: Commit**

```bash
git add backend/chord_engine.py backend/tests/test_pitch_select.py
git commit -m "feat: ML path builds pipeline from raw audio; classic keeps normalize_pitch (v50)"
```

---

### Task 5: rebaseline_v50.py + Makefile target

**Files:**
- Create: `backend/scripts/rebaseline_v50.py`
- Modify: `Makefile` (add `phase-v50` target next to `phase13-v49`)

**Interfaces:**
- Consumes: `eval_gold_mir.main()` CLI (same invocation as `rebaseline_v49.py:20-46`).
- Produces: `BASELINE_mir_gold_{DEV,TEST}_v50<tag>.json` in `backend/analysis/`, diff vs v49, `BASELINE_v50_README.md` on final run.
- CLI: `rebaseline_v50.py --select confidence|tta|off --margin 0.0 --dict submission --split dev|test|both --tag ""`

- [ ] **Step 1: Write the script**

Create `backend/scripts/rebaseline_v50.py`:

```python
#!/usr/bin/env python3
"""v50 — decode-both pitch selection rebaseline + diff vs v49.

Examples:
  .venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --tag _conf
  .venv/bin/python scripts/rebaseline_v50.py --split dev --select tta --tag _tta
  .venv/bin/python scripts/rebaseline_v50.py --split dev --dict extended --tag _dict_extended
  .venv/bin/python scripts/rebaseline_v50.py --split both --select confidence --margin 0.005 --tag ""
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ANALYSIS = BACKEND / "analysis"
V49 = {
    "dev": ANALYSIS / "BASELINE_mir_gold_DEV_v49.json",
    "test": ANALYSIS / "BASELINE_mir_gold_TEST_v49.json",
}
GUARD_TRACK = "another-one-bites-the-dust"
GUARD_MIN = 0.128


def _run_eval(split: str, out_path: Path, args) -> dict:
    os.environ["CHORD_ENGINE"] = "ml"
    os.environ["CHORD_ENGINE_STRICT"] = "1"
    os.environ["CHORD_ML_POSTPROCESS"] = "raw"
    os.environ["CHORD_ML_MODEL"] = "chordia"
    os.environ["CHORD_PITCH_SELECT"] = args.select
    os.environ["CHORD_PITCH_CONF_MARGIN"] = str(args.margin)
    os.environ["CHORD_CHORDIA_DICT"] = args.dict
    argv = [
        "eval_gold_mir.py", "--no-cache", "--split", split,
        "--require-audio-identity", "--write", str(out_path),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        import eval_gold_mir
        rc = eval_gold_mir.main()
        if rc != 0:
            raise RuntimeError(f"eval_gold_mir.py --split {split} exited {rc}")
    finally:
        sys.argv = old_argv
    return json.loads(out_path.read_text())


def _diff_split(name: str, v50: dict) -> dict:
    v49 = json.loads(V49[name].read_text())
    by49 = {t["id"]: t for t in v49["tracks"]}
    rows = []
    for t in v50["tracks"]:
        b = by49.get(t["id"], {})
        rows.append({
            "id": t["id"],
            "v49_majmin": b.get("majmin"),
            "v50_majmin": t["majmin"],
            "delta_majmin": round(t["majmin"] - b.get("majmin", 0), 4) if b else None,
        })
    rows.sort(key=lambda r: -(r["delta_majmin"] or 0))
    return {
        "split": name,
        "v49_summary": v49["summary"],
        "v50_summary": v50["summary"],
        "delta_majmin_agg": round(
            v50["summary"]["avg_majmin_wcsr"] - v49["summary"]["avg_majmin_wcsr"], 4,
        ),
        "tracks": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="confidence", choices=["confidence", "tta", "off"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--dict", default="submission",
                    choices=["submission", "ismir2017", "full", "extended"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "both"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    diffs = {}
    for split in splits:
        out = ANALYSIS / f"BASELINE_mir_gold_{split.upper()}_v50{args.tag}.json"
        print(f"Evaluating {split.upper()} → {out.name} "
              f"(select={args.select} margin={args.margin} dict={args.dict})…")
        result = _run_eval(split, out, args)
        diffs[split] = _diff_split(split, result)
        print(f"  {split.upper()} majmin {result['summary']['avg_majmin_wcsr']:.3f} "
              f"(Δ vs v49 {diffs[split]['delta_majmin_agg']:+.4f})")
        guard = next((t for t in result["tracks"] if GUARD_TRACK in t["id"]), None)
        if guard is not None and guard["majmin"] < GUARD_MIN:
            print(f"  WARNING: {GUARD_TRACK} regressed to {guard['majmin']:.3f} "
                  f"(< {GUARD_MIN}) — acceptance criterion violated")

    diff_out = ANALYSIS / f"BASELINE_v49_vs_v50{args.tag}.json"
    diff_out.write_text(json.dumps(diffs, indent=2) + "\n")
    print(f"Wrote {diff_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-check argument parsing (no eval)**

Run: `cd backend && .venv/bin/python scripts/rebaseline_v50.py --help`
Expected: usage text listing `--select --margin --dict --split --tag`

- [ ] **Step 3: Add Makefile target**

In `Makefile`, after the `phase13-v49` target (line 24-26), add (match the
existing target's `cd`/`$(PYTHON)` style exactly):

```makefile
phase-v50:
	cd $(BACKEND) && \
		$(PYTHON) scripts/rebaseline_v50.py --split both --select confidence
```

And add `phase-v50` to the `.PHONY` line.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/rebaseline_v50.py Makefile
git commit -m "feat: v50 rebaseline script with select-mode/margin/dict knobs"
```

---

### Task 6: DEV experiments → pick winner → TEST confirm → document

This task is measurement, not code. Each eval run is 14 tracks (DEV) of
5-checkpoint chordia inference on CPU — expect ~20-60 min per run; run them
sequentially in the background and check `another-one-bites-the-dust` per run.
Gold audio (83 files) is already in `backend/downloads/`; if a track is
missing, `make verify-gold-audio` will say so.

**Files:**
- Create: `backend/analysis/BASELINE_mir_gold_DEV_v50_*.json` (one per experiment)
- Create: `backend/analysis/BASELINE_mir_gold_{DEV,TEST}_v50.json` (winner)
- Create: `backend/analysis/BASELINE_v50_README.md`
- Modify: `backend/ML_SETUP.md` (serving env defaults section)

- [ ] **Step 1: Baseline sanity — confidence mode, margin 0, DEV**

Run: `cd backend && .venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --tag _conf`
Record DEV majmin and the guard-track warning (if any). Expected: ≥ 0.761 (v49 DEV) — decode-both should only ever replace a correction decision with a measured one.

- [ ] **Step 2: TTA mode, DEV**

Run: `cd backend && .venv/bin/python scripts/rebaseline_v50.py --split dev --select tta --tag _tta`

- [ ] **Step 3: Chord-dict sweep, DEV (lever 3 — three runs)**

```bash
cd backend
.venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --dict ismir2017 --tag _dict_ismir2017
.venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --dict full --tag _dict_full
.venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --dict extended --tag _dict_extended
```

Note: non-default dicts change the label vocabulary; `eval_gold_mir` maps via
Harte so scores stay comparable. If a dict run crashes on label mapping, record
it as ineligible and move on — do not patch mapping code inside this task.

- [ ] **Step 4: Margin sweep only if confidence mode shows guard-track regression**

If `another-one-bites-the-dust` warned in Step 1, rerun with
`--margin 0.005 --tag _conf_m005` and `--margin 0.01 --tag _conf_m01`; pick the
smallest margin that restores the guard without losing aggregate DEV.

- [ ] **Step 5: Pick the winner on DEV**

Winner = highest DEV `avg_majmin_wcsr` among eligible runs (guard track OK).
If no run beats v49 DEV (0.761), STOP — report results to the user instead of
running TEST; the spec's fallback (double full pipeline instead of stem-shift)
becomes the next candidate and needs a fresh decision.

- [ ] **Step 6: Confirm on TEST — one run only**

Run with the winning flags, e.g.:
`cd backend && .venv/bin/python scripts/rebaseline_v50.py --split both --select confidence --margin 0.0 --dict submission --tag ""`
Acceptance: TEST ≥ 0.76 target (≥ 0.743 = beats v48 is the minimum shippable), guard track ≥ 0.128, DEV reported alongside.

- [ ] **Step 7: Write BASELINE_v50_README.md**

Create `backend/analysis/BASELINE_v50_README.md` following the v49 README
format: winner flags, DEV/TEST majmin vs v49/v48, per-track deltas for
`another-one-bites-the-dust`, `let-it-be`, `help`, `yellow-submarine`, one
paragraph on why decode-both replaced the gate, and the full experiment table
(every `_tag` run with its DEV score).

- [ ] **Step 8: Update ML_SETUP.md serving defaults**

Document `CHORD_PITCH_SELECT` (default `confidence`), `CHORD_PITCH_CONF_MARGIN`
(default from winner), `CHORD_CHORDIA_DICT` (winner value), and that
`CHORD_PITCH_CORRECT=0` maps to `CHORD_PITCH_SELECT=off` on the ML path.

- [ ] **Step 9: Commit**

```bash
git add backend/analysis/BASELINE_*v50* backend/analysis/BASELINE_v50_README.md backend/ML_SETUP.md
git commit -m "docs: v50 baselines — decode-both pitch selection results"
```

---

## Self-Review Notes

- Spec coverage: lever 0 → Task 1; lever 1 → Tasks 2-4; lever 2 → Task 3 (tta mode); lever 3 → Task 6 Step 3; measurement protocol + acceptance → Tasks 5-6; stem-shift-vs-full-pipeline fallback → Task 6 Step 5 stop condition. Phase B is explicitly out of scope (separate spec).
- Types checked: `recognize_chordia_probs` tuple order (probs, hmm, entry) is consistent across Tasks 1 and 3; `pitch_shift_audio(y, sr, shift)` applies `-shift` (matches old `_pitch_shift` semantics) — Task 3's un-shift call passes `-applied_shift` accordingly.
- Known judgment call: key estimation on a detuned winner uses the corrected stem but the raw full mix (`_extract_chroma_stack(y, ...)`); v48 corrected both. If DEV shows key-related regressions on detuned tracks, the Task 6 Step 5 stop condition catches it.
