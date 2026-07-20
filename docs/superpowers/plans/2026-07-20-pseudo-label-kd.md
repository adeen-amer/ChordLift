# Pseudo-labeling / KD on FMA audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pseudo-labeling pipeline that expands the 161-track chordia fine-tune corpus with confidence-filtered pseudo-labels generated from FMA (Free Music Archive) audio, reusing the existing serving inference path and training pipeline unchanged.

**Architecture:** A new staged CLI script, `backend/chord_training/pseudo_label.py`, mirroring `fetch_train_data.py`'s `--stage` pattern: `fetch` (sample + download FMA audio), `label` (run the production chordia ensemble via `chord_engine_chordia.recognize_chordia_probs`, drop low-confidence segments, write `.lab` files), `manifest` (merge pseudo pairs into `train_manifest.txt`, gated by the existing gold-leakage verifier). `finetune.py` and `dataset.py` are not modified — the confidence filter operates entirely upstream, and `encode_labels`'s existing "no segment covers this frame → X" masking absorbs the dropped segments for free.

**Tech Stack:** Python 3.11, `requests` (FMA API — already installed transitively, `pip` shows `requests==2.34.2` in `backend/.venv`), `librosa`/`numpy` (already required), `pytest`.

## Global Constraints

- No changes to `chord_training/dataset.py` or `chord_training/finetune.py` — spec requires the confidence filter to live entirely in the new script (design doc, "Design" section).
- Hard pseudo-labels only, no soft-label/KD loss (spec Non-goals).
- No genre-targeted FMA sampling — broad random sample only (spec Non-goals).
- Target sample size ~150-250 FMA tracks (spec Design, `--stage fetch`).
- Segment confidence threshold and minimum retained-coverage are CLI flags, not hardcoded (spec Design: "default TBD empirically").
- Minimum retained coverage to keep a pseudo-labeled track: ~50% of duration (spec Design, `--stage label`).
- FMA audio is training input only, never redistributed (spec, Licensing note).
- The merged manifest must pass the existing `scripts/verify_training_leakage.py` gold-holdout check before being usable (repo-wide protocol, `build_manifest.py:40` precedent — not explicitly in the spec but required by the same gold-leakage rule every other manifest-building path in this repo follows).

---

### Task 0: FMA API access spike (manual gate, no code committed)

**Files:** none (verification only — nothing in this task is committed)

**Interfaces:**
- Produces: confirmation that the FMA metadata API (`https://freemusicarchive.org/api/get/tracks.json`) and file host (`https://files.freemusicarchive.org/`) are live and reachable with a real API key, and the exact shape of a `get_track` response (specifically the `track_file` and `track_duration` fields Task 4 depends on). If this fails, STOP and rescope (spec Risk: "FMA full-length archive access/fetch friction").

- [ ] **Step 1: Request an FMA API key**

Go to https://freemusicarchive.org/api/agreement, request a key, and set it in your shell:

```bash
export FMA_KEY=your_key_here
```

- [ ] **Step 2: Verify the metadata endpoint**

Create a scratch file `backend/_fma_spike.py` (do not commit this file):

```python
import os
import requests

key = os.environ["FMA_KEY"]
r = requests.get(
    "https://freemusicarchive.org/api/get/tracks.json",
    params={"track_id": 2, "api_key": key},
    timeout=30,
)
r.raise_for_status()
payload = r.json()
print("errors:", payload.get("errors"))
track = payload["dataset"][0]
print("track_id:", track.get("track_id"))
print("track_file:", track.get("track_file"))
print("track_duration:", track.get("track_duration"))
```

Run: `python backend/_fma_spike.py`

Expected: prints `errors: []` (or `None`), a non-empty `track_file` path string, and a numeric `track_duration`. If this raises (auth error, 404, connection error), the API or key is not usable as documented — STOP and rescope Task 4/5 before continuing (do not guess at a different endpoint shape).

- [ ] **Step 3: Verify the audio download**

Append to the same scratch file and re-run:

```python
audio = requests.get("https://files.freemusicarchive.org/" + track["track_file"], timeout=60)
audio.raise_for_status()
print("bytes:", len(audio.content))
```

Expected: `bytes:` > 0 (a few MB for a real track). Delete `backend/_fma_spike.py` once both steps pass — it's a one-time verification, not part of the shipped pipeline.

- [ ] **Step 4: Build the flat track pool CSV**

The official `fma_metadata.zip` (`https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`, ~342MB) ships `tracks.csv` with a 3-row multi-index header, which Task 4's `select_random_track_ids` deliberately does not parse directly (too easy to get the multi-index wrong without a real file to test against). Instead, derive a flat two-column CSV once:

```python
import pandas as pd

tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1, 2])
pool = tracks[("track", "", "duration")].dropna()
pool.to_frame("duration_sec").rename_axis("track_id").to_csv(
    "backend/chord_training/data/fma_pool.csv"
)
```

(Adjust the tuple key to match whatever the real header rows resolve to — inspect `tracks.columns.tolist()` first if `("track", "", "duration")` doesn't match; `pandas` is already installed in `backend/.venv`.) Confirm the output has a header `track_id,duration_sec` and a few thousand rows before moving on.

---

### Task 1: Confidence-filtering pure functions

**Files:**
- Create: `backend/chord_training/pseudo_label.py`
- Test: `backend/tests/test_pseudo_label.py`

**Interfaces:**
- Produces: `frame_confidences(probs: list[np.ndarray]) -> np.ndarray`, `segment_frame_range(seg: dict, sr: int, hop: int, n_frames: int) -> tuple[int, int]`, `segment_mean_confidence(seg: dict, frame_conf: np.ndarray, sr: int, hop: int) -> float`, `filter_low_confidence_segments(segments: list[dict], frame_conf: np.ndarray, sr: int, hop: int, threshold: float) -> list[dict]`. `seg` dicts use the same `{"start_time": float, "end_time": float, "chord": str}` shape `chord_engine_chordia.decode_chordia_probs` already returns.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_pseudo_label.py`:

```python
"""backend/tests/test_pseudo_label.py

Task 3 adds a module-level `pytest.importorskip("lv_chordia")` once
label_track needs it, same convention as test_chord_training_finetune.py:
the whole file skips together when lv_chordia/torch aren't installed,
rather than gating individual tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "chord_training"))


def test_frame_confidences_is_max_over_classes():
    from pseudo_label import frame_confidences

    probs = [np.array([[0.1, 0.9], [0.6, 0.4]])]
    assert list(frame_confidences(probs)) == pytest.approx([0.9, 0.6])


def test_segment_mean_confidence_averages_covered_frames():
    from pseudo_label import segment_mean_confidence

    frame_conf = np.array([0.2, 0.8, 0.8, 0.2])
    # hop=100, sr=1000 -> 0.1s/frame; segment [0.1s, 0.3s) covers frames 1,2
    seg = {"start_time": 0.1, "end_time": 0.3}
    assert segment_mean_confidence(seg, frame_conf, sr=1000, hop=100) == pytest.approx(0.8)


def test_filter_low_confidence_segments_drops_below_threshold():
    from pseudo_label import filter_low_confidence_segments

    frame_conf = np.array([0.9, 0.9, 0.1, 0.1])
    segs = [
        {"start_time": 0.0, "end_time": 0.2, "chord": "C:maj"},
        {"start_time": 0.2, "end_time": 0.4, "chord": "X"},
    ]
    kept = filter_low_confidence_segments(segs, frame_conf, sr=1000, hop=100, threshold=0.5)
    assert kept == [segs[0]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: `ModuleNotFoundError: No module named 'pseudo_label'` (the file doesn't exist yet).

- [ ] **Step 3: Write the module with the confidence-filtering functions**

Create `backend/chord_training/pseudo_label.py`:

```python
#!/usr/bin/env python3
"""backend/chord_training/pseudo_label.py

Pseudo-labels FMA audio using the production chordia ensemble, so the
161-track hand-labeled fine-tune corpus (BASELINE_v51_README.md) can be
supplemented without hand-annotating more Harte labels. Staged like
fetch_train_data.py:

  --stage fetch     sample track ids from a flat (track_id, duration_sec)
                     pool CSV and download audio via the FMA API.
  --stage label     run recognize_chordia_probs on each fetched track,
                     drop low-confidence segments, write .lab files.
  --stage manifest  merge the pseudo (audio, lab) pairs into an existing
                     train_manifest.txt, gated by verify_training_leakage.py.

dataset.py/finetune.py are unchanged: encode_labels already treats "no
segment covers this frame" as masked (X), so dropped low-confidence
segments are handled by the existing pipeline for free.

Built up incrementally (Tasks 1-5); imports and module-level constants are
added in the task that first needs them, not all upfront, so every
commit's file stays free of unused imports.
"""
from __future__ import annotations

import numpy as np


# ---- confidence filtering ---------------------------------------------

def frame_confidences(probs: list[np.ndarray]) -> np.ndarray:
    """Per-frame confidence: max triad posterior, same quantity
    chord_engine_chordia.chordia_confidence averages over the whole track."""
    return np.asarray(probs[0]).max(axis=1)


def segment_frame_range(seg: dict, sr: int, hop: int, n_frames: int) -> tuple[int, int]:
    """Frame index range [lo, hi) covered by a decoded segment.

    Mirrors dataset.py:encode_labels's forward mapping (frame f covers
    time f*hop/sr), inverted: t -> f = t*sr/hop.
    """
    lo = int(seg["start_time"] * sr / hop)
    hi = int(seg["end_time"] * sr / hop)
    lo = max(0, min(lo, n_frames))
    hi = max(lo, min(hi, n_frames))
    return lo, hi


def segment_mean_confidence(seg: dict, frame_conf: np.ndarray, sr: int, hop: int) -> float:
    lo, hi = segment_frame_range(seg, sr, hop, len(frame_conf))
    if hi <= lo:
        return 0.0
    return float(frame_conf[lo:hi].mean())


def filter_low_confidence_segments(
    segments: list[dict], frame_conf: np.ndarray, sr: int, hop: int, threshold: float,
) -> list[dict]:
    """Drop any segment whose mean frame confidence is below threshold.

    Dropped segments simply aren't written to the .lab file; the caller
    (label_track) relies on encode_labels's existing no-segment-covers-
    this-frame -> X masking to keep the loss ignoring these frames.
    """
    return [
        s for s in segments
        if segment_mean_confidence(s, frame_conf, sr, hop) >= threshold
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/pseudo_label.py backend/tests/test_pseudo_label.py
git commit -m "feat: confidence-filtering functions for FMA pseudo-labeling"
```

---

### Task 2: `.lab` writer and coverage calculation

**Files:**
- Modify: `backend/chord_training/pseudo_label.py`
- Modify: `backend/tests/test_pseudo_label.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `write_lab(path: str, segments: list[dict]) -> None`, `retained_coverage(segments: list[dict], track_duration: float) -> float`. `dataset.py:read_lab` (existing, `list[tuple[float, float, str]]`) is the consumer that must round-trip `write_lab`'s output.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_pseudo_label.py`:

```python
def test_write_lab_round_trips_through_read_lab(tmp_path):
    from dataset import read_lab
    from pseudo_label import write_lab

    segs = [
        {"start_time": 0.0, "end_time": 1.5, "chord": "C:maj"},
        {"start_time": 1.5, "end_time": 3.0, "chord": "G:maj"},
    ]
    path = tmp_path / "out.lab"
    write_lab(str(path), segs)
    assert read_lab(str(path)) == [(0.0, 1.5, "C:maj"), (1.5, 3.0, "G:maj")]


def test_retained_coverage_fraction():
    from pseudo_label import retained_coverage

    segs = [{"start_time": 0.0, "end_time": 3.0}, {"start_time": 5.0, "end_time": 6.0}]
    assert retained_coverage(segs, track_duration=10.0) == pytest.approx(0.4)


def test_retained_coverage_zero_duration_is_zero():
    from pseudo_label import retained_coverage

    assert retained_coverage([], track_duration=0.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v -k "write_lab or retained_coverage"`
Expected: `ImportError: cannot import name 'write_lab'` (and `retained_coverage`).

- [ ] **Step 3: Implement**

First, add `Path` to the imports at the top of `backend/chord_training/pseudo_label.py` (needed by `write_lab`):

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
```

Then add, after `filter_low_confidence_segments`:

```python
# ---- .lab writing + coverage -------------------------------------------

def write_lab(path: str, segments: list[dict]) -> None:
    lines = [f"{s['start_time']}\t{s['end_time']}\t{s['chord']}\n" for s in segments]
    Path(path).write_text("".join(lines))


def retained_coverage(segments: list[dict], track_duration: float) -> float:
    if track_duration <= 0:
        return 0.0
    covered = sum(s["end_time"] - s["start_time"] for s in segments)
    return covered / track_duration
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/pseudo_label.py backend/tests/test_pseudo_label.py
git commit -m "feat: .lab writer and coverage calc for FMA pseudo-labeling"
```

---

### Task 3: `label_track` orchestration

**Files:**
- Modify: `backend/chord_training/pseudo_label.py`
- Modify: `backend/tests/test_pseudo_label.py`

**Interfaces:**
- Consumes: `frame_confidences`, `filter_low_confidence_segments`, `write_lab`, `retained_coverage` (Tasks 1-2); `chord_engine_chordia.recognize_chordia_probs`/`decode_chordia_probs` (existing, `backend/chord_engine_chordia.py:52-117`); `lv_chordia.settings.DEFAULT_SR`/`DEFAULT_HOP_LENGTH` (existing, same constants `dataset.py` uses).
- Produces: `label_track(audio_path: str, lab_out_path: str, threshold: float, min_coverage: float = 0.5) -> float | None` — returns retained-coverage fraction on success, `None` (and writes nothing) if coverage falls below `min_coverage`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_pseudo_label.py` (this needs `lv_chordia`, so gate the whole file at import time — add this near the top, right after the existing imports):

```python
lv_chordia = pytest.importorskip("lv_chordia")


def _sine_chord(sr=22050, duration=5.0):
    """C major triad — same recipe as tests/test_chordia_probs.py."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [261.63, 329.63, 392.00]
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return y.astype(np.float32)
```

Then add the tests:

```python
def test_label_track_threshold_zero_keeps_most_of_track(tmp_path):
    import soundfile as sf

    from pseudo_label import label_track

    y = _sine_chord()
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), y, 22050)
    lab = tmp_path / "clip.lab"

    coverage = label_track(str(wav), str(lab), threshold=0.0)

    assert coverage is not None
    assert coverage > 0.9
    assert lab.exists()


def test_label_track_impossible_threshold_returns_none(tmp_path):
    import soundfile as sf

    from pseudo_label import label_track

    y = _sine_chord()
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), y, 22050)
    lab = tmp_path / "clip.lab"

    coverage = label_track(str(wav), str(lab), threshold=1.01, min_coverage=0.5)

    assert coverage is None
    assert not lab.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v -k label_track`
Expected: `ImportError: cannot import name 'label_track'`.

- [ ] **Step 3: Implement**

First, add `sys` to the imports and introduce the `HERE`/`BACKEND` path constants (needed to make `chord_engine_chordia`, which lives in `backend/` not `backend/chord_training/`, importable):

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
```

Then add, after `retained_coverage`:

```python
# ---- track-level orchestration -----------------------------------------

def label_track(
    audio_path: str, lab_out_path: str, threshold: float, min_coverage: float = 0.5,
) -> float | None:
    """Pseudo-label one track with the production chordia ensemble.

    Returns retained-coverage fraction and writes lab_out_path on success;
    returns None (writes nothing) if confidence-filtering leaves less than
    min_coverage of the track's duration covered -- same skip-and-report
    convention dataset.py:build_storages uses for unencodable pairs.
    """
    import librosa

    sys.path.insert(0, str(BACKEND))
    from chord_engine_chordia import decode_chordia_probs, recognize_chordia_probs
    from lv_chordia.settings import DEFAULT_HOP_LENGTH as HOP, DEFAULT_SR as SR

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    probs, hmm, entry = recognize_chordia_probs(y, sr)
    segments = decode_chordia_probs(probs, hmm, entry)
    frame_conf = frame_confidences(probs)
    kept = filter_low_confidence_segments(segments, frame_conf, sr, HOP, threshold)
    coverage = retained_coverage(kept, duration)
    if coverage < min_coverage:
        return None
    write_lab(lab_out_path, kept)
    return coverage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: 8 passed. (This test is slow — real model inference on a 5s clip, similar runtime to `tests/test_chordia_probs.py`.)

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/pseudo_label.py backend/tests/test_pseudo_label.py
git commit -m "feat: label_track orchestration for FMA pseudo-labeling"
```

---

### Task 4: FMA client and track sampling

**Files:**
- Modify: `backend/chord_training/pseudo_label.py`
- Modify: `backend/tests/test_pseudo_label.py`

**Interfaces:**
- Consumes: the flat `track_id,duration_sec` pool CSV produced by Task 0 Step 4.
- Produces: `_track_download_url(track_file: str) -> str`, `class FreeMusicArchive` (`get_track(track_id: int) -> dict`, `download_track(track_file: str, dest_path: str) -> None`), `select_random_track_ids(pool_csv_path: str, n: int, min_duration_sec: float, seed: int) -> list[int]`, `fetch_fma_sample(pool_csv_path: str, api_key: str, dest_dir: str, n: int, min_duration_sec: float, seed: int) -> list[str]` (list of downloaded audio file paths).

Only `_track_download_url` and `select_random_track_ids` are unit tested — `FreeMusicArchive`/`fetch_fma_sample` make real network calls and are validated by the Task 0 spike + a manual run, the same convention `fetch_train_data.py` uses (it has zero unit tests; `build_manifest.py`/`dataset.py`/`finetune.py`, which operate on local data only, are the tested layer).

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_pseudo_label.py`:

```python
def test_track_download_url():
    from pseudo_label import _track_download_url

    assert (
        _track_download_url("some/path/track.mp3")
        == "https://files.freemusicarchive.org/some/path/track.mp3"
    )


def test_select_random_track_ids_filters_duration_and_is_deterministic(tmp_path):
    from pseudo_label import select_random_track_ids

    csv_path = tmp_path / "pool.csv"
    csv_path.write_text(
        "track_id,duration_sec\n"
        "1,30\n"
        "2,120\n"
        "3,200\n"
        "4,180\n"
    )

    ids = select_random_track_ids(str(csv_path), n=2, min_duration_sec=60.0, seed=42)

    assert len(ids) == 2
    assert 1 not in ids  # below min_duration_sec
    assert set(ids) <= {2, 3, 4}
    assert select_random_track_ids(str(csv_path), n=2, min_duration_sec=60.0, seed=42) == ids


def test_select_random_track_ids_caps_at_pool_size(tmp_path):
    from pseudo_label import select_random_track_ids

    csv_path = tmp_path / "pool.csv"
    csv_path.write_text("track_id,duration_sec\n1,120\n2,180\n")

    ids = select_random_track_ids(str(csv_path), n=10, min_duration_sec=60.0, seed=0)

    assert set(ids) == {1, 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v -k "download_url or select_random"`
Expected: `ImportError: cannot import name '_track_download_url'`.

- [ ] **Step 3: Implement**

First, add `csv`, `os`, `random`, and `requests` to the imports, and the FMA base-URL constants, right after the `HERE`/`BACKEND` constants:

```python
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/

FMA_API_BASE = "https://freemusicarchive.org/api/get/"
FMA_FILES_BASE = "https://files.freemusicarchive.org/"
```

Then add, before the confidence-filtering functions (`_track_download_url` through `fetch_fma_sample` are the FMA-facing layer; keeping them above `frame_confidences` groups "talks to the network" separately from "pure computation"):

```python
def _track_download_url(track_file: str) -> str:
    return FMA_FILES_BASE + track_file


class FreeMusicArchive:
    """Minimal client for the FMA metadata API -- only the two calls this
    pipeline needs, adapted from mdeff/fma's utils.py (MIT-licensed)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_track(self, track_id: int) -> dict:
        r = requests.get(
            f"{FMA_API_BASE}tracks.json",
            params={"track_id": track_id, "api_key": self.api_key},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            raise RuntimeError(f"FMA API error for track {track_id}: {payload['errors']}")
        return payload["dataset"][0]

    def download_track(self, track_file: str, dest_path: str) -> None:
        r = requests.get(_track_download_url(track_file), stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def select_random_track_ids(
    pool_csv_path: str, n: int, min_duration_sec: float, seed: int,
) -> list[int]:
    """Deterministic (given seed) random sample of track ids from a flat
    track_id,duration_sec CSV (Task 0 Step 4 derives this from the official
    FMA tracks.csv), filtered to tracks at least min_duration_sec long."""
    candidates = []
    with open(pool_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["duration_sec"]) >= min_duration_sec:
                candidates.append(int(row["track_id"]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def fetch_fma_sample(
    pool_csv_path: str, api_key: str, dest_dir: str, n: int, min_duration_sec: float, seed: int,
) -> list[str]:
    """Download a random FMA sample. Tracks that fail (API error, download
    error) are skipped and reported to stderr, same convention
    dataset.py:build_storages uses for unencodable pairs."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    track_ids = select_random_track_ids(pool_csv_path, n, min_duration_sec, seed)
    client = FreeMusicArchive(api_key)
    fetched = []
    for track_id in track_ids:
        dest = os.path.join(dest_dir, f"fma-{track_id}.mp3")
        if os.path.exists(dest):
            fetched.append(dest)
            continue
        try:
            meta = client.get_track(track_id)
            client.download_track(meta["track_file"], dest)
            fetched.append(dest)
        except Exception as exc:
            print(f"skipping fma track {track_id}: {exc}", file=sys.stderr)
    return fetched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/pseudo_label.py backend/tests/test_pseudo_label.py
git commit -m "feat: FMA client and random track sampling for pseudo-labeling"
```

---

### Task 5: CLI stages (`fetch` / `label` / `manifest`)

**Files:**
- Modify: `backend/chord_training/pseudo_label.py`
- Modify: `backend/tests/test_pseudo_label.py`

**Interfaces:**
- Consumes: `fetch_fma_sample` (Task 4), `label_track` (Task 3), `build_manifest.AUDIO_EXTS` (existing, `backend/chord_training/build_manifest.py:15`), `scripts/verify_training_leakage.py` (existing, invoked as a subprocess exactly like `build_manifest.py:40` already does).
- Produces: `merge_manifests(train_manifest_path: str, pseudo_manifest_path: str) -> list[str]` (returns the newly-added lines) and `main() -> int`, the `--stage fetch|label|manifest` CLI entrypoint.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_pseudo_label.py`:

```python
def test_merge_manifests_appends_new_and_skips_duplicates(tmp_path):
    from pseudo_label import merge_manifests

    train = tmp_path / "train_manifest.txt"
    train.write_text("a.wav\ta.lab\n")
    pseudo = tmp_path / "pseudo_manifest.txt"
    pseudo.write_text("a.wav\ta.lab\nb.wav\tb.lab\n")

    added = merge_manifests(str(train), str(pseudo))

    assert added == ["b.wav\tb.lab"]
    assert train.read_text().splitlines() == ["a.wav\ta.lab", "b.wav\tb.lab"]


def test_cli_label_stage_writes_manifest(tmp_path):
    import subprocess
    import sys as _sys

    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    y = _sine_chord(duration=5.0)
    sf.write(str(audio_dir / "fma-1.wav"), y, 22050)

    manifest_path = tmp_path / "pseudo_manifest.txt"
    r = subprocess.run(
        [_sys.executable, str(BACKEND / "chord_training" / "pseudo_label.py"),
         "--stage", "label", "--data-dir", str(audio_dir),
         "--confidence-threshold", "0.0", "--pseudo-manifest", str(manifest_path)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert manifest_path.exists()
    lines = manifest_path.read_text().splitlines()
    assert len(lines) == 1
    audio_path, lab_path = lines[0].split("\t")
    assert Path(lab_path).exists()


def test_cli_manifest_stage_merges_and_passes_leakage_check(tmp_path):
    import subprocess
    import sys as _sys

    train = tmp_path / "train_manifest.txt"
    train.write_text("")
    pseudo = tmp_path / "pseudo_manifest.txt"
    (tmp_path / "fma-1.wav").write_text("x")
    (tmp_path / "fma-1.lab").write_text("0.0\t1.0\tC:maj\n")
    pseudo.write_text(f"{tmp_path / 'fma-1.wav'}\t{tmp_path / 'fma-1.lab'}\n")

    r = subprocess.run(
        [_sys.executable, str(BACKEND / "chord_training" / "pseudo_label.py"),
         "--stage", "manifest", "--train-manifest", str(train),
         "--pseudo-manifest", str(pseudo)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "fma-1.wav" in train.read_text()
    assert "PASS" in r.stdout or "OK" in r.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v -k "merge_manifests or cli_"`
Expected: `ImportError: cannot import name 'merge_manifests'` for the first test; the two subprocess tests fail with a non-zero return code (argparse error, no `--stage` handling yet).

- [ ] **Step 3: Implement**

First, add `argparse` and `subprocess` to the imports, and the `DATA_DIR`/`VERIFY_SCRIPT` constants, right after the `FMA_API_BASE`/`FMA_FILES_BASE` constants:

```python
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
DATA_DIR = HERE / "data"
VERIFY_SCRIPT = BACKEND / "scripts" / "verify_training_leakage.py"

FMA_API_BASE = "https://freemusicarchive.org/api/get/"
FMA_FILES_BASE = "https://files.freemusicarchive.org/"
```

Then add, after `fetch_fma_sample`:

```python
# ---- manifest merging ----------------------------------------------------

def merge_manifests(train_manifest_path: str, pseudo_manifest_path: str) -> list[str]:
    """Append pseudo-labeled (audio, lab) pairs into an existing
    train_manifest.txt, skipping lines already present."""
    with open(train_manifest_path) as f:
        existing = {line.strip() for line in f if line.strip()}
    with open(pseudo_manifest_path) as f:
        candidates = [line.strip() for line in f if line.strip()]
    to_add = [line for line in candidates if line not in existing]
    with open(train_manifest_path, "a") as f:
        for line in to_add:
            f.write(line + "\n")
    return to_add


# ---- CLI ------------------------------------------------------------------

def main() -> int:
    from build_manifest import AUDIO_EXTS

    ap = argparse.ArgumentParser(description="Pseudo-label FMA audio for chordia fine-tuning")
    ap.add_argument("--stage", required=True, choices=["fetch", "label", "manifest"])
    ap.add_argument("--pool-csv", help="fetch stage: flat track_id,duration_sec CSV (Task 0 Step 4)")
    ap.add_argument("--api-key", default=os.environ.get("FMA_KEY", ""), help="fetch stage: FMA API key")
    ap.add_argument("--data-dir", default=str(DATA_DIR / "pseudo"),
                     help="fetch: download destination; label: audio source + .lab output dir")
    ap.add_argument("--n", type=int, default=200, help="fetch stage: sample size")
    ap.add_argument("--min-duration", type=float, default=60.0,
                     help="fetch stage: minimum track duration in seconds")
    ap.add_argument("--seed", type=int, default=0, help="fetch stage: sampling seed")
    ap.add_argument("--confidence-threshold", type=float, default=0.7,
                     help="label stage: minimum segment confidence to keep")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                     help="label stage: minimum retained-duration fraction to keep a track")
    ap.add_argument("--pseudo-manifest", default=str(DATA_DIR / "pseudo_manifest.txt"),
                     help="label stage output / manifest stage input")
    ap.add_argument("--train-manifest", help="manifest stage: train_manifest.txt to append into")
    args = ap.parse_args()

    if args.stage == "fetch":
        if not args.pool_csv:
            print("--pool-csv is required for the fetch stage", file=sys.stderr)
            return 1
        if not args.api_key:
            print("FMA_KEY env var or --api-key is required for the fetch stage", file=sys.stderr)
            return 1
        fetched = fetch_fma_sample(args.pool_csv, args.api_key, args.data_dir,
                                    args.n, args.min_duration, args.seed)
        print(f"fetched: {len(fetched)}/{args.n}")
        return 0

    if args.stage == "label":
        audio_files = sorted(
            p for ext in AUDIO_EXTS for p in Path(args.data_dir).glob(f"*{ext}")
        )
        kept = 0
        with open(args.pseudo_manifest, "w") as manifest:
            for audio_path in audio_files:
                lab_path = audio_path.with_suffix(".lab")
                coverage = label_track(str(audio_path), str(lab_path),
                                        args.confidence_threshold, args.min_coverage)
                if coverage is None:
                    print(f"skipping {audio_path.name}: retained coverage below "
                          f"{args.min_coverage}", file=sys.stderr)
                    continue
                manifest.write(f"{audio_path}\t{lab_path}\n")
                kept += 1
        print(f"labeled: {kept}/{len(audio_files)}")
        return 0

    # manifest stage
    if not args.train_manifest:
        print("--train-manifest is required for the manifest stage", file=sys.stderr)
        return 1
    added = merge_manifests(args.train_manifest, args.pseudo_manifest)
    print(f"merged {len(added)} new pairs into {args.train_manifest}")
    leak_rc = subprocess.run([sys.executable, str(VERIFY_SCRIPT), args.train_manifest]).returncode
    return 1 if leak_rc != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py -v`
Expected: 14 passed. (The two subprocess tests are slower — real model inference and a real subprocess call to `verify_training_leakage.py`.)

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/pseudo_label.py backend/tests/test_pseudo_label.py
git commit -m "feat: fetch/label/manifest CLI stages for FMA pseudo-labeling"
```

---

### Task 6: Dependency pin, runbook doc, full test sweep

**Files:**
- Modify: `backend/requirements-ml.txt`
- Create: `backend/chord_training/PSEUDO_LABEL_RUNBOOK.md`

**Interfaces:** none (documentation + dependency manifest only).

- [ ] **Step 1: Pin `requests` in requirements-ml.txt**

`requests` is already installed transitively (`backend/.venv` has `requests==2.34.2`, pulled in by `spotdl`), but `pseudo_label.py` now imports it directly, so it needs an explicit line. Add to `backend/requirements-ml.txt`, after the `spotdl==4.2.11` line:

```
requests  # direct import in chord_training/pseudo_label.py (FMA API client)
```

- [ ] **Step 2: Write the runbook doc**

Create `backend/chord_training/PSEUDO_LABEL_RUNBOOK.md`:

```markdown
# FMA pseudo-labeling RUNBOOK

Produces a `train_manifest.txt` supplement of confidence-filtered FMA
pseudo-labels, to be fed into `pack_bundle.py` and fine-tuned via the
existing `RUNBOOK.md` (desktop GPU fine-tune + eval) unchanged.

## 1. One-time setup (Task 0)

- Request an FMA API key: https://freemusicarchive.org/api/agreement
- Download `fma_metadata.zip` (~342MB) from
  https://os.unil.cloud.switch.ch/fma/fma_metadata.zip and extract
  `tracks.csv`.
- Derive the flat pool CSV `chord_training/data/fma_pool.csv`
  (`track_id,duration_sec`) per `pseudo_label.py`'s Task 0 Step 4 snippet.

## 2. Fetch a sample

```bash
cd backend
FMA_KEY=your_key_here python chord_training/pseudo_label.py \
    --stage fetch --pool-csv chord_training/data/fma_pool.csv \
    --n 200 --min-duration 60 --seed 0
```

Writes audio to `chord_training/data/pseudo/`. Re-runnable: already-fetched
files are skipped.

## 3. Label with confidence filtering

```bash
python chord_training/pseudo_label.py --stage label \
    --confidence-threshold 0.7 --min-coverage 0.5
```

Start with `--confidence-threshold 0.7`; before the real fine-tune run,
spot-check a handful of the written `.lab` files by ear against their
source audio (spec Risk: "confidence threshold too loose"). Adjust the
threshold and re-run this stage (cheap — no download needed) if the sample
looks wrong.

## 4. Merge into the real train manifest

```bash
python chord_training/pseudo_label.py --stage manifest \
    --train-manifest /path/to/train_manifest.txt
```

Fails (non-zero exit) if the merge introduces any gold-holdout leakage —
this should never happen with FMA (a disjoint catalog from Isophonics), but
the check runs unconditionally, same as every other manifest-building path
in this repo.

## 5. Fine-tune and evaluate

Follow the existing `RUNBOOK.md` unchanged, using the now-supplemented
`train_manifest.txt`. Evaluation protocol (DEV picks, guard track >= 0.128,
TEST once) is identical to v50/v51 — see the design spec's Evaluation
section.

## Licensing note

FMA audio is mixed-license (CC-BY, CC-BY-NC, CC0). Used here as training
input only, never redistributed — fine for this use, not a blocker.
```

- [ ] **Step 3: Run the full test suite**

Run: `cd backend && python -m pytest tests/test_pseudo_label.py tests/test_chord_training_finetune.py tests/test_chordia_probs.py -v`
Expected: all pass, no regressions in the existing training-pipeline tests.

- [ ] **Step 4: Commit**

```bash
git add backend/requirements-ml.txt backend/chord_training/PSEUDO_LABEL_RUNBOOK.md
git commit -m "docs: FMA pseudo-labeling runbook; pin requests in requirements-ml.txt"
```
