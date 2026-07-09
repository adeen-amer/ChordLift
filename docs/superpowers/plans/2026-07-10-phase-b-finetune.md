# Phase B Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Dispatch code-execution subagents with `model: sonnet` (Sonnet 5) per user request — EXCEPT Task 1 (exploratory spike), which needs a capable model.**

**Goal:** Fine-tune the five serving lv_chordia checkpoints on non-gold Isophonics (Billboard-guarded validation) so gold-v2 TEST majmin exceeds 0.750, via a training kit Adeen runs on his RTX 2060 Super.

**Architecture:** Spike-first. Task 1 proves warm-start + the data contract on this Mac (CPU) and commits its findings; Tasks 2–5 build the kit (`backend/chord_training/`) against those findings; Task 6 is the measured execution loop (Mac data build → desktop training via runbook → Mac rebaseline).

**Tech Stack:** Python 3.11 (`backend/.venv`), lv_chordia internals (`FramedH5DataStorage`, `FramedDataProvider`, `CQTPitchShifter`, `ComplexChordShifter`, `NetworkInterface.train_supervised`), librosa, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-10-phase-b-finetune-design.md`. Branch: `phase-b-finetune`.
- **Task 1 is a kill-switch:** if warm-start or the data contract cannot be proven, STOP the phase and report — do not proceed to Tasks 2–6.
- All backend commands: `cd backend && .venv/bin/python …`; tests: `.venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures`.
- Never edit anything under `backend/.venv/` — but READING package internals is expected; the packaged `.sdict` checkpoints live in `backend/.venv/share/lv-chordia/cache_data/`.
- Zero gold leakage: both manifests must pass `scripts/verify_training_leakage.py` against `analysis/gold_holdout_v2.json` before any training run.
- Gold protocol: DEV picks, TEST once at the very end, both reported. v50 numbers to beat: DEV 0.803, TEST 0.750; guard track `another-one-bites-the-dust` (DEV) ≥ 0.128.
- Fine-tuned checkpoints deploy ONLY via the `CHORD_CHORDIA_MODELS` env override (Task 4); packaged files stay untouched.
- Known package facts (verified): loading an existing `.sdict` sets `finalized=True` and `train_supervised` refuses to train; `ChordNet(cross_subpart_counter)` needs a counter for its reweighted loss — pickles ship at `lv_chordia/data/cross_subpart_weight{0..4}.pkl`; the original training entry is `lv_chordia/chordnet_ismir_naive.py` `__main__` (lines ~265–307) — crib from it, don't reinvent.

---

### Task 1: Warm-start + data-contract spike (Mac, CPU) — CAPABLE MODEL, NOT SONNET

**Files:**
- Create: `backend/chord_training/spike_warmstart.py` (runnable end-to-end proof)
- Create: `backend/chord_training/FINDINGS.md` (the data contract, written for the Task 3 implementer)
- Modify: `backend/chord_training/__init__.py` (docstring → `"""Phase B fine-tuning kit for the lv_chordia serving checkpoints."""`)

**Interfaces:**
- Consumes: package internals listed in Global Constraints.
- Produces: `FINDINGS.md` documenting, concretely and with code snippets that ran: (1) how to build the X storage (CQT with shift margin) and Y storage (complex-chord arrays) from an (audio.wav, harte.lab) pair; (2) provider wiring; (3) warm-start recipe; (4) how the new checkpoint is loaded back. Task 3 implements `dataset.py`/`finetune.py` FROM this file.

- [ ] **Step 1: Study the original training entry** — read `backend/.venv/lib/python3.11/site-packages/lv_chordia/chordnet_ismir_naive.py` `__main__` block, `storage_creation.py`, `mir/nn/data_storage.py` (`FramedH5DataStorage` write API), `mir/nn/data_provider.py` (`FramedDataProvider.link`), `extractors/cqt.py` (CQTV2, shift margin), `complex_chord.py` (Harte→complex-chord array). Goal: the exact recipe that turns (audio, .lab) into the two storages the provider links.

- [ ] **Step 2: Write `spike_warmstart.py`** — must do ALL of, on CPU, with 2 short audio clips (generate 20s sine-chord wavs in-script — C–F–G loop — and matching hand-written `.lab` files in a tempdir; no downloads):
  1. Build X/Y storages from the 2 (wav, lab) pairs.
  2. Wire providers exactly like the package `__main__` (crib the `FramedDataProvider`/`CQTPitchShifter`/`ComplexChordShifter` calls; `LSTM_TRAIN_LENGTH`, `SHIFT_LOW`, `SHIFT_HIGH`, `SPEC_DIM` import from `chordnet_ismir_naive`).
  3. Load `cross_subpart_weight0.pkl` from the package data dir; construct `ChordNet(counter)`.
  4. Warm-start: `NetworkInterface(ChordNet(counter), NEW_NAME, load_checkpoint=False)` where NEW_NAME = `'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft0_s0.best'`; then load pretrained weights from the packaged s0 `.sdict` via `torch.load(path, map_location='cpu')['net']` → `iface.net.load_state_dict(...)`; assert `iface.finalized is False`.
  5. `train_supervised(train_provider, val_provider, batch_size=2, learning_rates_dict=1e-4, round_per_val=-1, early_end_epochs=1, val_batch_size=1)` — must complete ≥1 training round and write the new `.sdict`.
  6. Reload proof: fresh `NetworkInterface(ChordNet(None), NEW_NAME, load_checkpoint=False)` finds the new `.sdict`, runs `.inference` on one clip's CQT, output shapes match the packaged model's 6-head contract.
  End with `print("SPIKE OK")` only if every assertion passed.

- [ ] **Step 3: Run it** — `cd backend && .venv/bin/python chord_training/spike_warmstart.py`. Expected: `SPIKE OK`. Iterate on failures; package quirks (dtype=np.int16 storages, meta files, `need_shuffle`, `subrange`) are the expected fight. If genuinely blocked (API cannot be made to train), STOP: report BLOCKED with the specific wall — do not fake it.

- [ ] **Step 4: Write `FINDINGS.md`** — the working recipe from Step 2, as prose + the exact code fragments that ran, including every quirk hit and its fix, storage dtypes, file layout on disk, and the Harte-label subset the encoder accepts (what `complex_chord` parsing rejects, how to handle `X`/`N` labels).

- [ ] **Step 5: Clean up spike artifacts** — ensure the spike writes storages/checkpoints under a tempdir or `chord_training/spike_out/` (gitignored via one line in `backend/.gitignore`), never into the shared `cache_data`. Exception: the warm-start test checkpoint MAY be written to a tempdir copy of cache layout — verify the packaged `cache_data` dir is byte-identical after the run (`git status` clean + no new files there).

- [ ] **Step 6: Commit**

```bash
git add backend/chord_training/ backend/.gitignore
git commit -m "feat: Phase B spike — warm-start + data contract proven (FINDINGS.md)"
```

---

### Task 2: Manifest builder + Billboard→lab converter

**Files:**
- Create: `backend/chord_training/build_manifest.py`
- Create: `backend/chord_training/billboard_to_lab.py`
- Test: `backend/tests/test_chord_training_data.py`

**Interfaces:**
- Consumes: `scripts/verify_training_leakage.py` (CLI: takes a manifest path, exits nonzero on gold overlap), `analysis/gold_holdout_v2.json`.
- Produces:
  - Manifest format (Tasks 3/5/6 rely on it): one line per track, `<abs_audio_path>\t<abs_lab_path>`.
  - `build_manifest.py` CLI: `--data-dir DIR --out FILE [--min-pairs N]` (default 10) — scans DIR for `<stem>.(mp3|wav|m4a)` + `<stem>.lab` pairs, writes the manifest, then runs `verify_training_leakage.py` on it and exits nonzero if leakage found or if pair count < 10; prints `pairs=<N> leakage=OK`.
  - `billboard_to_lab.py` CLI: `--annotations-dir DIR --out-labs DIR` — converts McGill Billboard `salami_chords.txt` files (format: `<time>\t<chord>` event lines with section markers) into Harte `.lab` lines `<start>\t<end>\t<chord>`; skips tracks whose annotation cannot be parsed; prints per-file `ok`/`skip` and a final count.

- [ ] **Step 1: Write the failing tests**

```python
"""backend/tests/test_chord_training_data.py"""
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_build_manifest_pairs_and_leakage(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for stem in ("track-a", "track-b"):
        (data / f"{stem}.mp3").write_bytes(b"\x00")
        (data / f"{stem}.lab").write_text("0.0\t1.0\tC:maj\n")
    (data / "orphan.mp3").write_bytes(b"\x00")  # no lab -> excluded
    out = tmp_path / "train_manifest.txt"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "build_manifest.py"),
         "--data-dir", str(data), "--out", str(out), "--min-pairs", "2"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    audio, lab = lines[0].split("\t")
    assert audio.endswith(".mp3") and lab.endswith(".lab")


def test_build_manifest_rejects_gold_track(tmp_path):
    gold_stem = "let-it-be"  # known gold-24 id; must trip the leakage gate
    data = tmp_path / "data"
    data.mkdir()
    (data / f"{gold_stem}.mp3").write_bytes(b"\x00")
    (data / f"{gold_stem}.lab").write_text("0.0\t1.0\tC:maj\n")
    out = tmp_path / "m.txt"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "build_manifest.py"),
         "--data-dir", str(data), "--out", str(out), "--min-pairs", "1"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode != 0


def test_billboard_to_lab_converts_events(tmp_path):
    ann = tmp_path / "ann" / "0003"
    ann.mkdir(parents=True)
    (ann / "salami_chords.txt").write_text(
        "# title: Example\n# artist: Example Artist\n"
        "0.0\tsilence\n"
        "0.5\tA, intro, | C:maj | F:maj |\n"
        "4.5\tB, verse, | G:maj | C:maj |\n"
        "8.5\tend\n"
    )
    out = tmp_path / "labs"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "billboard_to_lab.py"),
         "--annotations-dir", str(tmp_path / "ann"), "--out-labs", str(out)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    lab = (out / "0003.lab").read_text().strip().splitlines()
    # 4 bars over two 4s sections -> 4 chord segments, Harte labels preserved
    assert len(lab) == 4
    first_start, first_end, first_chord = lab[0].split("\t")
    assert first_chord == "C:maj"
    assert float(first_start) == pytest.approx(0.5)
    assert float(first_end) == pytest.approx(2.5)
```

Note for the implementer: the McGill format divides each annotation line's time span evenly among its `|`-delimited bars, one chord per bar in the simple case (`| C:maj | F:maj |` over 0.5–4.5s → C:maj 0.5–2.5, F:maj 2.5–4.5). Handle repeated-chord bars (`| C:maj |*2`? no — `x2` suffix means repeat the bar group: `| C:maj | F:maj | x2`) by expanding repeats; bars containing multiple chords (`| C:maj F:maj |`) split the bar evenly; `&pause`, `silence`, `end`, `noise` and section-only lines produce no chord segments; `N` passes through as `N`. Skip files that still fail with a warning — coverage report over correctness gambles.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chord_training_data.py -v`
Expected: FAIL (scripts don't exist)

- [ ] **Step 3: Implement both scripts** — `build_manifest.py` is ~60 lines (scan, pair, write, then `subprocess.run([sys.executable, "scripts/verify_training_leakage.py", manifest])` and propagate failure; `--min-pairs` default 10). `billboard_to_lab.py` is the parser described in the test note (~120 lines). No classes, no config objects.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chord_training_data.py -v`
Expected: 3 PASSED. Then full suite: `.venv/bin/python -m pytest tests/ -q --ignore=tests/fixtures` — all pass.

- [ ] **Step 5: Check `verify_training_leakage.py` CLI contract** — read the script; if its argument handling differs from `verify_training_leakage.py <manifest>` (e.g. expects `data/train_manifest.txt` fixed path), adapt `build_manifest.py`'s invocation, NOT the script.

- [ ] **Step 6: Commit**

```bash
git add backend/chord_training/build_manifest.py backend/chord_training/billboard_to_lab.py backend/tests/test_chord_training_data.py
git commit -m "feat: Phase B manifest builder + McGill Billboard lab converter"
```

---

### Task 3: dataset.py + finetune.py (from FINDINGS.md)

**Files:**
- Create: `backend/chord_training/dataset.py`
- Create: `backend/chord_training/finetune.py`
- Test: `backend/tests/test_chord_training_finetune.py`

**Interfaces:**
- Consumes: `backend/chord_training/FINDINGS.md` (Task 1's committed contract — READ IT FIRST; it contains the storage/provider recipe that actually ran), manifest format from Task 2.
- Produces:
  - `dataset.py`: `build_storages(pairs: list[tuple[str, str]], out_dir: str) -> tuple[str, str]` (returns X/Y storage paths; skips-and-reports pairs whose lab fails encoding) and `make_providers(storage_x_path: str, storage_y_path: str, train: bool) -> provider` (train=True → shift augmentation + shuffle per FINDINGS; train=False → val config).
  - `finetune.py` CLI: `--train-manifest F --val-manifest F --work-dir D --seeds 0,1,2,3,4 --lr 1e-4 --epochs-cap 30 --batch-size 8 --dry-run` → per seed: warm-start from packaged `…_s{i}.best`, save as `…_ft1_s{i}.best` under `D/cache_data/`; `--dry-run` builds storages + providers + loads weights + one forward/backward, no save; prints `SEED {i} DONE` per seed or `DRY-RUN OK`.
- Task 5's runbook invokes exactly this CLI; Task 6 evals the produced names `joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft1_s{i}.best`.

- [ ] **Step 1: Write the failing test** — end-to-end tiny run on CPU, reusing the spike's synthetic-audio approach:

```python
"""backend/tests/test_chord_training_finetune.py

CPU end-to-end: synthetic 20s clips -> storages -> dry-run -> 1-round train.
Slow (~1-3 min). Skipped when torch/lv_chordia missing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lv_chordia")
BACKEND = Path(__file__).resolve().parent.parent
PY = sys.executable


def _make_pair(d: Path, stem: str):
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 20.0, sr * 20, endpoint=False)
    freqs = [(261.63, 329.63, 392.0), (349.23, 440.0, 523.25)]  # C, F triads
    y = np.concatenate([
        sum(np.sin(2 * np.pi * f * t[: sr * 10]) for f in trio) / 3
        for trio in freqs
    ]).astype(np.float32)
    sf.write(str(d / f"{stem}.wav"), y, sr)
    (d / f"{stem}.lab").write_text("0.0\t10.0\tC:maj\n10.0\t20.0\tF:maj\n")


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    d = tmp_path_factory.mktemp("ft_data")
    for stem in ("clip-a", "clip-b"):
        _make_pair(d, stem)
    train = d / "train_manifest.txt"
    val = d / "val_manifest.txt"
    train.write_text(f"{d}/clip-a.wav\t{d}/clip-a.lab\n")
    val.write_text(f"{d}/clip-b.wav\t{d}/clip-b.lab\n")
    return train, val, d


def test_dry_run_ok(manifests, tmp_path):
    train, val, _ = manifests
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "finetune.py"),
         "--train-manifest", str(train), "--val-manifest", str(val),
         "--work-dir", str(tmp_path), "--seeds", "0", "--dry-run"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY-RUN OK" in r.stdout


def test_one_round_train_writes_checkpoint(manifests, tmp_path):
    train, val, _ = manifests
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "finetune.py"),
         "--train-manifest", str(train), "--val-manifest", str(val),
         "--work-dir", str(tmp_path), "--seeds", "0",
         "--lr", "1e-4", "--epochs-cap", "1", "--batch-size", "2"],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    assert "SEED 0 DONE" in r.stdout
    out = list((tmp_path / "cache_data").glob("*_ft1_s0.best*"))
    assert out, "fine-tuned checkpoint not written"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chord_training_finetune.py -v`
Expected: FAIL (`finetune.py` missing)

- [ ] **Step 3: Implement from FINDINGS.md** — `dataset.py` and `finetune.py` are productionizations of the spike script: same storage build, same provider wiring, same warm-start recipe, plus the CLI and the per-seed loop. Redirect the checkpoint save location to `--work-dir/cache_data` (FINDINGS.md documents how `NetworkInterface` resolves its base path — use whatever mechanism the spike proved: `load_path` arg or env; do not write into the shared venv cache). Epochs cap maps onto `learning_rates_dict={lr: epochs_cap}` + `early_end_epochs=10`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_chord_training_finetune.py -v` (expect 2 PASSED, minutes-slow) then the full suite.

- [ ] **Step 5: Commit**

```bash
git add backend/chord_training/dataset.py backend/chord_training/finetune.py backend/tests/test_chord_training_finetune.py
git commit -m "feat: Phase B dataset builder + per-seed warm-start finetune CLI"
```

---

### Task 4: Serving env override + rebaseline_v51.py

**Files:**
- Modify: `backend/chord_engine_chordia.py` (model-list resolution)
- Create: `backend/scripts/rebaseline_v51.py`
- Modify: `Makefile` (add `phase-v51` target + .PHONY)
- Test: `backend/tests/test_chordia_probs.py` (append one test)

**Interfaces:**
- Consumes: `recognize_chordia_probs` internals (Task 1 of v50 — it currently imports `MODEL_NAMES` from the package).
- Produces: env `CHORD_CHORDIA_MODELS` = comma-separated save_names (default: packaged `MODEL_NAMES`); env `CHORD_CHORDIA_MODEL_DIR` = extra dir whose `.sdict` files are used (default: shared cache only). `rebaseline_v51.py` CLI mirrors `rebaseline_v50.py` plus `--models` / `--model-dir` passthrough; diffs vs v50 baselines; same guard-track warning.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_chordia_probs.py`):

```python
def test_model_list_env_override(monkeypatch):
    import chord_engine_chordia as cec

    monkeypatch.setenv("CHORD_CHORDIA_MODELS", "name_a.best,name_b.best")
    assert cec._model_names() == ["name_a.best", "name_b.best"]
    monkeypatch.delenv("CHORD_CHORDIA_MODELS")
    from lv_chordia.chord_recognition import MODEL_NAMES
    assert cec._model_names() == list(MODEL_NAMES)
```

- [ ] **Step 2: Run to verify it fails** — `cd backend && .venv/bin/python -m pytest tests/test_chordia_probs.py::test_model_list_env_override -v` → FAIL (`_model_names` missing).

- [ ] **Step 3: Implement** in `chord_engine_chordia.py`:

```python
def _model_names() -> list[str]:
    """Serving checkpoint list; CHORD_CHORDIA_MODELS overrides (Phase B)."""
    raw = os.getenv("CHORD_CHORDIA_MODELS", "").strip()
    if raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    from lv_chordia.chord_recognition import MODEL_NAMES
    return list(MODEL_NAMES)
```

and in `recognize_chordia_probs` replace the `MODEL_NAMES` import/loop with `for model_name in _model_names():`. For `CHORD_CHORDIA_MODEL_DIR`: FINDINGS.md documents how the spike redirected `NetworkInterface`'s base path; apply the same mechanism here when the env var is set (if the spike used `load_path`, pass it through; if it used the shared-cache env, document that instead and drop the extra var — one mechanism, not two).

- [ ] **Step 4: Write `rebaseline_v51.py`** — copy `rebaseline_v50.py` structurally: v51 output/diff filenames, `V50` baselines as the comparison, same guard constants, plus:

```python
    ap.add_argument("--models", default="")
    ap.add_argument("--model-dir", default="")
    # in _run_eval, before the eval import:
    if args.models:
        os.environ["CHORD_CHORDIA_MODELS"] = args.models
    if args.model_dir:
        os.environ["CHORD_CHORDIA_MODEL_DIR"] = args.model_dir
```

(keep all seven forced env vars from v50). Makefile target `phase-v51` mirroring `phase-v50`.

- [ ] **Step 5: Run tests** — the appended test passes; full suite passes; `scripts/rebaseline_v51.py --help` shows the new flags.

- [ ] **Step 6: Commit**

```bash
git add backend/chord_engine_chordia.py backend/scripts/rebaseline_v51.py Makefile backend/tests/test_chordia_probs.py
git commit -m "feat: CHORD_CHORDIA_MODELS override + rebaseline_v51"
```

---

### Task 5: Bundle packer + desktop RUNBOOK

**Files:**
- Create: `backend/chord_training/pack_bundle.py`
- Create: `backend/chord_training/RUNBOOK.md`
- Create: `backend/chord_training/requirements-train.txt`

**Interfaces:**
- Consumes: manifest format (Task 2), `finetune.py` CLI (Task 3).
- Produces: `training_bundle.zip` layout: `data/` (audio+labs), `train_manifest.txt`, `val_manifest.txt` (paths rewritten RELATIVE to bundle root), `kit/` (dataset.py, finetune.py, FINDINGS.md), `checkpoints_in/` (the five packaged `.sdict` files copied from the venv shared cache), `RUNBOOK.md`, `requirements-train.txt`.

- [ ] **Step 1: Write `pack_bundle.py`** — CLI `--train-manifest F --val-manifest F --out training_bundle.zip`; copies audio/labs into `data/`, rewrites manifests to relative paths, includes kit files + packaged checkpoints + runbook; prints final size and a content summary. ~80 lines, stdlib `zipfile`/`shutil` only. `finetune.py` must accept relative manifest paths resolved against its CWD (verify; adjust if the spike hardcoded absolutes).

- [ ] **Step 2: Smoke test** (append to `backend/tests/test_chord_training_data.py`):

```python
def test_pack_bundle_layout(tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    (data / "a.wav").write_bytes(b"\x00")
    (data / "a.lab").write_text("0.0\t1.0\tC:maj\n")
    m = tmp_path / "train.txt"
    m.write_text(f"{data}/a.wav\t{data}/a.lab\n")
    v = tmp_path / "val.txt"
    v.write_text(f"{data}/a.wav\t{data}/a.lab\n")
    out = tmp_path / "bundle.zip"
    r = subprocess.run(
        [PY, str(BACKEND / "chord_training" / "pack_bundle.py"),
         "--train-manifest", str(m), "--val-manifest", str(v), "--out", str(out)],
        capture_output=True, text=True, cwd=BACKEND,
    )
    assert r.returncode == 0, r.stderr
    import zipfile
    names = zipfile.ZipFile(out).namelist()
    assert "RUNBOOK.md" in names and "kit/finetune.py" in names
    assert "train_manifest.txt" in names
    assert any(n.startswith("checkpoints_in/") and n.endswith(".sdict") for n in names)
```

Run (fails), implement, run (passes).

- [ ] **Step 3: Write `RUNBOOK.md`** — the desktop procedure, complete and copy-pasteable:
  1. Prereqs: Python 3.11, NVIDIA driver, then `pip install -r requirements-train.txt` in a venv (requirements-train.txt: `lv_chordia` pinned to the Mac's installed version — read it from `pip show lv_chordia`; `torch` with a `--index-url https://download.pytorch.org/whl/cu121` note — cu121 wheels support sm_75; `soundfile`, `librosa`, matching Mac versions).
  2. Unzip bundle; `cd bundle`.
  3. `python kit/finetune.py --train-manifest train_manifest.txt --val-manifest val_manifest.txt --work-dir out --seeds 0 --dry-run` — must print `DRY-RUN OK` before the long run.
  4. `python kit/finetune.py ... --seeds 0,1,2,3,4 --lr 1e-4 --epochs-cap 30 --batch-size 8` — note expected runtime (fill from spike timing × dataset scale), VRAM knob (`--batch-size 4` if OOM), and that seeds run sequentially.
  5. Copy back: `out/cache_data/*_ft1_s*.best.sdict` → Mac.
  Troubleshooting table: CUDA not found → CPU fallback flag and consequence; OOM → batch size; val loss never improves → runbook's val-swap knob (rebuild val manifest from held-out Isophonics — command included).

- [ ] **Step 4: Full suite + commit**

```bash
git add backend/chord_training/pack_bundle.py backend/chord_training/RUNBOOK.md backend/chord_training/requirements-train.txt backend/tests/test_chord_training_data.py
git commit -m "feat: Phase B training bundle packer + desktop runbook"
```

---

### Task 6: Data build → desktop handoff → v51 measurement (controller-run, pauses for Adeen)

No new code. Execution sequence with two human gates.

- [ ] **Step 1: Acquire training data (Mac, long-running)** — Isophonics non-gold: enumerate available `.lab` labs via `scripts/extract_isophonics_labs.py` inventory; fetch audio with the existing spotdl/downloader machinery into `backend/chord_training/data/train/`; duration identity gate (±2s vs lab end) applied — reject mismatches. Billboard val: download McGill Billboard annotations (public archive), convert via `billboard_to_lab.py`, pick ~40 tracks with clean conversions, fetch audio the same way into `data/val/`. Floors: ≥80 train pairs (spec), ≥25 val pairs — below floor, STOP and report coverage.
- [ ] **Step 2: Build manifests** — `build_manifest.py` on both dirs (leakage gate runs inside). Archive the leakage output next to the manifests.
- [ ] **Step 3: Pack** — `pack_bundle.py` → `training_bundle.zip`. Report to Adeen: bundle path, size, RUNBOOK pointer. **PAUSE — Adeen runs the desktop training and returns `*_ft1_s{0..4}.best.sdict`.**
- [ ] **Step 4: Deploy + DEV eval matrix** — place returned checkpoints in a repo-local dir (e.g. `backend/chord_training/checkpoints_ft1/`), then:
  - (a) `rebaseline_v51.py --split dev --models <five ft1 names> --model-dir <dir> --tag _ft1`
  - (b) if (a) is within 1pp of v50 DEV either way: mixed ensemble run `--models <ft1 s0,s1,s2 + packaged s3,s4>` `--tag _mixed` (one configuration, not a sweep).
  - Guard track checked per run.
- [ ] **Step 5: Decide** — winner = highest DEV ≥ 0.803 with guard OK. No candidate beats v50 DEV → STOP, report honestly (fine-tune failed; document in README; do not run TEST).
- [ ] **Step 6: TEST once** — winner flags, `--split both --tag ""` → `BASELINE_mir_gold_{DEV,TEST}_v51.json`. Shippable: TEST > 0.750; target ≥ 0.760.
- [ ] **Step 7: Document + commit** — `analysis/BASELINE_v51_README.md` (v49-README style): data (counts, sources, leakage-check output), hyperparameters, per-seed val curves if captured, DEV matrix, TEST result, per-track deltas for the usual four tracks, protocol statement (TEST consulted once). Update `ML_SETUP.md` (CHORD_CHORDIA_MODELS/MODEL_DIR usage). If shipped: repoint `eval_gold_mir.py` BASELINES to v51 (the v50 lesson — the ratchet must track the shipped baseline; workflow step-label rename still blocked on `gh auth refresh -s workflow`).

```bash
git add backend/analysis/BASELINE_*v51* backend/analysis/BASELINE_v51_README.md backend/ML_SETUP.md backend/eval_gold_mir.py
git commit -m "docs: v51 baselines — fine-tuned serving checkpoints"
```

---

## Self-Review Notes

- Spec coverage: Task 0 spike → Task 1; data section → Tasks 2 + 6.1; kit → Tasks 3 + 5; deployment/eval → Task 4 + 6.4-7; runbook flow → Task 5 + 6.3; acceptance criteria → Task 6.5-7. Billboard val-swap fallback → RUNBOOK troubleshooting (Task 5.3).
- Honesty note: Tasks 3's internals intentionally defer to `FINDINGS.md` (a committed artifact produced by Task 1) rather than inventing storage-API code this plan cannot verify; the spike is the plan's ground-truth generator, and its kill-switch stops the phase before fantasy code ships.
- Type consistency: manifest format (`audio\tlab`) identical in Tasks 2/3/5; checkpoint naming `…_ft1_s{i}.best` identical in Tasks 3/6; env names `CHORD_CHORDIA_MODELS`/`CHORD_CHORDIA_MODEL_DIR` identical in Tasks 4/6.
- Model note: Task 1 must NOT go to a cheap model — it is exploratory API archaeology with a kill-switch decision. Tasks 2, 4, 5 are sonnet-fine; Task 3 is sonnet reading FINDINGS.md.
