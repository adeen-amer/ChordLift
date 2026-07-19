# Beat/Downbeat Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 3 is exploratory API archaeology with a kill-switch decision — use a capable model, not a cheap one.**

**Goal:** Replace `beat_tracking.py`'s librosa heuristic with a model-based beat/downbeat dispatcher (madmom now, Beat-Transformer as a spiked enhancement), and surface bar boundaries in the chord sequence UI — mirroring chordmini's beat-detection setup.

**Architecture:** Value-first, research isolated. Tasks 1–2 ship a complete, independently useful upgrade (madmom dispatcher + bar-marker UI) with no research risk. Task 3 is a spike that either produces a verified Beat-Transformer integration contract or cleanly kill-switches (dispatcher already degrades gracefully to madmom either way). Tasks 4–5 implement and verify the transformer path only if Task 3 succeeds.

**Tech Stack:** Python 3.11 (`backend/.venv`), madmom 0.16.1, torch (already a dependency), existing `StemBundle`/`BeatGrid` dataclasses, React/TypeScript + vitest (frontend).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-beat-downbeat-detection-design.md`.
- Backend commands (Windows): `cd backend && .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/fixtures`. Single file: `.venv/Scripts/python.exe -m pytest tests/test_beat_engine.py -v`.
- Frontend commands: `cd frontend && npm test` (vitest) — this repo has **no React component-rendering test infra** (no `@testing-library/react`, no jsdom-based component tests); every existing frontend test is a pure-function unit test in `src/utils/*.test.ts`. Follow that pattern — do not add new test tooling.
- `BeatGrid` (`backend/beat_tracking.py`) and `bar_decode.py`'s consumption of it are the frozen contract — every engine returns the same dataclass shape; `bar_decode.py` itself is never modified by this plan.
- `beats.downbeat_times` already ships in the API response today (`presentation_timeline.py:181-186`, typed in `frontend/src/types.ts:28-33` as `BeatGridInfo`) — **no backend API/contract change is needed for the frontend bar-marker feature.** It is currently unused by any component; this plan wires it up. Accuracy of what's already shipped improves automatically once Tasks 1/4 land, with no frontend change required for that part.
- Never edit `backend/.venv/`; read-only.
- Task 3 (Beat-Transformer) is a genuine kill-switch: if a reliable, non-interactive way to obtain the pretrained checkpoint doesn't exist, or the model can't be run on CPU in this environment, STOP after Task 3, report honestly, and do not attempt Tasks 4–5. Tasks 1–2's shipped work stands on its own regardless.

---

### Task 1: madmom beat/downbeat engine + dispatcher

**Files:**
- Modify: `backend/requirements-ml.txt` (add `madmom==0.16.1`)
- Modify: `backend/beat_tracking.py` (add `_track_beats_madmom`, `track_beats_auto`, `BEAT_ENGINE` module constant)
- Modify: `backend/chord_pipeline.py:90` (call `track_beats_auto` instead of `track_beats`)
- Test: `backend/tests/test_beat_engine.py` (new)

**Interfaces:**
- Consumes: `StemBundle` (`backend/stem_separation.py`, fields `full`/`chord_signal`/`bass`/`method`), existing `BeatGrid`/`track_beats`/`BEATS_PER_BAR` (`beat_tracking.py`, untouched).
- Produces: `track_beats_auto(stems: StemBundle, sr: int, hop_length: int = 512, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid` — the new single entry point `chord_pipeline.py` calls. Module-level `BEAT_ENGINE` string (`"auto" | "madmom" | "librosa"`, from `CHORD_BEAT_ENGINE` env, default `"auto"`) — Task 4 later adds a `"transformer"` value and upgrades `"auto"`'s branch; both are read fresh from the module global at call time (same pattern as `bar_decode.BAR_DECODE_STRICT`), so tests monkeypatch it directly: `monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "madmom")`.

- [ ] **Step 1: Add madmom to requirements**

Append to `backend/requirements-ml.txt`:

```
madmom==0.16.1
```

madmom's C extensions need numpy present at build time; if `pip install -r requirements-ml.txt` fails on madmom specifically, the fix is `pip install numpy Cython && pip install --no-build-isolation madmom==0.16.1` — note this in a comment above the line:

```
# madmom needs numpy/Cython present *before* it builds its C extensions.
# If plain `pip install -r requirements-ml.txt` fails on madmom:
#   pip install numpy Cython && pip install --no-build-isolation madmom==0.16.1
madmom==0.16.1
```

- [ ] **Step 2: Write the failing tests**

```python
"""backend/tests/test_beat_engine.py"""
import numpy as np
import pytest

import beat_tracking
from beat_tracking import BeatGrid, track_beats_auto
from stem_separation import separate_stems


def _sine_chord(sr=22050, duration=6.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 110 * t)
    y += 0.25 * np.sin(2 * np.pi * 138 * t)
    y += 0.2 * np.sin(2 * np.pi * 165 * t)
    return y.astype(np.float32)


def test_librosa_engine_still_works(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "librosa")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 4


def test_madmom_engine_returns_valid_beat_grid(monkeypatch):
    pytest.importorskip("madmom")
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "madmom")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 1
    assert len(grid.downbeat_times) >= 1
    assert np.all(np.diff(grid.beat_times) >= 0)


def test_auto_falls_back_to_librosa_on_madmom_failure(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("madmom unavailable")

    monkeypatch.setattr(beat_tracking, "_track_beats_madmom", _boom)
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_beat_engine.py -v`
Expected: FAIL (`track_beats_auto` not defined)

- [ ] **Step 4: Implement in `beat_tracking.py`**

Add near the top (after existing imports) and at the end of the file:

```python
import logging
import os

logger = logging.getLogger(__name__)

BEAT_ENGINE = os.getenv("CHORD_BEAT_ENGINE", "auto").lower().strip()


def _track_beats_madmom(y_full: np.ndarray, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid:
    """Model-based beat/downbeat tracking on the full mix (no stems needed)."""
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    signal = Signal(y_full.astype(np.float32), sample_rate=sr, num_channels=1)
    activations = RNNDownBeatProcessor()(signal)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar], fps=100)
    beat_positions = proc(activations)  # rows: [time_sec, beat_number]

    if len(beat_positions) == 0:
        raise RuntimeError("madmom returned no beats")

    beat_times = np.asarray(beat_positions[:, 0], dtype=np.float64)
    downbeat_mask = beat_positions[:, 1].astype(int) == 1
    downbeat_times = beat_times[downbeat_mask]
    if len(downbeat_times) == 0:
        downbeat_times = beat_times[::beats_per_bar]

    tempo_bpm = 60.0 / float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 120.0

    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=np.asarray(downbeat_times, dtype=np.float64),
        tempo_bpm=tempo_bpm,
        beats_per_bar=beats_per_bar,
    )


def track_beats_auto(
    stems,
    sr: int,
    hop_length: int = 512,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BeatGrid:
    """
    Dispatch to the configured beat/downbeat engine (CHORD_BEAT_ENGINE).

    "auto" (default) tries madmom, falling back to the librosa heuristic on
    any failure. "madmom" forces madmom (errors surface). "librosa" keeps
    today's heuristic. Task 4 adds a "transformer" engine and upgrades
    "auto" to try it first when real Demucs stems are available.
    """
    engine = BEAT_ENGINE

    if engine == "madmom":
        return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)

    if engine == "auto":
        try:
            return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)
        except Exception:
            logger.warning("madmom beat tracking failed, falling back to librosa heuristic", exc_info=True)

    return track_beats(
        stems.chord_signal, stems.bass, sr, hop_length=hop_length, beats_per_bar=beats_per_bar,
    )
```

- [ ] **Step 5: Wire into the pipeline**

In `backend/chord_pipeline.py:83-91`, change:

```python
def build_chord_pipeline_context(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> ChordPipelineContext:
    """Separate stems and track beats/downbeats before chord decoding."""
    stems = separate_stems(y, sr)
    beats = track_beats(stems.chord_signal, stems.bass, sr, hop_length=hop_length)
    return ChordPipelineContext(stems=stems, beats=beats, sr=sr, hop_length=hop_length)
```

to:

```python
def build_chord_pipeline_context(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> ChordPipelineContext:
    """Separate stems and track beats/downbeats before chord decoding."""
    stems = separate_stems(y, sr)
    beats = track_beats_auto(stems, sr, hop_length=hop_length)
    return ChordPipelineContext(stems=stems, beats=beats, sr=sr, hop_length=hop_length)
```

and update the import at the top of the file:

```python
from beat_tracking import BeatGrid, track_beats_auto
```

(`track_beats` is no longer imported here — `beat_tracking.py`'s own module code still uses it internally, and `tests/test_chord_pipeline.py`/`tests/test_ml_beat_snap.py`/`scripts/diagnose_phase8_dev.py` import it directly from `beat_tracking`, so it stays exported and those files need no changes.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_beat_engine.py -v`
Expected: 3 PASSED (madmom test skips if not installed).

Then the full suite: `.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/fixtures` — all pass, including the untouched `test_chord_pipeline.py` (it calls `track_beats` directly, still exported unchanged) and `test_ml_beat_snap.py`.

- [ ] **Step 7: Commit**

```bash
git add backend/requirements-ml.txt backend/beat_tracking.py backend/chord_pipeline.py backend/tests/test_beat_engine.py
git commit -m "feat: madmom beat/downbeat engine + CHORD_BEAT_ENGINE dispatcher"
```

---

### Task 2: Frontend bar markers in the chord sequence

**Files:**
- Modify: `frontend/src/utils/timeline.ts` (add `barIndexForTime`, `barNumbersForTimeline`)
- Modify: `frontend/src/utils/timeline.test.ts` (append tests)
- Modify: `frontend/src/components/ChordSequence.tsx` (accept `downbeatTimes` prop, render dividers)
- Modify: `frontend/src/pages/HomePage.tsx:407-412` (pass `data.beats?.downbeat_times`)
- Modify: `frontend/src/index.css` (divider styling)

**Interfaces:**
- Consumes: `ChordEvent[]` timeline (existing, `types.ts:9-26`), `BeatGridInfo.downbeat_times: number[]` (existing, `types.ts:28-33`, already in the API response — no backend change).
- Produces: `barIndexForTime(downbeatTimes: number[], t: number): number` and `barNumbersForTimeline(timeline: {time: number}[], downbeatTimes: number[]): number[]` (exported from `timeline.ts`) — `ChordSequence.tsx` uses the latter to know where to render a divider.

- [ ] **Step 1: Write the failing tests** (append to `frontend/src/utils/timeline.test.ts`):

```typescript
import { barIndexForTime, barNumbersForTimeline } from './timeline';

describe('barIndexForTime', () => {
  const downbeats = [0, 2, 4, 6];

  it('returns 0 before the first downbeat', () => {
    expect(barIndexForTime(downbeats, -1)).toBe(0);
  });

  it('finds the containing bar', () => {
    expect(barIndexForTime(downbeats, 0)).toBe(0);
    expect(barIndexForTime(downbeats, 1.9)).toBe(0);
    expect(barIndexForTime(downbeats, 2)).toBe(1);
    expect(barIndexForTime(downbeats, 5.5)).toBe(2);
  });

  it('clamps to the last bar past the final downbeat', () => {
    expect(barIndexForTime(downbeats, 100)).toBe(3);
  });

  it('returns 0 for an empty downbeat list', () => {
    expect(barIndexForTime([], 5)).toBe(0);
  });
});

describe('barNumbersForTimeline', () => {
  it('maps each chord to its bar index', () => {
    const downbeats = [0, 2, 4];
    const timeline = [{ time: 0 }, { time: 1 }, { time: 2.1 }, { time: 5 }];
    expect(barNumbersForTimeline(timeline, downbeats)).toEqual([0, 0, 1, 2]);
  });

  it('returns all zeros with no downbeat data', () => {
    const timeline = [{ time: 0 }, { time: 3 }];
    expect(barNumbersForTimeline(timeline, [])).toEqual([0, 0]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- timeline.test.ts`
Expected: FAIL (`barIndexForTime` not exported)

- [ ] **Step 3: Implement in `timeline.ts`** (append):

```typescript
/** Index of the bar containing `t`, given sorted downbeat times. Same binary search shape as activeSegmentIndex. */
export function barIndexForTime(downbeatTimes: number[], t: number): number {
  if (!downbeatTimes.length) return 0;

  let lo = 0;
  let hi = downbeatTimes.length - 1;
  let candidate = 0;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (downbeatTimes[mid] <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return candidate;
}

/** Bar index for every chord segment's start time, in timeline order. */
export function barNumbersForTimeline(
  timeline: { time: number }[],
  downbeatTimes: number[],
): number[] {
  return timeline.map((seg) => barIndexForTime(downbeatTimes, seg.time));
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- timeline.test.ts`
Expected: all PASSED.

- [ ] **Step 5: Render dividers in `ChordSequence.tsx`**

Add the prop and compute bar numbers once per render (small list, no memo needed — matches the file's existing style of plain per-render `.map`):

```typescript
import { forwardRef, useState } from 'react';
import type { ChordEvent } from '../types';
import { ChordDiagram } from './ChordDiagram';
import { segmentTimeKey, timelineListKey } from '../utils/chordCorrections';
import { barNumbersForTimeline } from '../utils/timeline';

interface ChordSequenceProps {
  timeline: ChordEvent[];
  downbeatTimes?: number[];
  onCorrectChord?: (time: number, chord: string) => void;
}

export const ChordSequence = forwardRef<HTMLDivElement, ChordSequenceProps>(
  ({ timeline, downbeatTimes, onCorrectChord }, ref) => {
    const [editingKey, setEditingKey] = useState<string | null>(null);
    const [draft, setDraft] = useState('');
    const barNumbers = barNumbersForTimeline(timeline, downbeatTimes ?? []);
```

(keep the rest of the existing function body as-is). Inside the `timeline.map((chord, idx) => { ... })` block, right before the `return (` that renders the chord card, insert a divider whenever the bar number changes from the previous card:

```typescript
          {timeline.map((chord, idx) => {
            const segKey = segmentTimeKey(chord.time);
            const editKey = String(idx);
            const tier = chord.user_corrected
              ? 'high'
              : (chord.confidence_tier ?? (chord.is_low_confidence ? 'low' : 'high'));
            const tierClass = tier !== 'high' ? ` confidence-${tier}` : '';
            const adjustedClass = chord.display_adjusted ? ' display-adjusted' : '';
            const correctedClass = chord.user_corrected ? ' user-corrected' : '';
            const showBarDivider =
              downbeatTimes && downbeatTimes.length > 0 && (idx === 0 || barNumbers[idx] !== barNumbers[idx - 1]);

            return (
              <>
                {showBarDivider && (
                  <div className="bar-divider" key={`bar-${timelineListKey(idx)}`}>
                    <span className="bar-number">{barNumbers[idx] + 1}</span>
                  </div>
                )}
                <div
                  key={timelineListKey(idx)}
                  className={`chord-card${tierClass}${adjustedClass}${correctedClass}`}
                  id={`chord-${timelineListKey(idx)}`}
                  data-seg-index={idx}
                  data-time={segKey}
                  title={
                    chord.model_chord && chord.model_chord !== chord.chord
                      ? `Model: ${chord.model_chord}`
                      : undefined
                  }
                >
```

and close the added `</>` fragment where the original card `</div>` closes, right before the `);` that ends the `.map()` callback (the fragment must wrap exactly the divider + the existing card `<div>...</div>`).

- [ ] **Step 6: Pass the prop from `HomePage.tsx`**

At `frontend/src/pages/HomePage.tsx:407-412`, change:

```typescript
              <ChordSequence
                key={`${data.video_id}-${effectivePresentationMode}`}
                timeline={displayTimeline}
                ref={chordContainerRef}
                onCorrectChord={handleCorrectChord}
              />
```

to:

```typescript
              <ChordSequence
                key={`${data.video_id}-${effectivePresentationMode}`}
                timeline={displayTimeline}
                downbeatTimes={data.beats?.downbeat_times}
                ref={chordContainerRef}
                onCorrectChord={handleCorrectChord}
              />
```

- [ ] **Step 7: Add divider CSS** to `frontend/src/index.css`, near the existing `.chord-container`/`.chord-card` rules (around line 390):

```css
.bar-divider {
  flex: 0 0 auto;
  align-self: stretch;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  width: 1px;
  background: rgba(255, 255, 255, 0.12);
  position: relative;
}
.bar-divider .bar-number {
  position: absolute;
  top: -14px;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.4);
  white-space: nowrap;
}
```

- [ ] **Step 8: Typecheck + run full frontend test suite**

Run: `cd frontend && npx tsc --noEmit && npm test`
Expected: no type errors, all tests pass.

- [ ] **Step 9: Manual verification in the browser**

Run the dev server (`npm run dev` + backend `uvicorn main:app --reload`), analyze a track, confirm bar dividers with numbers appear in the chord sequence at roughly every 4th chord for a steady 4/4 track, and that playback highlighting still works (dividers must not break the `data-seg-index`-based rAF highlighting loop, since divider elements carry no `data-seg-index`).

- [ ] **Step 10: Commit**

```bash
git add frontend/src/utils/timeline.ts frontend/src/utils/timeline.test.ts frontend/src/components/ChordSequence.tsx frontend/src/pages/HomePage.tsx frontend/src/index.css
git commit -m "feat: bar dividers in chord sequence from existing downbeat data"
```

---

### Task 3: Beat-Transformer contract spike — CAPABLE MODEL, NOT SONNET

**Files:**
- Create: `backend/beat_transformer/__init__.py`
- Create: `backend/beat_transformer/FINDINGS.md`
- Create: `backend/beat_transformer/spike_infer.py`

**Interfaces:**
- Consumes: upstream `zhaojw1998/Beat-Transformer` (MIT license) — read `code/DilatedTransformer.py` and `code/DilatedTransformerLayer.py` (model definition, forward signature, expected input channel count/output heads), `code/spectrogram_dataset.py` (the exact spectrogram type/params the model was trained on — this is the ground-truth preprocessing contract), `preprocessing/demixing.py` (how the 4 Spleeter stems + mix become the model's input channels — the order/normalization to replicate with Demucs stems from `backend/stem_separation.py`'s `StemBundle`), `code/eight_fold_test.py` (canonical checkpoint-load → forward → beat/downbeat post-processing), and the README's checkpoint/`checkpoint` folder section (exact acquisition mechanism). `backend/stem_separation.py`'s `StemBundle` (bass/drums/vocals/other/full).
- Produces: `FINDINGS.md` — the verified contract Task 4 implements from, or a documented kill-switch. Committed pinned commit SHA of the upstream repo read.

- [ ] **Step 1: Read the upstream source** — fetch and read, in order: `code/DilatedTransformer.py`, `code/DilatedTransformerLayer.py`, `code/spectrogram_dataset.py`, `preprocessing/demixing.py`, `code/eight_fold_test.py`, and the README's checkpoint section, at the current default-branch commit. Record the exact commit SHA (via `gh api repos/zhaojw1998/Beat-Transformer/commits/main` or the GitHub UI's commit hash for the files read). Goal: answer, with certainty from source (not the README's prose summary) — model constructor args, `forward()` input tensor shape (channels/freq-bins/time), spectrogram library+params (sample rate, hop length, mel bins or STFT, log scaling), stem channel order, output tensor shape and how `eight_fold_test.py` converts it into beat/downbeat times + a frame rate.

- [ ] **Step 2: Resolve checkpoint acquisition** — determine the actual, current mechanism to obtain pretrained weights (a direct URL, a `gdown`-able Google Drive ID, a Hugging Face mirror, or only-via-Colab). Attempt it. If a non-interactive download succeeds, save the file under `backend/models/beat_transformer.pth` (already gitignored per `backend/.gitignore`/`.gitignore` `*.pth` rules) and record its SHA-256 and byte size in `FINDINGS.md`.
  - **If no reliable non-interactive acquisition method exists** (Colab-only, expired/private Drive link, requires manual browser auth): STOP here. Write `FINDINGS.md` documenting exactly what was tried and why it doesn't work headlessly. Do not proceed to Step 3. Report this outcome plainly — Tasks 1–2's madmom+UI work already ships independently; `CHORD_BEAT_ENGINE` simply never gains a working `"transformer"` value.

- [ ] **Step 3: Write `spike_infer.py`** (only if Step 2 succeeded) — prove, on CPU, with the same synthetic sine-chord clip pattern as Task 1's test fixture:
  1. Build a `StemBundle` via `separate_stems()` on a ~6s synthetic clip (reuse `_sine_chord()` from `tests/test_beat_engine.py` — import it or inline an equivalent).
  2. Construct the traced spectrogram input from the `StemBundle`'s stems per Step 1's findings, substituting Demucs stem order/semantics for the reference Spleeter ones (document any assumption this requires — e.g. if Spleeter's 4-stem order differs from Demucs's `StemBundle` field order, the mapping must be explicit here, not assumed silently).
  3. Load `backend/models/beat_transformer.pth` into a freshly constructed model per `DilatedTransformer.py`'s signature; run `model.eval()` + a no-grad forward pass on CPU.
  4. Assert the output tensor shape matches what Step 1 documented, and convert it into `(beat_times, downbeat_times)` arrays per `eight_fold_test.py`'s post-processing logic.
  End with `print("SPIKE OK")` only if every assertion passed.

- [ ] **Step 4: Run it** — `cd backend && .venv/Scripts/python.exe beat_transformer/spike_infer.py`. Expected: `SPIKE OK`. Iterate on real failures (shape/dtype/channel-order mismatches are the expected fight — this is unfamiliar vendored-model integration, same character as the lv_chordia spike in `chord_training/spike_warmstart.py`). If genuinely blocked after honest effort (e.g. the model requires an op unavailable on CPU, or the traced spectrogram recipe can't be reproduced without the original Spleeter output), STOP and report BLOCKED with the specific wall — do not fake success.

- [ ] **Step 5: Write `FINDINGS.md`** — the verified recipe as prose + the exact code fragments that ran: spectrogram construction (library, params, per-channel meaning and order), model construction args, checkpoint load, forward call, output-to-`BeatGrid` conversion (frame rate, which output channel/argmax means "downbeat"). Include every quirk hit and its fix. This is what Task 4 implements from — it must not need to re-read the upstream repo.

- [ ] **Step 6: Commit**

```bash
git add backend/beat_transformer/
git commit -m "feat: Beat-Transformer contract spike (FINDINGS.md)"
```

(If Step 2 kill-switched: same commit command, `FINDINGS.md` documents the blocker instead of a working recipe. Either way, commit — the research is valuable even if the answer is "no".)

---

### Task 4: `beat_transformer` model + inference (from FINDINGS.md)

**Skip this task entirely if Task 3 kill-switched.** Everything below assumes `backend/beat_transformer/FINDINGS.md` documents a verified, working recipe and `backend/models/beat_transformer.pth` exists.

**Files:**
- Create: `backend/beat_transformer/model.py` (vendored model code)
- Create: `backend/beat_transformer/infer.py`
- Modify: `backend/model_cache.py` (add `get_beat_transformer_model()`)
- Modify: `backend/beat_tracking.py` (`_track_beats_transformer`, add `"transformer"` engine, upgrade `"auto"`)
- Test: `backend/tests/test_beat_engine.py` (append)

**Interfaces:**
- Consumes: `backend/beat_transformer/FINDINGS.md` (Task 3's committed contract — implement from it, do not re-derive), `StemBundle` (`stem_separation.py`).
- Produces: `beat_transformer.infer.run(stems: StemBundle, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid`. `model_cache.get_beat_transformer_model()` — lazy singleton, same pattern as `get_demucs_model()`. `beat_tracking._track_beats_transformer(stems, sr, beats_per_bar)` wraps `infer.run`.

- [ ] **Step 1: Vendor the model** in `backend/beat_transformer/model.py` — port `DilatedTransformer.py`/`DilatedTransformerLayer.py` per `FINDINGS.md`'s exact constructor signature, trimmed to inference-only (drop training-only code paths: loss functions, optimizer setup, data augmentation). Docstring at the top of the file:

```python
"""
Vendored from zhaojw1998/Beat-Transformer (MIT license), commit <SHA from FINDINGS.md>.
Inference-only: training code, loss functions, and augmentation stripped.
See backend/beat_transformer/FINDINGS.md for the full integration contract.
"""
```

- [ ] **Step 2: Write `infer.py`** implementing the spectrogram construction + forward pass + output conversion exactly as proven in `spike_infer.py`, but as a reusable function:

```python
"""Beat-Transformer inference: StemBundle -> BeatGrid. See FINDINGS.md for the contract."""
from __future__ import annotations

import numpy as np

from beat_tracking import BeatGrid, BEATS_PER_BAR


def run(stems, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid:
    """Run Beat-Transformer on a StemBundle's real (Demucs) stems. Raises on any failure — callers handle fallback."""
    from model_cache import get_beat_transformer_model

    model = get_beat_transformer_model()
    # [Spectrogram construction, forward pass, output->BeatGrid conversion:
    #  copy verbatim from spike_infer.py's proven Step 3 logic, per FINDINGS.md.]
    ...
```

(The implementer copies the exact proven logic from `spike_infer.py` here — `FINDINGS.md` and the spike script are the source of truth for this body; this plan does not re-specify it because Task 3 is what verifies it, per this plan's Global Constraints.)

- [ ] **Step 3: Add the singleton loader** to `backend/model_cache.py`:

```python
_beat_transformer_model: Any = None


def get_beat_transformer_model():
    global _beat_transformer_model
    if _beat_transformer_model is not None:
        return _beat_transformer_model
    with _lock:
        if _beat_transformer_model is None:
            import torch

            from beat_transformer.model import DilatedTransformerModel  # exact class name per FINDINGS.md

            logger.info("Loading Beat-Transformer model (cached for process lifetime)")
            model = DilatedTransformerModel()  # exact constructor args per FINDINGS.md
            state = torch.load("models/beat_transformer.pth", map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            _beat_transformer_model = model
        return _beat_transformer_model
```

(class name and constructor args here are placeholders for the implementer to fill from `FINDINGS.md` — this is the one spot in this plan where the exact upstream API name isn't yet known to the plan author; everything else is concrete.)

- [ ] **Step 4: Write the failing test** (append to `backend/tests/test_beat_engine.py`):

```python
from pathlib import Path

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "beat_transformer.pth"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Beat-Transformer checkpoint not present")
def test_transformer_engine_returns_valid_beat_grid(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "transformer")
    stems = separate_stems(_sine_chord(), 22050)
    grid = track_beats_auto(stems, 22050)
    assert isinstance(grid, BeatGrid)
    assert len(grid.beat_times) >= 1
    assert np.all(np.diff(grid.beat_times) >= 0)


def test_auto_prefers_transformer_for_demucs_stems(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")
    calls = []
    monkeypatch.setattr(beat_tracking, "_track_beats_transformer", lambda *a, **k: calls.append("transformer") or BeatGrid(np.array([0.0, 0.5]), np.array([0.0]), 120.0))
    monkeypatch.setattr(beat_tracking, "_track_beats_madmom", lambda *a, **k: calls.append("madmom") or BeatGrid(np.array([0.0, 0.5]), np.array([0.0]), 120.0))
    stems = separate_stems(_sine_chord(), 22050)
    stems.method  # existing field; force demucs-shaped call for this test
    object.__setattr__(stems, "method", "demucs")
    track_beats_auto(stems, 22050)
    assert calls == ["transformer"]
```

- [ ] **Step 5: Run to verify it fails** — `.venv/Scripts/python.exe -m pytest tests/test_beat_engine.py -v` → FAIL (`_track_beats_transformer` undefined).

- [ ] **Step 6: Implement in `beat_tracking.py`** — add:

```python
def _track_beats_transformer(stems, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid:
    from beat_transformer.infer import run as run_beat_transformer

    return run_beat_transformer(stems, sr, beats_per_bar=beats_per_bar)
```

and update `track_beats_auto`'s body:

```python
def track_beats_auto(
    stems,
    sr: int,
    hop_length: int = 512,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BeatGrid:
    engine = BEAT_ENGINE

    if engine == "transformer":
        return _track_beats_transformer(stems, sr, beats_per_bar=beats_per_bar)

    if engine == "madmom":
        return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)

    if engine == "auto":
        if stems.method == "demucs":
            try:
                return _track_beats_transformer(stems, sr, beats_per_bar=beats_per_bar)
            except Exception:
                logger.warning("Beat-Transformer failed, falling back to madmom", exc_info=True)
        try:
            return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)
        except Exception:
            logger.warning("madmom beat tracking failed, falling back to librosa heuristic", exc_info=True)

    return track_beats(
        stems.chord_signal, stems.bass, sr, hop_length=hop_length, beats_per_bar=beats_per_bar,
    )
```

- [ ] **Step 7: Run tests to verify they pass** — `.venv/Scripts/python.exe -m pytest tests/test_beat_engine.py -v` (transformer test skips if checkpoint absent elsewhere, e.g. CI); then full suite.

- [ ] **Step 8: Commit**

```bash
git add backend/beat_transformer/model.py backend/beat_transformer/infer.py backend/model_cache.py backend/beat_tracking.py backend/tests/test_beat_engine.py
git commit -m "feat: Beat-Transformer engine, auto-preferred for real Demucs stems"
```

---

### Task 5: Deterministic click-track sanity check

**Skip if Task 3 kill-switched — this task still applies to the madmom engine either way, adjust scope to whichever engines exist.**

**Files:**
- Test: `backend/tests/test_beat_engine.py` (append)

**Interfaces:**
- Consumes: `track_beats_auto`, engines built in Tasks 1/4.
- Produces: no new production code — a regression check with known ground truth, no network/audio-download dependency (per this repo's existing pattern of flaky yt-dlp downloads — a beat-detection sanity check must not depend on that).

- [ ] **Step 1: Write a deterministic click-track fixture + test**

```python
def _click_track(sr=22050, bpm=120.0, bars=8, beats_per_bar=4):
    """Synthetic click track: sharp clicks on every beat, louder on downbeats. Exact ground truth."""
    beat_dur = 60.0 / bpm
    total_beats = bars * beats_per_bar
    duration = total_beats * beat_dur + 1.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    click = np.exp(-np.linspace(0, 30, int(sr * 0.03))).astype(np.float32)  # short decaying click
    beat_times_gt = []
    downbeat_times_gt = []
    for i in range(total_beats):
        t = i * beat_dur
        beat_times_gt.append(t)
        if i % beats_per_bar == 0:
            downbeat_times_gt.append(t)
        start = int(t * sr)
        amp = 1.0 if i % beats_per_bar == 0 else 0.6
        end = min(start + len(click), len(y))
        y[start:end] += amp * click[: end - start]
    return y, np.array(beat_times_gt), np.array(downbeat_times_gt)


def test_madmom_downbeat_accuracy_on_click_track(monkeypatch):
    pytest.importorskip("madmom")
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "madmom")
    y, beat_gt, downbeat_gt = _click_track()
    stems = separate_stems(y, 22050)
    grid = track_beats_auto(stems, 22050)

    # every ground-truth downbeat should have a detected downbeat within 60ms
    tolerance = 0.06
    hits = 0
    for gt in downbeat_gt:
        if len(grid.downbeat_times) and np.min(np.abs(grid.downbeat_times - gt)) < tolerance:
            hits += 1
    assert hits / len(downbeat_gt) >= 0.8, (
        f"only {hits}/{len(downbeat_gt)} ground-truth downbeats matched within {tolerance}s"
    )
```

- [ ] **Step 2: Run it** — `.venv/Scripts/python.exe -m pytest tests/test_beat_engine.py::test_madmom_downbeat_accuracy_on_click_track -v`. Expected: PASS. If it fails, this is a real signal — either the madmom wiring in Task 1 is wrong (check `_track_beats_madmom`'s `beats_per_bar`/fps handling) or the tolerance/threshold needs adjustment for a synthetic click track's known easiness (a real DBN tracker should handle a clean click track near-perfectly; do not loosen the threshold to make it pass without understanding why it failed).

- [ ] **Step 3: Full suite + commit**

```bash
git add backend/tests/test_beat_engine.py
git commit -m "test: click-track downbeat accuracy sanity check"
```

---

## Self-Review Notes

- Spec coverage: engine dispatcher + madmom → Task 1; frontend bar markers → Task 2 (discovered during planning that `downbeat_times` already ships in the API — spec's proposed new `bar_number` per-segment field is unnecessary; the frontend derives it locally, a smaller diff achieving the same acceptance criterion); Beat-Transformer contract verification → Task 3; Beat-Transformer implementation + `auto` engine selection by `StemBundle.method` → Task 4; accuracy sanity check → Task 5. `CHORD_BEAT_ENGINE` env var (spec's engine-selection table) → Tasks 1 + 4 jointly implement all four values.
- Honesty note: Task 4's `infer.py` body and `model_cache.py`'s exact class name are intentionally left as "copy from FINDINGS.md/spike_infer.py" rather than invented — this plan cannot know Beat-Transformer's exact API surface before Task 3 reads the source. This mirrors `docs/superpowers/plans/2026-07-10-phase-b-finetune.md`'s Task 3, which deferred to a committed `FINDINGS.md` the same way.
- Type consistency: `BeatGrid` fields (`beat_times`, `downbeat_times`, `tempo_bpm`, `beats_per_bar`) used identically across Tasks 1/4/5. `BEAT_ENGINE` values (`"auto"`/`"madmom"`/`"librosa"`/`"transformer"`) consistent across Tasks 1 and 4. `barIndexForTime`/`barNumbersForTimeline` names consistent between Task 2's test and implementation.
- Task 3/4 split exists specifically so a kill-switch on the checkpoint-acquisition step (a real possibility per the design doc's research) doesn't leave the repo in a half-implemented state — Tasks 1-2 are already a complete, shippable unit before Task 3 even starts.
