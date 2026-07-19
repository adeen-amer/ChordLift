# Beat/downbeat detection (chordmini parity) — design

**Date:** 2026-07-20
**Status:** approved
**Owner:** Adeen

## Context

ChordLift's only beat/downbeat estimation today is [beat_tracking.py](../../../backend/beat_tracking.py):
`librosa.beat.beat_track` (classic onset-strength tempo tracker) for beat times, plus a
heuristic for downbeats — pick the highest-bass-onset beat in the first 32 beats as the
"bar anchor," then mark every 4th beat from there as a downbeat, assuming 4/4 throughout.
This feeds `BeatGrid` (`beat_times`, `downbeat_times`, `tempo_bpm`), consumed only
internally by [bar_decode.py](../../../backend/bar_decode.py) to snap/collapse chord
segments to bar boundaries. There is no beat/downbeat UI — this machinery is invisible to
the user.

chordmini (the reference app, https://github.com/ptnghia-j/ChordMiniApp) uses a real
model for this: Beat-Transformer primary, madmom fallback. Beat-Transformer
(https://github.com/zhaojw1998/Beat-Transformer, MIT license, ISMIR 2022) takes
demixed spectrograms (mix + 4 stems) as input; chordmini demixes with Spleeter.
ChordLift already has 4-stem separation via Demucs in
[stem_separation.py](../../../backend/stem_separation.py) (`CHORD_STEM_PREFER_DEMUCS`,
off by default — HPSS pseudo-stems are the default fast path), so Beat-Transformer can
reuse that instead of adding Spleeter as a new dependency.

Beat-Transformer is **not pip-installable** — its checkpoint is bundled in-repo but the
model/inference code must be vendored, and its exact spectrogram/channel-stacking
contract must be traced from source (same style of reverse-engineering already done for
lv_chordia in
[chord_training/FINDINGS.md](../../../backend/chord_training/FINDINGS.md)), since it
expects Spleeter-shaped input and ChordLift will feed it Demucs-shaped input instead —
these should be equivalent (both are bass/drums/vocals/other stems) but the wiring is
not a config swap.

Production runs CPU-only (`backend/Dockerfile`, `python:3.11-slim`, no CUDA base;
deployed to Hugging Face — see project memory). Beat-Transformer inference plus real
Demucs stem separation adds real latency on CPU.

## Goal

Replace the librosa heuristic with a model-based beat/downbeat tracker (Beat-Transformer
primary, madmom fallback — mirroring chordmini's setup), and surface bar boundaries in
the UI as dividers/bar numbers in the chord sequence. `bar_decode.py`'s consumers must
see no contract change — only `BeatGrid`'s accuracy improves.

## Non-goals

- No Spleeter dependency — stems come from the existing `separate_stems()`.
- No new chord-model work (BTC, multi-model selection) — tracked separately.
- No metronome pulse or BPM readout UI — bar markers only, per this round's scope.
- No always-on Demucs by default — stem mode policy (`CHORD_STEM_MODE`/
  `CHORD_STEM_PREFER_DEMUCS`) is unchanged by this work; beat engine selection adapts to
  whatever stem mode is already active.

## Design

### Backend: `beat_transformer.py` (new)

- Vendors the Beat-Transformer model definition + inference code (MIT license permits
  this) and its bundled pretrained checkpoint, loaded lazily via a `model_cache.py`-style
  singleton (matches the existing `get_demucs_model()` pattern).
- Input: the `StemBundle` already produced by `separate_stems()` (bass/drums/vocals/other
  + full mix). First integration task: trace the reference repo's Spleeter-based
  preprocessing (spectrogram type, channel order, normalization) and confirm the Demucs
  stems can be substituted directly — document this the way `FINDINGS.md` documents the
  lv_chordia contract, since assumptions here are unverified until read against source.
- Output: converts the model's beat/downbeat frame predictions into the existing
  `BeatGrid` dataclass — same fields, same units. No downstream code changes.

### Backend: madmom fallback

- `madmom.features.downbeats.DBNDownBeatTrackingProcessor` (new dependency) operates
  directly on the full mix — no stems required.
- Used when: `separate_stems()` returned HPSS pseudo-stems (not real Demucs — Beat-
  Transformer's accuracy assumption doesn't hold on pseudo-stems), or when
  Beat-Transformer inference raises.
- Also produces `BeatGrid` — same contract.

### Backend: engine selection

New env var `CHORD_BEAT_ENGINE` (mirrors `CHORD_STEM_MODE`'s convention):

| value | behavior |
|---|---|
| `auto` (default) | Beat-Transformer when the `StemBundle.method` returned by `separate_stems()` is `"demucs"`; madmom when it's `"hpss"` |
| `transformer` | force Beat-Transformer (errors surface, no silent fallback — for eval/debugging) |
| `madmom` | force madmom |
| `librosa` | today's heuristic (escape hatch / CI-fast path, keeps `beat_tracking.py`'s current function intact) |

`track_beats()` in `beat_tracking.py` becomes the dispatcher across these four; its
existing librosa implementation is kept as the `librosa` branch, unchanged.

### Data flow

`analyzer.py` pipeline: after `separate_stems()`, call the dispatcher → `BeatGrid` →
`bar_decode.py` (unchanged) → `presentation_timeline.py` adds a `bar_number` field to
each chord segment in the API response (new, small).

### Frontend: `ChordSequence.tsx`

Render a divider + bar-number label wherever `bar_number` changes between consecutive
chord cards. No new component; CSS addition for the divider/label styling.

### Testing

- Contract test: madmom and Beat-Transformer paths both return a `BeatGrid` with valid
  shape/monotonic times (mirrors how `chord_engine_chordia.py`'s env-override test
  checks contract, not accuracy).
- One fixture-track sanity check: compare downbeat estimate against a known reference
  track (reuse the existing gold-audio fixture pattern from `eval_gold_mir.py` /
  `gold_audio_bundle.py`) — smallest check that fails if the wiring breaks, per the
  self-check convention already used elsewhere in this backend.
- `dataset.py`/`FINDINGS.md`-style contract doc for the Beat-Transformer integration
  (dtype, input shape, stem order) before wiring it into `analyzer.py`.

## Acceptance criteria

- `CHORD_BEAT_ENGINE=auto` selects Beat-Transformer when Demucs stems are active, madmom
  otherwise; both produce a valid `BeatGrid`.
- `bar_decode.py` and all its existing tests pass unmodified (contract preserved).
- Chord sequence UI shows bar dividers/numbers driven by real `bar_number` data, not a
  placeholder.
- No new hard dependency on Spleeter.
- Beat-Transformer input contract (Demucs stems substituting for Spleeter) is documented
  and verified, not assumed.

## Risks

| Risk | Mitigation |
|---|---|
| Beat-Transformer's Spleeter-trained weights don't transfer cleanly to Demucs stems (different separation artifacts) | Verify via the fixture sanity check before shipping as default; `auto` still falls back to madmom if accuracy looks off in eval, without a code change (env var) |
| CPU latency (Beat-Transformer + real Demucs stems) too slow for production | `auto` only engages Beat-Transformer when Demucs is already opted into (`CHORD_STEM_PREFER_DEMUCS`); default stem mode is unchanged, so this doesn't regress today's latency by default |
| madmom install friction on Windows (flagged by chordmini's own README) | Production is Docker/Linux (unaffected); local Windows dev may need a documented workaround (WSL or a known-good wheel source) — captured in the implementation plan, not blocking design |
| Vendoring Beat-Transformer's code drifts from upstream fixes | Vendor a pinned commit/tag, note it in the module docstring, same convention as other vendored contracts in this repo |
