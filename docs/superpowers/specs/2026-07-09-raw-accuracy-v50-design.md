# Raw accuracy lift, v50 — design

**Date:** 2026-07-09
**Status:** approved
**Owner:** Adeen

## Context

Serving (v49) runs lv-chordia raw segments plus a reliability-gated ±1st pitch
correction. Gold v2 baseline: **TEST majmin WCSR 73.0%, DEV 76.1%**
(`BASELINE_mir_gold_{DEV,TEST}_v49.json`). The v49 pitch gate
(`pitch_utils._estimate_reliable`) fixed `another-one-bites-the-dust`
(1.4% → 12.8%) but cost 1.3pp aggregate vs v48 (74.3%).

Key fact discovered during design: `lv_chordia.chord_recognition()` already
runs a 5-checkpoint ensemble, averages frame posteriors, and decodes them with
an XHMM (Viterbi) decoder. The ChordLift call site
(`backend/chord_engine_chordia.py`) discards the posteriors and keeps only
decoded labels. All levers below operate at that posterior level.

## Goal

+3–5pp TEST majmin WCSR over v49 (target 76–78%). DEV picks candidates, TEST
confirms once, both splits reported for every change (gold-v2 rules
unchanged). `another-one-bites-the-dust` must not regress below its v49 score.

## Non-goals

- BTC ensemble (no BTC code in repo; pretrained weights saw Isophonics).
- Presentation-layer / perceived-accuracy work (Phase 13 owns that).
- New eval infrastructure (`eval_gold_mir.py` + rebaseline scripts suffice).
- Genre generalization beyond the gold v2 ruler.

## Design

### Lever 0 — expose frame posteriors (plumbing)

New function in `backend/chord_engine_chordia.py`:

```
recognize_chordia_probs(y_chord, sr) -> (probs, hmm, entry)
```

Replicates `lv_chordia.chord_recognition()` internals (CQTV2 extraction,
5 × `ChordNet.inference`, per-checkpoint prob averaging) but returns the
averaged posteriors plus the `XHMMDecoder` and `DataEntry` needed to decode.
No edits inside site-packages. Existing `recognize_chordia()` becomes a thin
wrapper (probs → `hmm.decode_to_chordlab`).

Self-check: posterior rows sum to ~1; wrapper reproduces the current serving
output exactly on a fixture track.

### Lever 1 — decode-both pitch selection (replaces threshold gate)

Current gate predicts whether pitch correction helps via four tuned thresholds
(harmonic/full-mix delta, key-fit gain, bass-heavy heuristics). Prediction is
the failure mode: v49's gate skips corrections that would have helped
(−1.3pp aggregate).

Replace prediction with measurement. When `|shift| ≥ MIN_SHIFT_SEMITONES`:

1. Build the pipeline context once from raw audio; pitch-shift the **chord
   stem** (not the full mix) for the corrected branch. Stems and beat grid are
   computed once — only chordia runs twice.
2. Decode both branches (lever 0 path); score each by decoder confidence =
   mean over frames of max posterior.
3. Keep the corrected timeline only if its confidence beats raw by a margin
   (tunable, single constant, picked on DEV). Tie or below margin → raw.
   Winner's shift value propagates to `pitch_correction_semitones` (capo UI
   depends on it).

Serving today corrects the full mix *before* `build_chord_pipeline_context`
(`chord_engine.py:48`), so stems derive from corrected audio. Shifting only
the stem is near-equivalent and half the cost; if DEV shows a gap vs v48's
known corrected-path numbers, fall back to running the full pipeline twice.

Deletes `_estimate_reliable` and its four threshold constants; keeps
`MIN_SHIFT_SEMITONES` / `MAX_SHIFT_SEMITONES` clamps. `chord_engine.py:48` is
the only serving caller. Cost: 2× chordia inference on detuned tracks only
(analysis is async and cached).

### Lever 2 — TTA posterior averaging (alternative to lever 1)

Same plumbing: average frame posteriors from {raw, corrected} runs, decode the
average once. Shares ~90% of lever 1's code. Levers 1 and 2 are mutually
exclusive in serving — DEV decides which ships.

### Lever 3 — chord-dict sweep (config only)

`CHORD_CHORDIA_DICT` supports `submission` (current), `ismir2017`, `full`,
`extended`. One DEV eval per dict; if a non-default dict wins on DEV, confirm
on TEST. Zero code.

## Measurement protocol

- One experiment = one `scripts/rebaseline_v50.py` run (same pattern as
  `rebaseline_v49.py`), producing `BASELINE_mir_gold_{DEV,TEST}_v50.json`.
- Order: lever 0 self-check → lever 3 sweep (cheapest) → lever 1 vs lever 2
  on DEV → winning combination confirmed on TEST once.
- Per-track deltas reported for the gate-sensitive tracks
  (`another-one-bites-the-dust`, `let-it-be`, `help`, `yellow-submarine`).

## Acceptance criteria

- TEST majmin WCSR ≥ 76.0% (stretch 78%), DEV reported alongside.
- `another-one-bites-the-dust` ≥ 12.8% (v49 level).
- No serving latency regression beyond 2× on detuned tracks.
- `verify_model_disagreement.py` CI guard still passes.

## Phase B (separate spec, after Phase A ships)

Fine-tune the serving model itself on the 2060 Super: lv_chordia ships its
training code (`chordnet_ismir_naive.py`, `datasets.py`), so fine-tuned
checkpoints drop in via `MODEL_NAMES`. Training data: Isophonics minus the 24
gold tracks, leakage verified by `scripts/verify_training_leakage.py`
(track id AND isophonics path AND recording basename). Do not start until
Phase A's ceiling is measured — its result changes what Phase B must beat.

## Risks

| Risk | Mitigation |
|---|---|
| Confidence contest picks wrong on near-ties | margin + default-to-raw |
| TTA blurs genuinely detuned tracks | lever 1 vs 2 decided empirically on DEV |
| Stem-shift branch diverges from full-mix correction | fall back to double pipeline if DEV gap vs v48 |
| Posterior plumbing drifts from lv_chordia internals on upgrade | wrapper self-check fixture pins behavior |
| DEV overfit (14 tracks) | TEST confirmed once, at the end, per gold rules |
