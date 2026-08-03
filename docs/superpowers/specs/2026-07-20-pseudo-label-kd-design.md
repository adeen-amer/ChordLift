# Pseudo-labeling / knowledge distillation on FMA audio — design

**Date:** 2026-07-20
**Status:** approved
**Owner:** Adeen

## Context

v51's fine-tune attempt (`BASELINE_v51_README.md`) plateaued: DEV majmin
79.3% vs v50's 80.3%, and the guard track
`another-one-bites-the-dust` collapsed to 0.014 (floor 0.128). Root cause,
confirmed by direct A/B (fine-tuned ensemble's mean decoder confidence on
that track was *higher* than v50's pretrained model, 0.538 vs 0.487 — it's
confidently wrong, not undertrained): the 161-track fine-tuning corpus
(Isophonics Beatles + Queen, plus a Billboard-sourced val set) is
Beatles/Billboard-skewed pop-rock and underrepresents genres like this
track's bass-and-drums-led funk/disco. v51 was not shipped; v50 remains the
serving baseline.

This is one of five items on the "match chordmini's feature set" backlog
(chosen first of the five). chordmini's 2026 paper "Enhancing Automatic
Chord Recognition via Pseudo-Labeling and Knowledge Distillation" targets
exactly this kind of corpus-coverage gap by expanding effective training
data from unlabeled audio rather than hand-annotating more Harte labels.

Existing pipeline this design reuses instead of rebuilding:

- `chord_engine_chordia.py:recognize_chordia_probs` — runs the production
  v50 5-checkpoint ensemble and returns per-frame softmax posteriors before
  HMM decoding. `decode_chordia_probs` turns posteriors into
  `{start_time, end_time, chord}` segments.
- `chord_training/dataset.py:encode_labels` — frame-aligns a `.lab` file to
  training labels; frames with **no segment covering them** already fall
  back to `X` (masked out of the loss), the same path used today for
  unparseable Harte labels.
- `chord_training/finetune.py` + `dataset.py:build_storages` — consume a
  `(audio_path, lab_path)` manifest, build CQT/label storages, run the
  per-seed warm-start fine-tune. Neither needs to change.
- `chord_training/fetch_train_data.py` — staged fetch pattern (labs → plan →
  download) this new script mirrors.

## Goal

Produce a pseudo-labeled supplement to the 161-track training corpus, mixed
into the existing fine-tune manifest, and re-run the v50/v51 fine-tune
protocol to see whether it clears the shipping bar v51 missed
(`TEST > 75.0%` minimum, DEV no regression, guard track ≥ 0.128).

## Non-goals

- No soft-label / KL-divergence distillation loss — `lv_chordia`'s
  `conditional_classifier_loss` expects hard class indices; introducing a
  new loss function is a materially bigger change than this phase needs.
  Hard pseudo-labels, confidence-filtered, reuse the existing loss and
  masking path unchanged.
- No genre-targeted curation — broad random FMA sampling only. (Targeted
  funk/disco sourcing was considered and explicitly declined in favor of
  general corpus diversity; can be revisited later if broad sampling
  doesn't move the needle.)
- No BTC or any second model architecture (separate backlog item).
- No changes to `dataset.py`'s label-encoding/masking logic — the confidence
  filter operates entirely in the new pseudo-labeling script, upstream of
  the existing pipeline.

## Design

### New script: `chord_training/pseudo_label.py`

Staged, mirroring `fetch_train_data.py`:

1. **`--stage fetch`** — pull a broad random sample from FMA's full-length
   audio archive (not the `small`/`medium` 30s-clip tiers — those fall
   under `LSTM_TRAIN_LENGTH`'s sample-length floor the same way the
   existing 20s test fixtures do). Target ~150-250 tracks after filtering,
   comparable order of magnitude to the 161 real tracks (a supplement, not
   a corpus replacement — real hand-labeled data stays the trusted anchor
   each fine-tune epoch sees).
2. **`--stage label`** — for each fetched track:
   - `recognize_chordia_probs(y, sr)` → frame posteriors from the same v50
     ensemble already serving production.
   - `decode_chordia_probs(probs, hmm, entry)` → chord segments.
   - Compute each segment's mean confidence (max triad posterior over its
     frames, same quantity `chordia_confidence` already computes track-wide).
   - Drop any segment below `--confidence-threshold` (default TBD
     empirically — start around 0.7, tune against DEV before the real
     fine-tune run) before writing.
   - Write the surviving segments as a standard `.lab` file
     (`start end chord`, same format `read_lab`/`encode_labels` already
     parse). Dropped segments simply leave a gap; `encode_labels`'s
     existing "no segment covers this frame → X" path masks them from the
     loss with zero changes to `dataset.py`.
   - If retained segment coverage falls below ~50% of track duration
     (ambient/spoken-word/noisy FMA entries the model is naturally
     unsure about), skip the track and report it to stderr — same
     convention `build_storages` already uses for unencodable pairs.
3. **`--stage manifest`** — append the pseudo-labeled `(audio, lab)` pairs
   to `train_manifest.txt` alongside the 161 real tracks.

### Training + evaluation

Run `finetune.py` exactly as today (same warm-start recipe, same
`--lr`/`--epochs-cap`/`--seeds` knobs) against the combined manifest. Reuse
the v50/v51 evaluation protocol unchanged: DEV gold picks candidates, guard
track (`another-one-bites-the-dust`) must stay ≥ 0.128, TEST is run once
only if DEV clears the bar. `scripts/rebaseline_v51.py`-style diffing
against v50 is reused, not rebuilt.

### Licensing note

FMA audio is mixed-license (CC-BY, CC-BY-NC, CC0). Used here only as
training input, not redistributed — acceptable, flagged for awareness, not
a design blocker.

## Testing

One `pytest` unit test for the segment-confidence-filtering function in
`pseudo_label.py`: given synthetic probs/segments, assert low-confidence
segments are dropped and the `.lab` writer produces valid, parseable
output. Matches the repo's existing narrow-unit-test convention (e.g.
`test_model_list_env_override_preserves_commas_in_parens`).

## Acceptance criteria

- Pseudo-label pipeline runs end-to-end on a small FMA sample and produces
  `.lab` files `read_lab`/`encode_labels` parse without modification.
- Combined-manifest fine-tune clears the v50/v51 protocol's bar: guard
  track ≥ 0.128, DEV no regression vs v50 (80.3%), and if DEV clears,
  TEST run once (target: beat v50's 75.0%, matching v51's original
  76.0% stretch goal).
- If the guard track or DEV bar isn't cleared, the result and diagnosis are
  documented in a `BASELINE_v52_README.md`-style writeup (same honesty
  standard as `BASELINE_v51_README.md`), same as any other non-shipped
  attempt.

## Risks

| Risk | Mitigation |
|---|---|
| Confidence threshold too loose → teacher's own funk/disco blind spots get baked in as "ground truth" | Start conservative (~0.7), validate a manual spot-check sample of pseudo-labels before the real fine-tune run |
| Confidence threshold too tight → most segments dropped, supplement too sparse to matter | Track-level coverage stat reported per fetch/label run; adjust threshold empirically |
| FMA full-length archive access/fetch friction (large files, sharded by track id) | Small target sample (~150-250 tracks) keeps this bounded; staged `--stage fetch` script isolates the concern |
| Pseudo-label supplement doesn't move the needle | Per the backlog: BTC (a genuinely different architecture) is the fallback, not another hyperparameter pass on this same approach |
| Repeats v51's `_model_names()`-style silent-random-weights class of bug | Reuses `recognize_chordia_probs`/`finetune.py` verbatim — no new checkpoint-loading code path introduced |

## Protocol

Same as v50/v51: DEV picks candidates, TEST run once at the end, both
reported. This phase must not consult TEST more than once.
