# Phase B — fine-tune the serving chordia checkpoints — design

**Date:** 2026-07-10
**Status:** approved (user delegated approach choice; approach A selected)
**Owner:** Adeen

## Context

v50 shipped (merged PR #1): decode-both pitch selection, gold-v2 DEV 80.3% /
TEST 75.0% majmin WCSR. The 76% TEST spec target was not reached; the
remaining lift must come from the model itself. Serving runs a 5-checkpoint
lv_chordia ensemble (`joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{0..4}.best`,
`.sdict` files in `backend/.venv/share/lv-chordia/cache_data/`).

Ground truth about the training surface (verified in the installed package):

- `NetworkInterface.train_supervised(train_set, val_set, batch_size, …,
  learning_rates_dict, early_end_epochs)` exists (`mir/nn/train.py:104`).
- Loading an existing `.sdict` sets `finalized=True` and `train_supervised`
  **refuses to run** on a finalized model. Warm-start therefore means: create
  the interface under a NEW save_name, manually
  `net.load_state_dict(torch.load(orig_sdict)['net'])`, keep `finalized=False`,
  then train.
- The package's `create_*_dataset()` functions hardcode the original
  researcher's Windows paths — unusable. The kit builds its own train/val set
  objects (DataLoader-compatible: `.num_workers`, `.need_shuffle`) from
  (audio, .lab) pairs via the package's DataEntry/extractor APIs.
- Training hardware: Adeen's desktop (RTX 2060 Super, 8GB, sm_75). No remote
  access — Adeen runs commands there manually. The deliverable is a
  self-contained kit + runbook.
- Old `PHASE12_FINETUNE.md` is stale (BTC-centric; BTC was removed). Its
  contamination rules carry over; its model choice does not.

## Goal

Beat v50 on the gold-v2 ruler with fine-tuned serving checkpoints:
**TEST majmin ≥ 76.0%** (minimum shippable: > 75.0% = beats v50), DEV no
regression, guard track (`another-one-bites-the-dust`, DEV) ≥ 0.128. All v50
protocol rules kept (DEV picks, TEST once, both reported).

## Non-goals

- No new model architectures, no BTC revival, no Conformer.
- No gold-v3 ruler expansion in this phase (the Billboard tooling built here
  enables it later).
- No serving-latency changes; fine-tuned checkpoints are drop-in weights.
- No training-infra generalization beyond what this fine-tune needs.

## Design

### Task 0 — feasibility spike (Mac, CPU, blocks everything)

Prove on this Mac with 2 tracks and ~1 epoch that:
1. A pretrained `.sdict` can be loaded into a fresh `NetworkInterface` under a
   new save_name with `finalized=False`, and `train_supervised` runs and
   saves a new checkpoint.
2. A minimal train/val set object satisfying the DataLoader contract can be
   built from (audio, .lab) pairs, producing the CQT features + 6-component
   label tensors the loss expects (`tag.view((-1,6))` — triad/bass/7/9/11/13
   per frame, derived from Harte labels via the package's chord parsing).
3. The new checkpoint loads through `recognize_chordia_probs` and decodes.

If the spike fails on (1) or (2) after honest effort, STOP the phase and
rescope (fallback: sevenths/vocab accuracy work, no GPU needed).

### Data

- **Train:** Isophonics non-gold — Beatles and Queen tracks with Harte v1.2
  `.lab` annotations, excluding the 24 gold recordings. Labs via the existing
  `scripts/extract_isophonics_labs.py`; audio via the existing
  spotdl/downloader + duration identity gate (same pipeline that built gold
  audio). Target ≥ 120 usable tracks.
- **Val (early stopping):** McGill Billboard subset (~40 tracks, public
  chord annotations, 1958–91 pop/soul/country/disco/rock). Genre-wider than
  the train set by construction — early stopping on it resists
  Beatles-overfit. Annotation→`.lab` converter is new, small. Audio via the
  same downloader + identity gate.
- **Leakage:** every manifest passes `scripts/verify_training_leakage.py`
  (track id AND isophonics path AND recording basename vs
  `analysis/gold_holdout_v2.json`) before any run. Gold tracks appear in
  NEITHER train NOR val.
- **Augmentation:** none in v1. The inference path slices a fixed
  transposition window from the CQT (`SHIFT_HIGH*SHIFT_STEP`); if the spike
  reveals native pitch-shift augmentation support, it may be enabled as a
  flagged option — otherwise deferred.

### Training kit (`backend/chord_training/`)

Replaces the empty stub. Contents:

- `build_manifest.py` — scans a data dir of (audio, lab) pairs → 
  `train_manifest.txt` / `val_manifest.txt`; runs the leakage check.
- `dataset.py` — the train/val set objects (CQT extraction via package
  extractors, Harte→6-component label encoding, frame alignment).
- `finetune.py` — per-seed warm-start loop: for each of s0..s4, new save_name
  `…_ft1_s{i}.best`, load pretrained weights, `train_supervised` with Adam
  lr=1e-4 (pass `learning_rates_dict=1e-4` — the API accepts a plain float), 
  `early_end_epochs=10`, batch size tuned to 8GB. `--dry-run` flag validates
  data + one forward/backward without saving.
- `pack_bundle.py` (runs on Mac) — zips audio+labs+manifests+kit into
  `training_bundle.zip` for transfer to the desktop.
- `RUNBOOK.md` — desktop setup (Python 3.11, CUDA torch for sm_75,
  `pip install graphifyy`-style one-liners for lv_chordia + deps), exact
  commands (verify → dry-run → train s0..s4), expected runtimes, what to copy
  back (`checkpoints/*_ft1_s*.best.sdict`).

### Deployment + eval (Mac)

- `chord_engine_chordia.py`: read env `CHORD_CHORDIA_MODELS`
  (comma-separated save_names; default = the packaged five). Fine-tuned
  `.sdict` files are dropped into the same shared `cache_data` dir; no
  site-packages code edits. One test: env override changes the model list the
  wrapper uses.
- `scripts/rebaseline_v51.py` — v50-pattern script, `--models` passthrough,
  diff vs v50.
- Eval matrix on DEV only: (a) all-five fine-tuned, (b) mixed ensemble
  (fine-tuned + original checkpoints) if (a) is close. Winner → TEST once.

### Runbook flow (Adeen's loop)

1. Mac: build data + bundle (`pack_bundle.py`), copy to desktop.
2. Desktop: follow `RUNBOOK.md` → returns 5 fine-tuned `.sdict` files.
3. Mac: drop into cache_data, `rebaseline_v51.py --split dev`, iterate if
   needed (lr/epochs are runbook knobs), TEST once at the end.

## Acceptance criteria

- Spike (Task 0) passes before any kit/desktop work ships.
- TEST majmin ≥ 76.0% stretch; > 75.0% minimum shippable. DEV ≥ 80.3%
  (no regression). Guard track ≥ 0.128.
- Zero gold leakage: `verify_training_leakage.py` green on both manifests,
  output archived with the baselines.
- Fine-tuned checkpoints load via env override only; packaged files untouched.
- `BASELINE_v51_README.md` documents data, hyperparameters, all runs.

## Risks

| Risk | Mitigation |
|---|---|
| Warm-start API fight (finalized flag, optimizer state) | Task 0 spike on CPU before anything ships to the desktop |
| Label encoding mismatch (Harte → 6-head tags) | Spike asserts loss runs on real encoded batch; reuse package chord parsing, never hand-roll |
| Isophonics archives / audio unavailable | Manifest builder reports coverage; ≥80 tracks floor or stop and reassess |
| Billboard val too hard/noisy → early stop fires immediately | Runbook knob: fall back to a held-out Isophonics val split (genre guard lost but training unblocked; documented) |
| Fine-tune overfits Beatles despite val guard | Mixed-ensemble eval option; DEV-vs-TEST gap watched in v51 README |
| 8GB VRAM insufficient | Model is small (BiLSTM-scale); batch-size knob in runbook; CPU-fallback flag |
| Desktop env friction (CUDA/torch/python drift) | RUNBOOK pins exact versions; dry-run validates end-to-end before the long run |

## Protocol

DEV picks candidates, TEST once at the end, both reported. Fine-tune
hyperparameter iteration happens against the Billboard val loss and gold DEV
only. v50's TEST (0.750) is the number to beat; this phase must not consult
TEST more than once.
