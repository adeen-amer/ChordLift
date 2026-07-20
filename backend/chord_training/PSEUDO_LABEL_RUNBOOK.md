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
