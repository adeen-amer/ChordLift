# FMA pseudo-labeling RUNBOOK

Produces a `train_manifest.txt` supplement of confidence-filtered FMA
pseudo-labels, to be fed into `pack_bundle.py` and fine-tuned via the
existing `RUNBOOK.md` (desktop GPU fine-tune + eval) unchanged.

## 0. Why archive.org and not the FMA API

The design spec assumed FMA's own metadata API. It is retired: as of
2026-07-30 every documented endpoint returns 404 (HTML, not JSON),
including `freemusicarchive.org/api/agreement` — the page you would request
a key from. There is no key to obtain.

The same catalog is on archive.org as `collection:freemusicarchive`
(16,794 items, full-length audio, public endpoints, **no API key**), so the
fetch stage targets that. This also removes the spec's 342MB
`fma_metadata.zip` download: the id pool is scraped once into a ~500KB text
file and cached.

No setup step. No credentials. Skip straight to fetching.

## 1. Fetch a sample

```bash
cd backend
python chord_training/pseudo_label.py --stage fetch --n 200 --seed 0
```

Writes audio to `chord_training/data/pseudo/` and caches the id pool to
`chord_training/data/fma_ia_pool.txt` on first run (~10s, two requests).
Re-runnable: already-downloaded files are not re-fetched.

`--min-duration 60 --max-duration 600` are the defaults and both matter. A
random sample of this collection surfaces hour-long ambient drones and live
WFMU radio shows (talk interleaved with music); the teacher labels those
confidently with 30-second single-chord segments, which passes every
confidence filter downstream while being useless as song-like harmonic
training data. Raise `--max-duration` only if you specifically want
long-form material.

## 2. Label with confidence filtering

```bash
python chord_training/pseudo_label.py --stage label \
    --confidence-threshold 0.6 --min-coverage 0.5
```

Prints `labeled: <kept>/<total> (threshold=..., mean kept coverage=...)`.
Tune from that one line: many skips means the threshold is too tight, mean
coverage near 1.0 means it is too loose. Re-running this stage is cheap — no
download needed.

**Measured on real full-length FMA tracks** (two tracks, teacher mean triad
posterior ≈ 0.635 on both):

| `--confidence-threshold` | retained coverage |
|---|---|
| 0.5 | 0.77 / 0.70 |
| 0.6 | 0.65 / 0.49 |
| 0.7 | 0.43 / 0.37 |
| 0.8 | 0.17 / 0.25 |
| 0.9 | 0.02 / 0.14 |

This is why the default is **0.6, not the spec's suggested 0.7**: at 0.7
both tracks land under `--min-coverage 0.5` and get discarded outright — the
spec's "threshold too tight" risk, realized. Two tracks is not a corpus, so
re-read the printed stats on your first 200-track run before trusting 0.6.

Before the real fine-tune, spot-check a handful of written `.lab` files by
ear against their source audio (spec Risk: "confidence threshold too
loose").

## 3. Merge into the real train manifest

```bash
python chord_training/pseudo_label.py --stage manifest \
    --train-manifest /path/to/train_manifest.txt
```

Fails (non-zero exit) if the merge introduces any gold-holdout leakage —
this should never happen with FMA (a disjoint catalog from Isophonics), but
the check runs unconditionally, same as every other manifest-building path
in this repo.

## 4. Fine-tune and evaluate

Follow the existing `RUNBOOK.md` unchanged, using the now-supplemented
`train_manifest.txt`. Evaluation protocol (DEV picks, guard track >= 0.128,
TEST once) is identical to v50/v51 — see the design spec's Evaluation
section.

## Licensing note

FMA audio is mixed-license (CC-BY, CC-BY-NC, CC0). Used here as training
input only, never redistributed — fine for this use, not a blocker.
