# Phase B Desktop Training RUNBOOK

Run this on Adeen's Windows-or-Linux desktop (RTX 2060 Super, sm_75) to
fine-tune the five warm-start seeds against `training_bundle.zip`. Every
command below is meant to be copy-pasted as-is.

## 1. Prereqs

- **Python 3.11** (match the Mac dev venv's version — mismatched minor
  versions can break the `h5py`/`torch` wheel ABI).
- **NVIDIA driver** new enough for CUDA 12.1 (>= 530.xx). Check with
  `nvidia-smi`; the top-right "CUDA Version" must read >= 12.1.
- Create and activate a fresh venv:

  ```bash
  python3.11 -m venv train-venv
  # Linux:
  source train-venv/bin/activate
  # Windows (cmd.exe):
  train-venv\Scripts\activate.bat
  ```

- Install **torch from the CUDA wheel index first**, before anything else —
  this is what makes the GPU visible; a plain `pip install torch` pulls a
  CPU-only build:

  ```bash
  pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu121
  ```

  cu121 wheels support sm_75 (the 2060 Super's compute capability). If this
  exact version isn't on that index by the time you run this, check
  https://pytorch.org/get-started/locally/ for the current CUDA index name
  and substitute it here — the rest of this runbook is unaffected either way.

- Then install everything else (this will **not** touch the torch you just
  installed — `lv-chordia` only requires `torch>=1.4.0`):

  ```bash
  pip install -r requirements-train.txt
  ```

- Verify CUDA is actually visible before moving on:

  ```bash
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

  Must print `True` and the 2060 Super's name. If not, see Troubleshooting.

## 2. Unzip the bundle

```bash
unzip training_bundle.zip -d bundle
cd bundle
```

Layout you should see: `data/`, `train_manifest.txt`, `val_manifest.txt`,
`kit/` (`dataset.py`, `finetune.py`, `FINDINGS.md`), `checkpoints_in/` (five
pretrained `.sdict` files), this `RUNBOOK.md`, `requirements-train.txt`.

## 3. Dry-run gate — run this before the long run, every time

```bash
python kit/finetune.py --train-manifest train_manifest.txt --val-manifest val_manifest.txt \
    --work-dir out --pretrained-dir checkpoints_in --seeds 0 --dry-run
```

This does one forward/backward pass on seed 0 only and writes nothing. It
**must print `DRY-RUN OK`**. If it doesn't (traceback, hang, wrong path
errors), stop — do not start the full run until this is clean.

`--pretrained-dir checkpoints_in` points `finetune.py` at the bundle's own
copy of the five pretrained seed checkpoints rather than whatever
`lv-chordia`'s pip install happened to place under the venv's `share/`
directory — the bundle is self-contained regardless of that.

## 4. Full run

```bash
python kit/finetune.py --train-manifest train_manifest.txt --val-manifest val_manifest.txt \
    --work-dir out --pretrained-dir checkpoints_in \
    --seeds 0,1,2,3,4 --lr 1e-4 --epochs-cap 30 --batch-size 8
```

Seeds run **sequentially, one at a time** (not in parallel) — five full
per-seed fine-tunes, each up to 30 epochs (early-stops after 10 epochs of no
val-loss improvement).

**Runtime estimate.** The only measured timing we have is CPU on the Mac
spike (FINDINGS.md §10): one epoch over 2 songs x 12 pitch-shift samples,
batch size 2, 12 batches/epoch, took **5.3 s** (~0.44 s/batch). Runtime
scales with **track count**, not track length — every song contributes a
fixed 12 samples/epoch regardless of duration, as long as it clears the
~24 s floor (FINDINGS.md §1). Rough formula:

```
batches/epoch  = ceil(train_tracks * 12 / batch_size)
epoch_time     = batches/epoch * 0.44s        (CPU baseline, per batch)
full_run_time  = epoch_time * epochs_cap * len(seeds)   [worst case, no early stop]
```

Example for a 40-track train manifest at `--batch-size 8`: 60 batches/epoch
-> ~26 s/epoch (CPU-equivalent) -> ~13 min/seed at 30 epochs -> **~65 min**
worst case for all 5 seeds. Two things push the real number lower and one
pushes it higher:
- The 2060 Super should be meaningfully faster per batch than the CPU
  number above — untested here, so treat the formula as a **pessimistic
  upper bound**, not a promise.
- `early_end_epochs=10` patience means most seeds will stop well short of
  30 epochs once val loss plateaus.
- CQT extraction (`build_storages`, before any epoch starts) adds a
  one-time ~1 s/30 s-clip pass over the whole manifest (FINDINGS.md §10) —
  budget a few extra minutes up front for a real-sized dataset.

Recompute the batch count for your actual bundle with
`wc -l train_manifest.txt` before you commit to a schedule.

**VRAM knob:** if you hit `CUDA out of memory`, drop `--batch-size 4` (or
`2`). Re-run the Step 3 dry-run first to confirm it clears before starting
the full run again.

## 5. Copy back

When done, copy these back to the Mac:

```
out/cache_data/*_ft1_s*.best.best.sdict
```

One `.best.best.sdict` per seed (the val-loss-best checkpoint from that
seed's run) — five files total if all seeds completed. Note the **double**
`.best`: `save_name` already ends in `.best` (matching the pretrained
warm-start naming), and `train.py` appends another `.best.sdict` on val-loss
improvement (see `FINDINGS.md` §9). The single-suffixed `*_ft1_s*.best.sdict`
is the *final* end-of-schedule checkpoint, not the best-val one — loading it
instead silently ships a worse (sometimes badly overfit) model.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False`, or training is CPU-slow despite having a GPU | Driver/CUDA mismatch, or torch was installed from plain PyPI (CPU wheel) instead of the cu121 index | Reinstall: `pip uninstall torch -y && pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu121`. `lv_chordia` auto-detects the GPU (`torch.cuda.device_count()>0`) — there's no flag to force it on. To deliberately force **CPU** instead (e.g. to isolate a GPU-only bug), set `CUDA_VISIBLE_DEVICES=` (empty) before running — consequence: back to the ~5.3s/12-batch CPU baseline above, so the full run goes from ~1 hour to potentially many hours. |
| `RuntimeError: CUDA out of memory` | Batch too large for the 2060 Super's 8GB VRAM | `--batch-size 4` or `--batch-size 2`; re-run the Step 3 dry-run to confirm before the full run |
| Val loss never improves across seeds | Val set too small, or too similar/dissimilar to train, rather than a code bug | Swap in a different held-out val set — done **on the Mac**, not the desktop, since only the Mac has the full data + `build_manifest.py`: `cd backend && .venv/bin/python chord_training/build_manifest.py --data-dir /path/to/isophonics_heldout --out /path/to/new_val_manifest.txt --min-pairs 1`, then re-pack: `.venv/bin/python chord_training/pack_bundle.py --train-manifest train_manifest.txt --val-manifest /path/to/new_val_manifest.txt --out training_bundle_v2.zip`, and copy the new zip to the desktop |
