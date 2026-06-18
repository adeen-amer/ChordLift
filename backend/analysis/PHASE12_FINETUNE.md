# Phase 12 — Fine-tune (revised after Phase 11.6)

**Phase 11.6 (v47 ruler):** RAW chordia leads BTC on TEST (−3.5pp). BTC fine-tune base **deprioritized** — prefer chordia fine-tune or skip. See `analysis/PHASE11_6_BTC_V47.md`.

**Serving default stays lv-chordia (ensemble)** — unchanged.

**Previous plan (BTC base):** pretrained `btc_model_large_voca.pt` — retained as optional experiment only.

---

## Contamination rules (#1 risk)

1. **Hold out ALL 24 gold tracks** (DEV+TEST) from train AND val.
   Manifest: `analysis/gold_holdout_v2.json`
2. **BTC pretrained weights saw Isophonics** during original training — gold TEST RAW 65.6% may be **slightly optimistic**. Note in all Phase 12 reports.
3. Verify by **track id AND isophonics path AND recording basename** before any train run:
   ```bash
   python scripts/verify_training_leakage.py data/train_manifest.txt
   python scripts/finetune_btc.py --manifest ... --val-manifest ... --dry-run
   ```

---

## Training data (target)

| Source | Role |
|--------|------|
| UsPop2002 | Core pop/rock |
| McGill Billboard | Diverse pop |
| RWC Popular | Japanese/western pop |
| Isophonics (non-gold only) | Same label syntax; **exclude 24 gold .lab paths** |

**Not allowed:** any of the 24 gold recordings in train/val.

**AAM artificial audio:** optional ~1pp supplement only. Do **not** train on AAM alone (Billboard root 72.8 AAM vs 82.0 real).

---

## Augmentation

- **Pitch-shift:** 12 keys (−5..+6 semitones), transpose Harte labels with audio (mir_eval / pyrubberband).
- Standard, cheap, high-value for chord recognition.

---

## Fine-tune protocol

1. Load pretrained BTC large-vocab checkpoint + normalization stats from checkpoint.
2. Fine-tune with Adam lr=1e-4 (BTC default), early-stop on **gold-free val** (no improvement 10 epochs).
3. Eval on gold DEV+TEST via `eval_gold_mir.py` with `CHORD_MODEL=btc` — **post-process TBD** (bypass hurts pretrained BTC; re-measure after fine-tune).

---

## Gate

- **TEST majmin > 66.4** (v46) with **provable zero gold leakage**
- Report DEV+TEST + per-track
- **Target:** high-60s/low-70s realistic; mid-70s = strong on this gold

---

## Scaffold (in repo)

| File | Purpose |
|------|---------|
| `analysis/gold_holdout_v2.json` | 24 excluded track ids + isophonics paths |
| `scripts/verify_training_leakage.py` | Fail if manifest overlaps gold |
| `scripts/finetune_btc.py` | Entry point (manifest + dry-run; train loop next) |

---

## Next implementation steps

1. Download/prepare UsPop + Billboard + RWC + filtered Isophonics audio+labs
2. Build `train_manifest.txt` / `val_manifest.txt` (gold-free val split)
3. Wire pitch-shift dataset wrapper around BTC CQT features
4. Train → checkpoint → gold eval → compare v46

**Phase 13 (after base locked):** delete dead `polish_chord_timeline` / strict `bar_decode` from full path.
