# Phase 12.5 — post-processing ablation (v47 ruler)

Configs: **raw** | **pitch_only** | **merge_only** (chroma align + gated merge, no pitch) | **serving** (both)

## Aggregate majmin / seg

| Config | DEV majmin | DEV seg | TEST majmin | TEST seg |
|--------|------------|---------|-------------|----------|
| raw | 0.748 | 0.827 | 0.710 | 0.669 |
| pitch_only | 0.787 | 0.820 | 0.743 | 0.662 |
| merge_only | 0.723 | 0.792 | 0.707 | 0.654 |
| serving | 0.762 | 0.789 | 0.707 | 0.628 |

v47 serving baseline (old): DEV 0.762 / TEST 0.707 majmin

## Decision

**Chosen serving path:** `pitch_only` → shipped as **`CHORD_ML_POSTPROCESS=raw`** (v48 baseline)

Pitch correction alone improves TEST majmin (+3.3pp vs raw); chroma-align + gated light-merge **removed from default** (harmful on v47 — e.g. `let-it-be` 0.874→0.585 majmin). Revert via `CHORD_ML_POSTPROCESS=bypass`.

**v48 baselines:** DEV **78.7%** / TEST **74.3%** majmin (+2.5pp / +3.6pp vs v47 bypass serving). See `BASELINE_v48_README.md`.

## Pitch correction per track (majmin Δ pitch_only − raw)

| id | Δ majmin | shift (st) |
|----|----------|------------|
| another-one-bites-the-dust | -0.114 | 0.36 |
| play-the-game | -0.028 | -0.12 |
| yesterday | -0.025 | 0.13 |
| crazy-little-thing-called-love | -0.022 | -0.14 |
| dont-stop-me-now | -0.020 | -0.11 |
| back-in-the-ussr | -0.019 | -0.15 |
| i-saw-her-standing-there | -0.008 | 0.17 |
| help | -0.006 | 0.18 |
| something | -0.002 | 0.08 |
| seven-seas-of-rhye | -0.001 | 0.15 |
| twist-and-shout | -0.001 | 0.23 |
| let-it-be | +0.000 | 0.00 |
| come-together | +0.000 | 0.00 |
| norwegian-wood | +0.000 | 0.00 |
| cant-buy-me-love | +0.009 | -0.15 |
| while-my-guitar-gently-weeps | +0.011 | -0.07 |
| get-back | +0.016 | -0.07 |
| here-comes-the-sun | +0.018 | -0.33 |
| somebody-to-love | +0.044 | 0.18 |
| youre-my-best-friend | +0.045 | 0.19 |
| penny-lane | +0.116 | 0.18 |
| ticket-to-ride | +0.129 | 0.38 |
| hammer-to-fall | +0.204 | 0.37 |
| yellow-submarine | +0.531 | 0.18 |

JSON: `analysis/PHASE12_5_POSTPROCESS_ABLATION.json`
