# Phase 11 — RAW BTC vs RAW lv-chordia bake-off

**Measurement only** — no default change, no post-processing applied.  
**BTC:** pretrained `btc_model_large_voca.pt` (full mix, CQT features).  
**lv-chordia:** raw ensemble model on HPSS chord stem (`pipeline.y_chord`).

Run: `make phase11-btc-bakeoff` → `analysis/PHASE11_BTC_BAKEOFF.json`

---

## Aggregate (mir_eval majmin / seg)

| Split | RAW chordia majmin / seg | RAW BTC majmin / seg | Δ majmin |
|-------|--------------------------|----------------------|----------|
| **DEV (14)** | 61.5% / 75.2% | **66.3% / 76.6%** | **+4.8pp** |
| **TEST (9)** | 62.9% / 63.6% | **65.6% / 74.1%** | **+2.7pp** |

v46 production baseline (chordia + bypass post-process): DEV **66.9%** / **72.0%** | TEST **66.4%** / **61.3%**

RAW BTC on TEST (65.6%) is **below** v46 post-processed chordia (66.4%) — post-processing still earns ~1pp on TEST for lv-chordia.

---

## DEV per-track (majmin)

| id | chordia | btc | Δ |
|----|---------|-----|---|
| yellow-submarine | 0.013 | **0.761** | +0.748 |
| come-together | **0.863** | 0.748 | −0.115 |
| youre-my-best-friend | 0.722 | **0.772** | +0.050 |
| another-one-bites-the-dust | **0.308** | 0.280 | −0.028 |
| while-my-guitar-gently-weeps | **0.806** | 0.771 | −0.034 |
| help | **0.665** | 0.630 | −0.035 |
| let-it-be | 0.133 | 0.158 | +0.025 |
| twist-and-shout | **0.926** | 0.922 | −0.004 |
| i-saw-her-standing-there | 0.933 | **0.946** | +0.013 |
| something | **0.744** | 0.740 | −0.004 |
| ticket-to-ride | 0.690 | **0.716** | +0.026 |
| dont-stop-me-now | 0.628 | **0.637** | +0.008 |
| play-the-game | 0.575 | **0.591** | +0.016 |
| cant-buy-me-love | 0.608 | **0.611** | +0.003 |

---

## TEST per-track (majmin)

| id | chordia | btc | Δ |
|----|---------|-----|---|
| hammer-to-fall | 0.316 | **0.708** | +0.392 |
| here-comes-the-sun | 0.881 | **0.893** | +0.012 |
| penny-lane | 0.695 | **0.713** | +0.018 |
| yesterday | **0.860** | 0.827 | −0.033 |
| norwegian-wood | **0.888** | 0.882 | −0.006 |
| back-in-the-ussr | **0.423** | 0.418 | −0.005 |
| somebody-to-love | 0.441 | 0.442 | ~0 |
| crazy-little-thing-called-love | **0.578** | 0.525 | −0.053 |
| seven-seas-of-rhye | **0.579** | 0.500 | −0.079 |

No TEST track regresses majmin by >5pp vs RAW chordia.

---

## Decision gate

**TEST Δ majmin = +2.7pp (>2pp threshold)** → BTC is a **candidate** for default base.

**Recommended next steps (not done yet):**

1. Run **BTC + bypass/light-merge** bake-off (same script extension) — RAW BTC may still lose to post-processed chordia on aggregate TEST.
2. **Do not flip default** until BTC+post-process is measured; lv-chordia remains serving default (`CHORD_MODEL=ensemble`).
3. For Phase 12 fine-tuning: BTC has public training code + pretrained weights — strong fine-tune base even if serving default stays chordia short-term.

**Flag:** `CHORD_MODEL=btc` (alias `CHORD_ML_MODEL=btc`) — wired in `extract_chords_ml`, not default.
