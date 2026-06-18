# Phase 10 Lever 1 — bypass post-processing (SHIPPED)

**Default (v46):** `CHORD_ML_POSTPROCESS=bypass`  
**Revert:** `CHORD_ML_POSTPROCESS=full` (v45 polish + bar collapse path)

**Bypass path:** raw lv-chordia → merge adjacent → chroma align → key format → output  
**Skipped:** `polish_chord_timeline`, `finalize_bar_timeline`, beat-snap, short-segment merge, power-pattern

Lever 2 adds slow/sparse within-bar merge on top of bypass — see `PHASE10_LEVER2.md`.

---

## DEV (n=14) — v45 full vs bypass (Lever 1 compare)

| | majmin | seg |
|---|--------|-----|
| **full (v45)** | 48.8% | 59.1% |
| **bypass** | **63.8%** | **73.4%** |
| **Δ** | **+15.0pp** | **+14.3pp** |

### Per-track

| id | majmin full→bypass | Δ | seg full→bypass | Δ | est segs |
|----|-------------------|---|-----------------|---|----------|
| twist-and-shout | 0.575→0.927 | +0.353 | 0.275→0.882 | +0.607 | 19→103 |
| while-my-guitar-gently-weeps | 0.404→0.817 | +0.413 | 0.647→0.782 | +0.135 | 94→114 |
| help | 0.394→0.670 | +0.276 | 0.562→0.719 | +0.157 | 30→51 |
| i-saw-her-standing-there | 0.647→0.917 | +0.270 | 0.764→0.885 | +0.121 | 52→54 |
| dont-stop-me-now | 0.418→0.644 | +0.226 | 0.515→0.658 | +0.143 | 55→93 |
| come-together | 0.632→0.866 | +0.234 | 0.838→0.905 | +0.066 | 23→27 |
| something | 0.518→0.708 | +0.190 | 0.496→0.626 | +0.130 | 47→70 |
| youre-my-best-friend | 0.472→0.653 | +0.181 | 0.406→0.616 | +0.210 | 38→68 |
| cant-buy-me-love | 0.440→0.576 | +0.136 | 0.555→0.697 | +0.142 | 28→46 |
| yellow-submarine | 0.620→0.748 | +0.128 | 0.677→0.804 | +0.127 | 47→67 |
| play-the-game | 0.517→0.533 | +0.015 | 0.719→0.610 | −0.109 | 80→98 |
| ticket-to-ride | 0.744→0.735 | −0.009 | 0.833→0.758 | −0.075 | 48→48 |
| another-one-bites-the-dust | 0.003→0.002 | ~0 | 0.406→0.513 | +0.108 | 21→40 |
| **let-it-be** | **0.447→0.132** | **−0.314** | 0.576→0.815 | +0.238 | 107→130 |

**Known DEV regression (Lever 1 bypass only):** `let-it-be` — bar finalize helped this track in Phase 8; bypass removed that gain. **Recovered in Lever 2** (v46: 0.495 majmin).

---

## TEST (n=9) — v45 full vs bypass — **GATE PASS**

| | majmin | seg |
|---|--------|-----|
| **full (v45)** | 43.6% | 46.2% |
| **bypass** | **66.3%** | **61.3%** |
| **Δ** | **+22.7pp** | **+15.0pp** |

No TEST track regressed majmin by >5pp.

---

## Verdict

- **Lever 1 gate: PASS** on TEST (majmin + seg up; no majmin regression >5pp).
- **Default flipped to bypass** in v46; `full` remains selectable.
- Lever 2 addresses `let-it-be`; see `PHASE10_LEVER2.md`.

**Run:** `make phase10-lever1` (full vs bypass compare) | `make eval-gold-dev`
