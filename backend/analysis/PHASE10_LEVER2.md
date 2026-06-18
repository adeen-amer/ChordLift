# Phase 10 Lever 2 — light min-duration merge (slow/sparse archetype)

**Default path:** `CHORD_ML_POSTPROCESS=bypass` (v46)  
**Lever 2 add-on:** within-bar chroma merge + same-chord flicker removal, gated by slow/sparse detector

---

## Problem

`let-it-be` regressed **−31pp majmin** on pure bypass (Lever 1): raw chordia over-segments sustained ballad chords; the old strict bar-collapse fixed that track but destroyed fast-rock timelines (e.g. `twist-and-shout`).

## Solution

For **slow/sparse archetype only** (tuned on `let-it-be`, `yellow-submarine`):

1. `merge_segments_within_bar` with chroma scoring (NOT `collapse_timeline_to_bars`)
2. Same-chord / sandwich flicker merge (`CHORD_LIGHT_MERGE_MIN_SEC=1.0`)

**Archetype gate** (`_needs_light_duration_merge`):

- `segments/min ≥ 30`, median hold `< 2.0s`
- `0.85 ≤ segments/bars ≤ 1.30` (excludes fast rock like `twist-and-shout` at ~1.34)
- `beat_duration ≥ 0.40`

`full` post-process remains available via `CHORD_ML_POSTPROCESS=full`.

---

## Gate vs Lever 1 bypass baseline

| Split | bypass majmin / seg | v46 majmin / seg | Δ majmin |
|-------|---------------------|------------------|----------|
| **DEV (14)** | 63.8% / 73.4% | **66.9% / 72.0%** | **+3.1pp** |
| **TEST (9)** | 66.3% / 61.3% | **66.4% / 61.3%** | **+0.1pp** |

### Key per-track (DEV)

| id | bypass majmin | v46 majmin | Δ |
|----|---------------|------------|---|
| **let-it-be** | 0.132 | **0.495** | **+36.3pp** |
| twist-and-shout | 0.927 | 0.927 | 0 |
| yellow-submarine | 0.748 | 0.748 | 0 |

No track regressed majmin by >5pp on DEV or TEST.

**Lever 2 gate: PASS**

---

## Baselines

- `analysis/BASELINE_mir_gold_DEV_v46.json` — majmin **66.9%**, seg **72.0%**
- `analysis/BASELINE_mir_gold_TEST_v46.json` — majmin **66.4%**, seg **61.3%**

**Run:** `make eval-gold-dev` / `make eval-gold-test` (bypass is default)
