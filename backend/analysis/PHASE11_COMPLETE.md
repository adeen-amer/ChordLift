# Phase 11 — COMPLETE

## Step 1 — RAW bake-off ✅
See `PHASE11_BTC_BAKEOFF.md`. TEST RAW BTC +2.7pp vs RAW chordia (65.6% vs 62.9%).

## Step 2 — Serving comparison (BTC+bypass vs v46) ✅

**Gate:** BTC+bypass must beat v46 TEST **66.4%** by >2pp (>68.4%).

| Split | chordia+bypass | BTC+bypass | v46 |
|-------|----------------|------------|-----|
| DEV | **66.9%** | 62.6% | 66.9% |
| TEST | **66.4%** | **56.0%** | 66.4% |

**GATE HOLD** — keep `CHORD_MODEL=ensemble` serving default. No further serving-default debate this phase.

Note: bypass/light-merge is tuned for lv-chordia timelines; applying it to BTC **hurts** TEST (65.6% RAW → 56.0% post). Fine-tuned BTC will need its own post-process eval in Phase 12.

## yellow-submarine RAW chordia=0.013 — diagnosed ✅

RAW bake-off harness skips **pitch correction** (and all bypass post-process).

| Stage | majmin |
|-------|--------|
| raw, no pitch correct | **0.013** |
| raw + pitch correct | **0.748** |
| full bypass | 0.748 |

DEV RAW margin vs BTC is partly artifact. **Trust TEST (+2.7pp RAW).** Do not re-tune off yellow-submarine.

## Locked decision

**BTC is the Phase 12 fine-tuning base** regardless of serving gate.

Run: `make phase11-serving` | `scripts/diagnose_yellow_submarine.py`
