# Baseline v48 — pitch-only serving on v47 audio

v48 = `CHORD_ML_POSTPROCESS=raw` (no chroma-align / light-merge) + pitch correction.
Phase 12.5 ablation on identity-verified v47 audio — see `PHASE12_5_POSTPROCESS_ABLATION.md`.

## Aggregate majmin / seg

| Split | v47 majmin | v48 majmin | Δ | v47 seg | v48 seg |
|-------|------------|------------|---|---------|---------|
| DEV | 0.762 | 0.787 | +0.025 | 0.789 | 0.820 |
| TEST | 0.707 | 0.743 | +0.036 | 0.628 | 0.662 |

## Largest v48 gains vs v47 (bypass serving)

### DEV

| id | v47 | v48 | Δ majmin |
|----|-----|-----|----------|
| let-it-be | 0.585 | 0.874 | +0.289 |
| youre-my-best-friend | 0.748 | 0.795 | +0.047 |

### TEST

| id | v47 | v48 | Δ majmin |
|----|-----|-----|----------|
| here-comes-the-sun | 0.611 | 0.850 | +0.239 |
| penny-lane | 0.830 | 0.912 | +0.082 |
| norwegian-wood | 0.826 | 0.888 | +0.062 |

