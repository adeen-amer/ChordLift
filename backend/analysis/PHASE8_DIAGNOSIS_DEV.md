# Phase 8 — DEV accuracy leak diagnosis

**Measurement only** — no engine changes.

DEV baseline (v45): majmin **0.4879** | sevenths 0.4212 | seg 0.5905

## Experiment 1 — lv-chordia input signal (raw output, no post-processing)

| Input | majmin | sevenths | seg | n |
|-------|--------|----------|-----|---|
| (a) raw full mix | 0.579 | 0.510 | 0.747 | 14 |
| (b) HPSS chord stem (today) | 0.606 | 0.531 | 0.757 | 14 |
| (c) Demucs other+bass | 0.617 | 0.524 | 0.730 | 14 |

## Experiment 2 — per-stage post-processing (HPSS chordia → pipeline)

| Stage | majmin | sevenths | seg | Δ majmin vs prev |
|-------|--------|----------|-----|------------------|
| (0) raw chordia | 0.606 | 0.531 | 0.757 |  |
| (1) + chroma align | 0.646 | 0.567 | 0.724 | +0.039 |
| (2) + polish | 0.563 | 0.494 | 0.640 | -0.082 |
| (3) + bar finalize | 0.481 | 0.424 | 0.572 | -0.082 |
| (4) final format | 0.481 | 0.424 | 0.572 | +0.000 |

## Experiment 3 — label mapping loss (oracle self-score)

- Unique Harte ref labels: 96
- Quality preserved roundtrip: 28.1%
- Avg majmin lost to mapping alone: **0.0491**
- Avg sevenths lost to mapping alone: **0.0519**

## Gate verdict

- **DEV gap vs chordia raw ceiling:** baseline final 0.488 vs raw chordia 0.606 → pipeline throws away ~0.118 majmin.

- **Input signal:** raw mix 0.579 vs HPSS stem 0.606 (HPSS neutral/helpful; Δ=+0.027).

- **Demucs other+bass:** 0.617 (Δ vs HPSS +0.011).

- **Largest post-process drop:** stage `2_polish` Δ majmin -0.082 (raw→final net -0.125; align net +0.039).

- **Label mapping:** oracle majmin loss 0.0491 (0.0519 sevenths) — ~39% of raw→final gap.

- **Input signal verdict:** HPSS stem does *not* hurt lv-chordia (mix ≤ HPSS on avg). Hypothesis rejected.

- **Phase 10 target:** Post-processing is the main leak — `polish_chord_timeline` (-0.082) and `finalize_bar_timeline` (-0.082). Chroma align helps on average (+0.039); not the primary bug.

