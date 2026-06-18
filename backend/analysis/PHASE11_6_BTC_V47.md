# Phase 11.6 — BTC vs chordia on v47 corrected ruler

Re-measured on identity-verified v47 audio. Stale Phase 11 used v46 audio with wrong recordings.

## RAW majmin / seg (no post-processing)

| Split | v47 chordia | v47 BTC | Δ BTC−chordia | stale v46 Δ |
|-------|-------------|---------|---------------|-------------|
| DEV | 0.787 | 0.767 | -0.019 | +0.048 (n=14) |
| TEST | 0.743 | 0.707 | -0.035 | +0.028 (n=9) |

Stale v46 TEST RAW: chordia 0.629 / BTC 0.656 (Δ +0.028)

## bypass (current serving path, pitch-correct)

| Split | chordia+bypass | BTC+bypass | v47 serving | stale v46 Δ |
|-------|----------------|------------|-------------|-------------|
| DEV | 0.762 | 0.731 | 0.762 | -0.043 |
| TEST | 0.707 | 0.619 | 0.707 | -0.105 |

## Decision (fine-tune base only — serving default unchanged)

RAW chordia leads BTC on v47 TEST — chordia is sufficient; fine-tune chordia or skip fine-tune. Do not adopt BTC out of momentum.

JSON: `analysis/PHASE11_6_BTC_V47.json`
