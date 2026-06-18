# Baseline v47 — identity-verified audio (NOT comparable to v46)

v47 = re-measurement on YT-Music metadata-matched audio with hardened identity gate.
Majmin gains reflect **fixed reference recordings**, not model changes (byte-identical engine).

## Aggregate majmin / seg

| Split | v46 majmin | v47 majmin | Δ | v46 seg | v47 seg |
|-------|------------|------------|---|---------|---------|
| DEV | 0.669 | 0.762 | +0.092 | 0.720 | 0.789 |
| TEST | 0.664 | 0.707 | +0.043 | 0.613 | 0.628 |

## Largest v47 gains (corrected ruler)

### DEV

| id | v46 | v47 | Δ majmin |
|----|-----|-----|----------|
| cant-buy-me-love | 0.576 | 0.937 | +0.361 |
| help | 0.670 | 0.963 | +0.293 |
| ticket-to-ride | 0.735 | 0.960 | +0.225 |
| dont-stop-me-now | 0.644 | 0.787 | +0.142 |
| let-it-be | 0.495 | 0.585 | +0.090 |
| come-together | 0.866 | 0.954 | +0.088 |
| something | 0.722 | 0.780 | +0.058 |
| youre-my-best-friend | 0.709 | 0.748 | +0.039 |

### TEST

| id | v46 | v47 | Δ majmin |
|----|-----|-----|----------|
| back-in-the-ussr | 0.441 | 0.900 | +0.459 |
| penny-lane | 0.683 | 0.830 | +0.147 |
| crazy-little-thing-called-love | 0.555 | 0.618 | +0.063 |
| seven-seas-of-rhye | 0.609 | 0.582 | -0.027 |

