# Phase 11.5 — Audio source sensitivity

## Context

Gold audio filenames say `spotify-*` but **production downloader resolved Spotify → ytsearch → random YouTube**. Mean duration drift vs Isophonics `.lab` ≈ **1.0s** (max 2.4s on `get-back`).

Spotify catalogue = **2009 remaster** — not the 1987 masters Isophonics annotated. True Spotify rip is not automatically “more correct.”

**spotdl** (used here) downloads via **YouTube Music metadata match**, not a Spotify stream — but it fixes bad YT search hits.

---

## Step 1 — Spot-test (3 worst-drift DEV tracks) ✅

Tracks: `come-together` (1.96s), `i-saw-her-standing-there` (1.86s), `help` (1.85s)

| id | drift before → after | chordia majmin before → after | Δ | identity worse? |
|----|----------------------|-------------------------------|---|-----------------|
| come-together | 1.96s → **0.70s** | 0.866 → **0.954** | +8.8pp | no |
| i-saw-her-standing-there | 1.86s → 1.86s | 0.917 → 0.917 | 0 | no |
| help | 1.85s → **1.53s** | 0.670 → **0.963** | **+29.3pp** | no |

BTC bypass: come-together +10.8pp, help +29.3pp, i-saw-her 0pp.

**Decision: PROCEED** — majmin >2pp on 2/3 tracks with improved (or unchanged) alignment; identity did not get worse.

`help` strongly suggests the **old YT search match was the wrong recording**, not benign edge-silence drift.

JSON: `analysis/PHASE11_5_SPOTTEST.json`

---

## Step 2 — Full re-download + v47 re-baseline ✅

### Hardened identity gate

| Check | Threshold | Role |
|-------|-----------|------|
| **Spotify official duration** | ≤1.0s | Independent wrong-version anchor (SpotipyFree `duration_ms`) |
| **Lab end time** | ≤1.99s scrutiny; gross >5s fails | Catches truncated/wrong files; remaster edition drift flagged for ear-check |
| **Manual scrutiny** | lab drift >1s OR v46 majmin <0.50 | No auto ear-approve (`--approve-manual` after A/B) |

Curated YT overrides when spotdl picks a wrong take: `tests/fixtures/gold_youtube_overrides.json`
(`dont-stop-me-now`, `seven-seas-of-rhye`, `yellow-submarine`).

### Gate result

**24/24 pass** Spotify + lab sanity (2025-06-17). All approved in `gold_audio_identity.json`.

### v47 baselines — corrected ruler (NOT comparable to v46 as model improvement)

> v47 = re-measurement on identity-verified audio; majmin gains reflect fixed reference recordings. Engine byte-identical.

| Split | v46 majmin | v47 majmin | Δ | v46 seg | v47 seg |
|-------|------------|------------|---|---------|---------|
| DEV | 66.9% | **76.2%** | **+9.3pp** | 72.0% | 78.9% |
| TEST | 66.4% | **70.7%** | **+4.3pp** | 61.3% | 62.8% |

Largest corrected-track gains: `help` +29.3pp, `cant-buy-me-love` +36.1pp, `back-in-the-ussr` +45.9pp, `ticket-to-ride` +22.5pp.

Per-track diff: `analysis/BASELINE_v46_vs_v47.json`

Expect chordia and BTC to rise **together** on corrected tracks (e.g. `help` +29pp) — no BTC-vs-chordia re-rank.

### Commands

```bash
make phase11-5-v47
# or:
python scripts/download_gold_spotdl.py
python scripts/verify_gold_audio.py --approve-ear --approve-manual all
CHORD_ENGINE=ml CHORD_ENGINE_STRICT=1 python scripts/rebaseline_v47.py
```

---

## Production

`downloader.py`: Spotify links → **spotdl YT-Music match** + Spotify duration verification; falls back to ytsearch on failure.

Run spot-test: `python scripts/phase11_5_audio_spottest.py`
