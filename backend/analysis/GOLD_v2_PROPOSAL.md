# Phase 7 — Gold v2 catalog (APPROVED)

**Status:** APPROVED — labs extracted, audio identity gate committed, baselines on DEV/TEST.

**Scope limit:** This is a **classic-rock ruler** (1960s–80s Beatles + Queen pop/rock only). Scores do **not** generalize to jazz, metal, hip-hop, or post-2000 production.

**Prior baseline (v44, 8 Beatles):** root 58.8 | majmin 51.0 | sevenths 46.8 | seg 58.8

---

## Label provenance

| Tier | Source | Confidence |
|------|--------|------------|
| **tier_a** | Isophonics Beatles — Harte v1.2 syntax; community verified | High |
| **tier_b** | Isophonics Queen — Harte syntax; single annotator review | Moderate |

Carole King (Tier C) **excluded**. Pathological Queen tracks dropped: `bohemian-rhapsody`, `we-will-rock-you`, `bicycle-race`.

**Killer Queen substitute:** Isophonics has no Harte chord `.lab` for Killer Queen (segmentation only). Catalog uses **`seven-seas-of-rhye`** instead (seventh-rich, beat-stable).

---

## Stratification axes

| Axis | Meaning | Full-set target |
|------|---------|-----------------|
| **A** | Fast / repetitive changes | ≥4 |
| **B** | Minor-key or minor-heavy | ≥5 |
| **C** | Seventh / extended-heavy | ≥6 |
| **D** | Slow / sparse harmony | ≥2 |
| **E** | Medium complexity baseline | remainder |

---

## Full catalog — strata membership per track

| id | artist | label_tier | strata |
|----|--------|------------|--------|
| let-it-be | Beatles | tier_a | C, E |
| yellow-submarine | Beatles | tier_a | D, E |
| come-together | Beatles | tier_a | A, B, C |
| yesterday | Beatles | tier_a | D |
| help | Beatles | tier_a | B |
| twist-and-shout | Beatles | tier_a | A |
| something | Beatles | tier_a | C, E |
| ticket-to-ride | Beatles | tier_a | E |
| i-saw-her-standing-there | Beatles | tier_a | A, E |
| cant-buy-me-love | Beatles | tier_a | A, E |
| back-in-the-ussr | Beatles | tier_a | A, E |
| get-back | Beatles | tier_a | A, E |
| norwegian-wood | Beatles | tier_a | B |
| while-my-guitar-gently-weeps | Beatles | tier_a | B, C |
| penny-lane | Beatles | tier_a | C |
| here-comes-the-sun | Beatles | tier_a | E |
| another-one-bites-the-dust | Queen | tier_b | A, B |
| dont-stop-me-now | Queen | tier_b | E |
| crazy-little-thing-called-love | Queen | tier_b | E |
| somebody-to-love | Queen | tier_b | B, C |
| play-the-game | Queen | tier_b | C, E |
| youre-my-best-friend | Queen | tier_b | C, E |
| seven-seas-of-rhye | Queen | tier_b | C, E |
| hammer-to-fall | Queen | tier_b | E |

**24 tracks total:** 16 Beatles (tier_a) + 8 Queen (tier_b).

---

## DEV / TEST split (~58/42)

Report **both** splits for every accuracy change.

### DEV — 14 tracks

twist-and-shout, i-saw-her-standing-there, cant-buy-me-love, another-one-bites-the-dust, come-together, help, while-my-guitar-gently-weeps, let-it-be, something, play-the-game, youre-my-best-friend, ticket-to-ride, dont-stop-me-now, yellow-submarine

### TEST — 10 tracks

back-in-the-ussr, get-back, yesterday, norwegian-wood, penny-lane, here-comes-the-sun, crazy-little-thing-called-love, somebody-to-love, seven-seas-of-rhye, hammer-to-fall

---

## Audio identity gate (mandatory before baselining)

Script: `scripts/verify_gold_audio.py`  
Manifest: `tests/fixtures/gold_audio_identity.json`

1. **Auto-reject:** `|audio_duration − lab_end| > 2s`
2. **Ear-check:** 3 timestamps per track in manifest (`ear_check_sec`); `ear_check_status=approved` required for eval (`--require-audio-identity`)

Audio: Spotify 2009/2011 remasters → YouTube (same honesty caveat as v1 — not Isophonics CD mastering; no ±offset search).

---

## Commands

```bash
# Extract labs (Beatles + Queen archives)
python scripts/extract_isophonics_labs.py

# Download missing audio
python scripts/download_eval_tracks.py

# Identity gate → updates gold_audio_identity.json
python scripts/verify_gold_audio.py
python scripts/verify_gold_audio.py --require-ear   # fail if ear pending

# Baselines (identity-approved tracks only)
python eval_gold_mir.py --no-cache --split dev --require-audio-identity \
  --write analysis/BASELINE_mir_gold_DEV_v45.json
python eval_gold_mir.py --no-cache --split test --require-audio-identity \
  --write analysis/BASELINE_mir_gold_TEST_v45.json
```

2-track smoke (`let-it-be,yellow-submarine`) remains CI fast gate only.

---

## Phase 7 baselines (v44, identity-approved, 2025-06)

| Split | n | root | majmin | sevenths | seg | Notes |
|-------|---|------|--------|----------|-----|-------|
| **DEV** | 14 | 54.6% | 48.8% | 42.1% | 59.1% | All DEV tracks pass identity gate |
| **TEST** | 9 | 50.0% | 43.6% | 41.3% | 46.2% | `get-back` excluded (Δ=2.4s > 2s gate) |

Files: `analysis/BASELINE_mir_gold_DEV_v45.json`, `analysis/BASELINE_mir_gold_TEST_v45.json`
