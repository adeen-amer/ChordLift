# Phase 13 — Perceived accuracy / presentation

**Goal:** Make a ~74% engine *feel* like high-80s without model changes.

**Analyzer v45** — presentation layer + pitch reliability gate (model frozen at v48 path).

---

## A. Accuracy close-out ✅

| Item | Status |
|------|--------|
| Per-track pitch-correction gate (`pitch_utils._estimate_reliable`) | ✅ skip when harmonic/full-mix disagree or key-fit doesn't improve |
| Delete dead ML post-processing (chroma-align, light-merge, polish, strict bar_decode on ML path) | ✅ `chord_engine_ml.py` raw-only; legacy `bypass`/`full` env → raw |
| Cross-model disagreement CI guard | ✅ `scripts/verify_model_disagreement.py` + `tests/test_verify_model_disagreement.py` |

**Pitch gate intent:** recover `another-one-bites-the-dust` (−11pp from blind pitch) while keeping most of aggregate pitch gain.

**v49 WCSR (pitch gate):** TEST majmin **73.0%** (v48 74.3%, raw 71.0%) | DEV **76.1%** (v48 78.7%) | `another-one-bites-the-dust` **12.8%** (v48 1.4%)

---

## B. Presentation (render-time) ✅

`presentation_timeline.py` — wired in `analyze_audio()`:

1. **Beat-synced display** — snap chord boundaries to beat grid (`pipeline.beat_times`)
2. **Key-constrained vocabulary** — low/medium-confidence out-of-key chords → nearest diatonic (display only)
3. **Capo / tuning** — `capo` from `pitch_correction_semitones`; transpose chord names for open shapes
4. **Confidence tiers** — `low` / `medium` / `high` shading in UI
5. **User corrections** — double-click chord → localStorage (`chordlift-corrections:{video_id}`)

API fields: `timeline` (display), `model_timeline`, `beats`, `capo`, `presentation`.

Frontend A/B toggle: **Synced (recommended)** vs **Raw model** — for perceived-correctness study.

---

## GATE — blind A/B perceived correctness

**Not WCSR.** Protocol:

1. Randomize order: synced view vs raw `model_timeline` (same audio, same session).
2. Ask after each: *"How trustworthy/usable are these chords?"* (1–5 Likert).
3. Success: synced mean ≥ raw + 0.5 on Likert, n≥10 listeners.
4. Log `presentation` field + user corrections as optional telemetry.

---

## Commands

```bash
make test
make verify-disagreement          # CI guard
make verify-disagreement-baseline # refresh baseline (slow, ML models)
make eval-gold-test               # WCSR unchanged (presentation is display-only)
```

---

## Serving path (unchanged from v48)

- `CHORD_ENGINE=ml`
- `CHORD_ML_POSTPROCESS=raw` (default)
- `CHORD_PITCH_CORRECT=1` with reliability gate

Classic engine still uses `polish_chord_timeline` / `bar_decode` when `CHORD_ENGINE=classic`.
