# ChordLift backend

Requires **Python 3.11+** (matches Docker). Use the setup script:

```bash
cd backend
chmod +x scripts/setup_venv.sh
./scripts/setup_venv.sh
source .venv/bin/activate
```

For the ML chord engine, see [ML_SETUP.md](ML_SETUP.md).

## Eval / regression tests

```bash
# Download fixture audio (Spotify → YouTube)
python scripts/download_eval_tracks.py

# Run full suite (ML engine, all tracks required)
CHORD_ENGINE=ml python eval_chords.py --require-all --write-results eval-results.json

# Separate gates (also run in CI)
python eval_key_labels.py eval-results.json --min-rate 0.50
python eval_symbol_recall.py eval-results.json --min-rate 0.42
python eval_boundary.py eval-results.json --min-rate 0.20
python eval_timeline.py eval-results.json --min-rate 0.12

# Beatles subset vs Hooktheory
CHORD_ENGINE=ml python eval_beatles.py
```

**What eval checks**

| Metric | Meaning |
|--------|---------|
| Root recall (CI) | Predicted roots appear in fixture set — not per-segment correctness |
| Key labels (CI) | Global key vs Hooktheory |
| Symbol recall | Each segment’s chord is in the expected vocabulary |
| Boundary score | Change points vs beat-synced progression loop |
| Timeline score | Duration-weighted chord agreement over the whole song |
| Switch MAE | Mean timing error of matched chord changes (seconds) |

**Boundary / timeline eval** — tracks with a known `progression` get timestamped **chord stamps** (`reference_changes` + `reference_timeline`) for scoring when the chord engine should have switched. Stamps are precomputed from eval audio via phase-aligned chroma + flux snapping and stored in `tests/fixtures/chord_stamp_refs.json`:

```bash
# Regenerate stamps after progression or alignment changes (requires fixture audio)
python scripts/build_chord_stamp_eval.py

# Inspect alignment for a few tracks
python scripts/inspect_boundary_alignment.py --ids let-it-be,get-lucky

# Force live alignment during eval (ignore cached stamps)
CHORD_ENGINE=ml python eval_chords.py --live-boundaries --ids let-it-be
```

Optional `boundary_eval_start` / `boundary_eval_end` in `eval_enrichment.json` limit scoring to a section (e.g. skip intro). Use `boundary_method: beat_loop` on a track to force the legacy beat-index fallback when building stamps.

CI runs the same checks via `.github/workflows/chord-eval.yml`.

## Chord engine improve loop

Benchmark a small anchor set against **complete reference stamps**, inspect failures, patch the engine, repeat:

```bash
# 1. Reference stamps (phase-aligned from progression + audio)
python scripts/build_chord_stamp_eval.py

# 2. Provisional gold labels (ear-review pending)
python scripts/build_gold_from_stamps.py --ids let-it-be,get-lucky --merge

# 3. Run engine vs stamps — per-track diff + split summary
python scripts/analyze_chord_engine.py --split all --write analysis/chord_engine_report.json
python scripts/analyze_chord_engine.py --split held_out   # never used for tuning

# 4. After analyzer changes, re-run analysis then full eval
CHORD_ENGINE=ml python eval_chords.py --require-all --write-results eval-results.json
```

**Eval splits**
- `chord_benchmark.json` — 10 anchor tracks (diagnose failures)
- `chord_held_out.json` — 9 tracks (report only; do not tune on these)
- `chord_gold_labels.json` — reference timelines (override stamps; ear-verify when possible)
- Everything else — tune corpus

**What the analysis report shows:**
- Timeline / boundary / symbol scores
- Time spent with wrong **root** vs wrong **quality** (maj/min, 7 vs triad)
- Top mismatch windows (`pred=Am expected=A @ 42s`)
- Missed vs spurious chord **changes** (timing/segmentation issues)

Generated stamps come from Hooktheory progressions aligned to eval audio. Replace with hand labels in `chord_gold_labels.json` as you verify complete songs.

## Lyrics

During analysis, ChordLift fetches synced lyrics from [LRCLIB](https://lrclib.net) (with YouTube caption fallback) and maps each line into the chord segment whose time range contains it. Each chord card shows the lyrics for that bar; the active card highlights in sync with playback.

Re-analyze a song to attach lyrics to older cached results that predate this feature.
