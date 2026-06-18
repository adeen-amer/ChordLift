# Eval fixtures

Reference data for chord-engine regression tests. Audio files are **not** committed — download with:

```bash
cd backend
python scripts/download_eval_tracks.py
```

## Splits

| File | Role |
|------|------|
| `chord_refs.json` | Full corpus (~37 tracks) |
| `chord_benchmark.json` | 10 anchor tracks for diagnose/improve |
| `chord_held_out.json` | 9 tracks — report only, do not tune thresholds on these |
| `chord_gold_labels.json` | Provisional progression-aligned timelines (superseded for metrics by Isophonics gold) |
| `gold_mir_tracks.json` + `gold/lab/*.lab` | **Phase 1 honest eval** — Isophonics Harte v1.2 (8 Beatles, legacy) |
| `gold_mir_tracks_v2.json` + `gold_split_v2.json` | **Phase 7 gold v2** — 24-track classic-rock ruler (Beatles tier_a + Queen tier_b) |
| `gold_audio_identity.json` | Mandatory audio identity gate (duration ±2s + ear-check timestamps) |
| `chord_stamp_refs.json` | Phase-aligned boundary/timeline stamps |
| `eval_enrichment.json` | Per-track windows and progression overrides |

## Gold labels — human verification required

Entries in `chord_gold_labels.json` with `review_status: pending_human` are **provisional** (Hooktheory/progression-aligned). They are useful for development but **not ground truth** until ear-checked.

### How to verify (Let It Be, Get Lucky, etc.)

1. Open the eval MP3 from `backend/downloads/` in a DAW or editor with a chord chart.
2. Compare each segment `time` / `chord` against what you hear at that moment.
3. Adjust segments in `chord_gold_labels.json` or edit stamps and promote:
   ```bash
   python scripts/build_gold_from_stamps.py --ids let-it-be,get-lucky --merge
   ```
4. Set `review_status` to `verified` and add `verified_by` / `verified_at` if desired.

Until verified, treat gold timeline metrics as **directional only**.
