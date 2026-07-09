# ML chord engine — lv-chordia (raw segments + pitch reliability gate)

**Production:** `CHORD_ENGINE=ml` with `CHORD_MODEL=chordia` (default).

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt
```

Check readiness: `GET /api/health` → `ml_ready: true`.

## Eval

```bash
make eval          # gold v2 DEV+TEST vs v49 baseline
make phase13-v49   # regenerate v49 baselines
make phase-v50     # regenerate v50 baselines (decode-both pitch selection)
```

See `analysis/BASELINE_v50_README.md` for the v50 decode-both pitch-selection results (replaces the v49 reliability gate).

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `CHORD_ENGINE` | `ml` | `ml` or `classic` |
| `CHORD_MODEL` | `chordia` | `ensemble` accepted as deprecated alias |
| `CHORD_PITCH_CORRECT` | `1` | `0` maps to `CHORD_PITCH_SELECT=off` on the ML path |
| `CHORD_PITCH_SELECT` | `confidence` | `confidence` \| `tta` \| `off` — v50 decode-both branch selection (raw vs pitch-corrected chord stem), replaces the v49 reliability gate |
| `CHORD_PITCH_CONF_MARGIN` | `0.0` | Confidence margin the corrected branch must beat to win selection |
| `CHORD_PITCH_MAX_CANDIDATE_SHIFT` | `0.40` | Max \|shift\| (semitones) considered for correction; larger candidates are rejected as tuning-estimator error |
| `CHORD_STEM_MODE` | `auto` | `auto` \| `hpss` \| `demucs` |

Cached analyses invalidate when `ANALYZER_VERSION` or engine flags change.
