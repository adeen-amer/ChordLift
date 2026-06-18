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
```

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `CHORD_ENGINE` | `ml` | `ml` or `classic` |
| `CHORD_MODEL` | `chordia` | `ensemble` accepted as deprecated alias |
| `CHORD_PITCH_CORRECT` | `1` | Reliability-gated ±1 semitone detune fix |
| `CHORD_STEM_MODE` | `auto` | `auto` \| `hpss` \| `demucs` |

Cached analyses invalidate when `ANALYZER_VERSION` or engine flags change.
