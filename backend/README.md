# Backend eval (mir_eval gold v2)

Gold v2 DEV/TEST baselines and phase docs live here. **Source of truth:** `BASELINE_mir_gold_{DEV,TEST}_v49.json`.

```bash
make eval              # gate vs v49
make phase13-v49       # regenerate baselines
```

Legacy stamp/boundary eval harness removed — CI uses `eval_gold_mir.py` only.
