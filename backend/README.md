# Backend eval (mir_eval gold v2)

Gold v2 DEV/TEST baselines and phase docs live here. **Source of truth:** `BASELINE_mir_gold_{DEV,TEST}_v49.json`.

```bash
make eval              # gate vs v49 (local; needs downloads/)
make phase13-v49       # regenerate baselines
make verify-gold-audio # identity gate vs Spotify + lab anchors
make build-gold-bundle # pack 24 approved mp3s → gold_audio_v2.tar.gz (local only)
```

## CI gold audio (no YouTube in runners)

GitHub Actions **does not** download eval audio from YouTube/spotdl. The `gold-eval` workflow fetches a cached
`gold_audio_v2.tar.gz` from the private release tag `gold-audio-v2`, verifies durations against
`tests/fixtures/gold_audio_identity.json`, then runs `eval_gold_mir.py`.

**Bootstrap when the gold set changes (run on your machine):**

```bash
# 1. Re-download if needed (residential IP; requires SPOTIFY_CLIENT_ID/SECRET)
cd backend && source .venv/bin/activate
python scripts/download_gold_spotdl.py
python scripts/verify_gold_audio.py --require-ear

# 2. Build + upload bundle
make build-gold-bundle
gh release create gold-audio-v2 --title "Gold eval audio v2" --notes "CI bundle"  # first time only
gh release upload gold-audio-v2 backend/gold_audio_v2.tar.gz --clobber
```

Push/PR CI runs **unit-tests + frontend** only (`ci.yml`). Gold eval runs nightly / on demand (`gold-eval.yml`).

Legacy stamp/boundary eval harness removed — CI uses `eval_gold_mir.py` only.
