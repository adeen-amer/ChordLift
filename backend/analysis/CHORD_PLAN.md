# Chord accuracy directive — phase tracker

**Current phase:** cleanup sprint ✅ | serving **v49** (lv-chordia raw + pitch gate) | analyzer **v45**  
**Gold v2 TEST (v49):** majmin **73.0%** | seg **66.0%** — `BASELINE_mir_gold_TEST_v49.json`

---

## Phases 0–5 ✅

## Phase 6 — Productize ✅ complete

**Backend (v44):**
- `safe_paths.py` — block path traversal on `/api/audio/{video_id}` and cache paths
- `analysis_runtime.py` — semaphore (`MAX_CONCURRENT_ANALYSES`) + in-flight dedup
- `cache_eviction.py` — LRU eviction (`CACHE_MAX_ENTRIES`, `CACHE_MAX_BYTES`)
- `model_cache.py` — singleton Demucs + basic-pitch models
- Sanitized client errors (no raw exception text)
- Progress endpoint uses canonical `source_id` from download

**Frontend:**
- rAF loop gated on audio play state (idle when paused)
- Binary-search timeline lookup (`timeline.ts`)
- Audio load error handling + play() rejection catch
- Abort in-flight analysis on new request (AbortController + EventSource close)
- Vitest unit tests for timeline helpers; CI runs `npm test`

**Verify:**
```bash
cd backend && python -m pytest tests/test_phase6_productize.py -q
cd frontend && npm test
make eval
```

---

## Accuracy Campaign v2

**Phase 7 — Diversify gold ✅:** `analysis/GOLD_v2_PROPOSAL.md`, identity gate, DEV/TEST baselines  
**Phase 11.5 — Audio identity ✅:** spotdl re-download, hardened gate, **v47 corrected-ruler baselines** (`BASELINE_v47_README.md`)  
**Phase 8 — Diagnose leaks ✅:** `analysis/PHASE8_DIAGNOSIS_DEV.md`  
**Phase 10 — Lever 1 bypass ✅ SHIPPED:** `analysis/PHASE10_LEVER1.md` — default `CHORD_ML_POSTPROCESS=bypass` (v46)  
**Phase 10 — Lever 2 slow/sparse merge ✅:** `analysis/PHASE10_LEVER2.md` — `let-it-be` recovered (+36pp vs bypass)  
**Phase 10 — Lever 3 label mapping ✅:** Harte roundtrip fixes  
**Phase 11 — COMPLETE ✅:** `analysis/PHASE11_COMPLETE.md` — serving gate HOLD (keep chordia)  
**Phase 11.6 — BTC bake-off on v47 ruler ✅:** `analysis/PHASE11_6_BTC_V47.md` — chordia sufficient; no BTC fine-tune  
**Phase 12.5 — Post-process ablation ✅:** `analysis/PHASE12_5_POSTPROCESS_ABLATION.md` — **v48** pitch-only default (`CHORD_ML_POSTPROCESS=raw`); bypass merge retired from serving  
**Phase 12 — Fine-tune (revised):** chordia fine-tune or skip — see `analysis/PHASE12_FINETUNE.md`  
**Phase 13 — Presentation ✅:** `analysis/PHASE13.md` — beat-sync display, key constrain, capo, confidence tiers, user corrections; pitch reliability gate; disagreement CI guard
