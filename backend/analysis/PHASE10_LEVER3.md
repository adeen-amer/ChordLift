# Phase 10 Lever 3 — Harte label mapping

**Files:** `chord_label_utils.py` (`harte_label_to_internal`, `internal_to_harte_label`), `eval_mir_utils.chordlift_to_harte`

## Problem (Phase 8)

~**4.9pp majmin** oracle loss from Harte ↔ internal mapping: slash bass (`A:min/b7`), parenthetical extensions (`E:min7(4)`), `maj6`, bare roots (`C`), `hdim7`, etc.

## Fix

- Strip `/bass` and `(extensions)` before quality parse
- Map bare roots and slash labels to triad majors
- Round-trip `maj6`, `min9`, `hdim7`, `sus2/4`, etc. via shared `internal_to_harte_label`

## Validation

Unit tests in `tests/test_chord_label_utils.py` (roundtrip on common symbols).  
Gold lab label inventory: **100%** loose roundtrip on 2453 ref labels.

v46 baselines (Lever 2 + mapping) are the authoritative DEV/TEST scores — mapping gain is embedded in the v46 numbers above bypass+merge.
