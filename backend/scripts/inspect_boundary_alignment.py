#!/usr/bin/env python3
"""
Inspect phase-aligned boundary references for eval tracks (debug / tuning).

Computes reference changes from audio + progression at runtime — nothing is written
to fixtures. Use this to check alignment scores and phase/skip before adjusting
progression params in eval_enrichment.json.

Usage:
  python scripts/inspect_boundary_alignment.py --ids let-it-be,get-lucky
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval_chord_utils import resolve_boundary_references  # noqa: E402
from eval_fixture_utils import load_all_overlays, merge_track_enrichment  # noqa: E402

FIXTURES = BACKEND / "tests" / "fixtures" / "chord_refs.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dynamic boundary alignment")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--ids", type=str, default="let-it-be,get-lucky,baba-oriley")
    args = parser.parse_args()

    tracks = json.loads(args.fixtures.read_text())["tracks"]
    overlays = load_all_overlays()
    wanted = {x.strip() for x in args.ids.split(",") if x.strip()}

    for track in tracks:
        if track["id"] not in wanted:
            continue
        merged = merge_track_enrichment(track, overlays)
        if not merged.get("progression"):
            print(f"{merged['id']}: no progression")
            continue
        audio_path = BACKEND / merged["file"]
        if not audio_path.exists():
            print(f"{merged['id']}: missing {audio_path}")
            continue
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        changes, timeline, method = resolve_boundary_references(y, sr, merged)
        print(
            f"{merged['id']}: method={method} changes={len(changes)} "
            f"timeline={len(timeline)} "
            f"first={changes[0] if changes else None} "
            f"window={merged.get('boundary_eval_start')}–{merged.get('boundary_eval_end')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
