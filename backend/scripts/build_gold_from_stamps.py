#!/usr/bin/env python3
"""Promote phase-aligned stamp timelines into chord_gold_labels.json (v1, pending ear review)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval_fixture_utils import DEFAULT_GOLD_LABELS, DEFAULT_STAMP_REFS, load_all_overlays  # noqa: E402


def build_gold_entry(tid: str, stamp: dict, overlay: dict) -> dict:
    start = overlay.get("boundary_eval_start")
    end = overlay.get("boundary_eval_end", stamp.get("duration_sec"))
    segments = []
    for seg in stamp.get("reference_timeline", []):
        t0, t1 = float(seg["time"]), float(seg["end_time"])
        if end is not None and t0 >= end:
            continue
        if end is not None and t1 <= (start or 0):
            continue
        segments.append({
            "time": round(max(t0, start or t0), 3),
            "end_time": round(min(t1, end) if end else t1, 3),
            "chord": seg["chord"],
        })
    if segments and start is not None and segments[0]["time"] > start:
        segments[0]["time"] = round(start, 3)
    if segments and end is not None:
        segments[-1]["end_time"] = round(end, 3)
    return {
        "title": stamp.get("title"),
        "source": "progression-aligned-v1",
        "review_status": "pending_human",
        "progression": stamp.get("progression"),
        "eval_window": {"start": start, "end": end},
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="let-it-be,get-lucky")
    parser.add_argument("--out", type=Path, default=DEFAULT_GOLD_LABELS)
    parser.add_argument("--merge", action="store_true", help="Merge into existing gold file")
    args = parser.parse_args()

    stamps = json.loads(DEFAULT_STAMP_REFS.read_text()).get("tracks", {})
    overlays = load_all_overlays()
    wanted = {x.strip() for x in args.ids.split(",") if x.strip()}

    existing = {}
    if args.merge and args.out.exists():
        existing = json.loads(args.out.read_text()).get("tracks", {})

    for tid in wanted:
        if tid not in stamps:
            print(f"SKIP {tid}: no stamp")
            continue
        existing[tid] = build_gold_entry(tid, stamps[tid], overlays.get(tid, {}))
        print(f"  {tid}: {len(existing[tid]['segments'])} segments")

    payload = {
        "description": (
            "Reference timelines for eval. Override chord_stamp_refs when present. "
            "review_status=pending_human means not yet ear-verified."
        ),
        "tracks": existing,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
