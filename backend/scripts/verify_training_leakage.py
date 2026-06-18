#!/usr/bin/env python3
"""Verify training file lists exclude all 24 gold holdout tracks (Phase 12)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
HOLDOUT = BACKEND / "analysis" / "gold_holdout_v2.json"


def _load_holdout() -> dict:
    return json.loads(HOLDOUT.read_text())


def _matches_holdout(path_str: str, holdout: dict) -> str | None:
    norm = path_str.replace("\\", "/")
    base = Path(norm).name
    for tid in holdout["track_ids"]:
        if tid in norm.replace("_", "-").lower():
            return f"track_id:{tid}"
    for iso in holdout["isophonics_paths"]:
        if iso in norm or norm.endswith(iso):
            return f"isophonics:{iso}"
    for bn in holdout["isophonics_basenames"]:
        if base == bn or base.replace(" ", "_") == bn.replace(" ", "_"):
            return f"basename:{bn}"
    return None


def check_paths(paths: list[str], holdout: dict) -> list[dict]:
    leaks = []
    for p in paths:
        hit = _matches_holdout(p, holdout)
        if hit:
            leaks.append({"path": p, "match": hit})
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold leakage check for Phase 12 training")
    parser.add_argument(
        "paths_file",
        type=Path,
        help="Text file with one audio/lab path per line",
    )
    args = parser.parse_args()

    holdout = _load_holdout()
    lines = [ln.strip() for ln in args.paths_file.read_text().splitlines() if ln.strip()]
    leaks = check_paths(lines, holdout)

    print(f"Checked {len(lines)} paths against {len(holdout['track_ids'])} gold holdout tracks")
    if leaks:
        print(f"LEAKAGE: {len(leaks)} paths match gold holdout:")
        for item in leaks[:20]:
            print(f"  {item['match']} <- {item['path']}")
        if len(leaks) > 20:
            print(f"  ... and {len(leaks) - 20} more")
        return 1

    print("PASS: zero gold overlap detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
