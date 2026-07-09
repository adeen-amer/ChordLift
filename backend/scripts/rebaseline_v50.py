#!/usr/bin/env python3
"""v50 — decode-both pitch selection rebaseline + diff vs v49.

Examples:
  .venv/bin/python scripts/rebaseline_v50.py --split dev --select confidence --tag _conf
  .venv/bin/python scripts/rebaseline_v50.py --split dev --select tta --tag _tta
  .venv/bin/python scripts/rebaseline_v50.py --split dev --dict extended --tag _dict_extended
  .venv/bin/python scripts/rebaseline_v50.py --split both --select confidence --margin 0.005 --tag ""
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ANALYSIS = BACKEND / "analysis"
V49 = {
    "dev": ANALYSIS / "BASELINE_mir_gold_DEV_v49.json",
    "test": ANALYSIS / "BASELINE_mir_gold_TEST_v49.json",
}
GUARD_TRACK = "another-one-bites-the-dust"
GUARD_MIN = 0.128


def _run_eval(split: str, out_path: Path, args) -> dict:
    os.environ["CHORD_ENGINE"] = "ml"
    os.environ["CHORD_ENGINE_STRICT"] = "1"
    os.environ["CHORD_ML_POSTPROCESS"] = "raw"
    os.environ["CHORD_ML_MODEL"] = "chordia"
    os.environ["CHORD_PITCH_SELECT"] = args.select
    os.environ["CHORD_PITCH_CONF_MARGIN"] = str(args.margin)
    os.environ["CHORD_CHORDIA_DICT"] = args.dict
    argv = [
        "eval_gold_mir.py", "--no-cache", "--split", split,
        "--require-audio-identity", "--write", str(out_path),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        import eval_gold_mir
        rc = eval_gold_mir.main()
        if rc != 0:
            raise RuntimeError(f"eval_gold_mir.py --split {split} exited {rc}")
    finally:
        sys.argv = old_argv
    return json.loads(out_path.read_text())


def _diff_split(name: str, v50: dict) -> dict:
    v49 = json.loads(V49[name].read_text())
    by49 = {t["id"]: t for t in v49["tracks"]}
    rows = []
    for t in v50["tracks"]:
        b = by49.get(t["id"], {})
        rows.append({
            "id": t["id"],
            "v49_majmin": b.get("majmin"),
            "v50_majmin": t["majmin"],
            "delta_majmin": round(t["majmin"] - b.get("majmin", 0), 4) if b else None,
        })
    rows.sort(key=lambda r: -(r["delta_majmin"] or 0))
    return {
        "split": name,
        "v49_summary": v49["summary"],
        "v50_summary": v50["summary"],
        "delta_majmin_agg": round(
            v50["summary"]["avg_majmin_wcsr"] - v49["summary"]["avg_majmin_wcsr"], 4,
        ),
        "tracks": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="confidence", choices=["confidence", "tta", "off"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--dict", default="submission",
                    choices=["submission", "ismir2017", "full", "extended"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "both"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    diffs = {}
    for split in splits:
        out = ANALYSIS / f"BASELINE_mir_gold_{split.upper()}_v50{args.tag}.json"
        print(f"Evaluating {split.upper()} → {out.name} "
              f"(select={args.select} margin={args.margin} dict={args.dict})…")
        result = _run_eval(split, out, args)
        diffs[split] = _diff_split(split, result)
        print(f"  {split.upper()} majmin {result['summary']['avg_majmin_wcsr']:.3f} "
              f"(Δ vs v49 {diffs[split]['delta_majmin_agg']:+.4f})")
        guard = next((t for t in result["tracks"] if GUARD_TRACK in t["id"]), None)
        if guard is not None and guard["majmin"] < GUARD_MIN:
            print(f"  WARNING: {GUARD_TRACK} regressed to {guard['majmin']:.3f} "
                  f"(< {GUARD_MIN}) — acceptance criterion violated")

    diff_out = ANALYSIS / f"BASELINE_v49_vs_v50{args.tag}.json"
    diff_out.write_text(json.dumps(diffs, indent=2) + "\n")
    print(f"Wrote {diff_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
