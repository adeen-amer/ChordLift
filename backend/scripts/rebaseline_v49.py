#!/usr/bin/env python3
"""Phase 13 — v49 re-baseline (pitch reliability gate) + diff vs v48."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

V48_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v48.json"
V48_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v48.json"
V49_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v49.json"
V49_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v49.json"
DIFF_OUT = BACKEND / "analysis" / "BASELINE_v48_vs_v49.json"
DOC_OUT = BACKEND / "analysis" / "BASELINE_v49_README.md"


def _run_eval(split: str, out_path: Path) -> dict:
    import os

    os.environ["CHORD_ENGINE"] = "ml"
    os.environ["CHORD_ENGINE_STRICT"] = "1"
    os.environ.setdefault("CHORD_ML_POSTPROCESS", "raw")
    os.environ.setdefault("CHORD_ML_MODEL", "chordia")
    argv = [
        "eval_gold_mir.py",
        "--no-cache",
        "--split",
        split,
        "--require-audio-identity",
        "--write",
        str(out_path),
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


def _diff_split(name: str, v48_path: Path, v49: dict) -> dict:
    v48 = json.loads(v48_path.read_text())
    by48 = {t["id"]: t for t in v48["tracks"]}
    rows = []
    for t in v49["tracks"]:
        tid = t["id"]
        b = by48.get(tid, {})
        d_mm = round(t["majmin"] - b.get("majmin", 0), 4) if b else None
        rows.append({
            "id": tid,
            "v48_majmin": b.get("majmin"),
            "v49_majmin": t["majmin"],
            "delta_majmin": d_mm,
        })
    rows.sort(key=lambda r: -(r["delta_majmin"] or 0))
    return {
        "split": name,
        "v48_summary": v48["summary"],
        "v49_summary": v49["summary"],
        "delta_majmin_agg": round(
            v49["summary"]["avg_majmin_wcsr"] - v48["summary"]["avg_majmin_wcsr"], 4,
        ),
        "tracks": rows,
    }


def main() -> int:
    print("Evaluating DEV → v49…")
    dev = _run_eval("dev", V49_DEV)
    print("Evaluating TEST → v49…")
    test = _run_eval("test", V49_TEST)
    diff = {"dev": _diff_split("dev", V48_DEV, dev), "test": _diff_split("test", V48_TEST, test)}
    DIFF_OUT.write_text(json.dumps(diff, indent=2) + "\n")
    DOC_OUT.write_text(
        "# Baseline v49 — pitch reliability gate (Phase 13)\n\n"
        f"TEST majmin: {test['summary']['avg_majmin_wcsr']:.3f} "
        f"(v48 {json.loads(V48_TEST.read_text())['summary']['avg_majmin_wcsr']:.3f}, "
        f"Δ {diff['test']['delta_majmin_agg']:+.3f})\n\n"
        f"DEV majmin: {dev['summary']['avg_majmin_wcsr']:.3f} "
        f"(v48 {json.loads(V48_DEV.read_text())['summary']['avg_majmin_wcsr']:.3f}, "
        f"Δ {diff['dev']['delta_majmin_agg']:+.3f})\n\n"
        "Key fix: `another-one-bites-the-dust` recovered via bass-heavy pitch gate.\n"
    )
    print(f"Wrote {V49_DEV.name}, {V49_TEST.name}, {DIFF_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
