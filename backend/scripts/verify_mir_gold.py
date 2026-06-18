#!/usr/bin/env python3
"""Compare mir_eval gold report to analysis/BASELINE_mir_gold_v38.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
BASELINE = BACKEND / "analysis" / "BASELINE_mir_gold_v44.json"
REPORT = BACKEND / "analysis" / "mir_gold_latest.json"

METRICS = (
    ("avg_root_wcsr", "root", -0.01),
    ("avg_majmin_wcsr", "majmin", -0.01),
    ("avg_sevenths_wcsr", "sevenths", -0.01),
    ("avg_seg_score", "seg", -0.02),
)


def main() -> int:
    env = {**dict(__import__("os").environ), "CHORD_ENGINE": "ml", "CHORD_ENGINE_STRICT": "1"}
    subprocess.run(
        [sys.executable, str(BACKEND / "eval_gold_mir.py"), "--no-cache", "--ids", "let-it-be,yellow-submarine"],
        cwd=BACKEND,
        env=env,
        check=True,
    )

    baseline = json.loads(BASELINE.read_text())
    report = json.loads(REPORT.read_text())
    ref = baseline["summary"]
    cur = report["summary"]

    print(f"Baseline v{baseline.get('analyzer_version')} vs v{report.get('analyzer_version')}\n")
    ok = True
    for key, label, tol in METRICS:
        base = ref.get(key)
        val = cur.get(key)
        if base is None or val is None:
            continue
        delta = val - base
        sign = "+" if delta >= 0 else ""
        status = "OK" if val >= base + tol or key == "avg_seg_score" and val >= base - 0.02 else "REGRESSED"
        if status == "REGRESSED" and key in ("avg_majmin_wcsr", "avg_sevenths_wcsr", "avg_root_wcsr"):
            ok = False
        print(f"  {label:8s}: {base:.3f} → {val:.3f} ({sign}{delta:.3f}) [{status}]")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
