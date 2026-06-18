#!/usr/bin/env python3
"""Phase 11.5 Step 2 — v47 re-baseline + per-track diff vs v46."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

V46_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v46.json"
V46_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v46.json"
V47_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v47.json"
V47_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v47.json"
DIFF_OUT = BACKEND / "analysis" / "BASELINE_v46_vs_v47.json"
DOC_OUT = BACKEND / "analysis" / "BASELINE_v47_README.md"


def _run_eval(split: str, out_path: Path) -> dict:
    import os

    env = os.environ.copy()
    env["CHORD_ENGINE"] = "ml"
    env["CHORD_ENGINE_STRICT"] = "1"
    # Run in-process — subprocess can SIGSEGV on some macOS + torch combos when parent already imported ML libs
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


def _diff_split(name: str, v46_path: Path, v47: dict) -> dict:
    v46 = json.loads(v46_path.read_text())
    by46 = {t["id"]: t for t in v46["tracks"]}
    rows = []
    for t in v47["tracks"]:
        tid = t["id"]
        b = by46.get(tid, {})
        d_mm = round(t["majmin"] - b.get("majmin", 0), 4) if b else None
        rows.append({
            "id": tid,
            "v46_majmin": b.get("majmin"),
            "v47_majmin": t["majmin"],
            "delta_majmin": d_mm,
            "v46_seg": b.get("seg"),
            "v47_seg": t["seg"],
            "delta_seg": round(t["seg"] - b.get("seg", 0), 4) if b else None,
        })
    rows.sort(key=lambda r: -(r["delta_majmin"] or 0))
    return {
        "split": name,
        "v46_summary": v46["summary"],
        "v47_summary": v47["summary"],
        "delta_majmin_agg": round(
            v47["summary"]["avg_majmin_wcsr"] - v46["summary"]["avg_majmin_wcsr"], 4,
        ),
        "tracks": rows,
    }


def _write_doc(diff: dict) -> None:
    dev = diff["dev"]
    test = diff["test"]
    lines = [
        "# Baseline v47 — identity-verified audio (NOT comparable to v46)",
        "",
        "v47 = re-measurement on YT-Music metadata-matched audio with hardened identity gate.",
        "Majmin gains reflect **fixed reference recordings**, not model changes (byte-identical engine).",
        "",
        "## Aggregate majmin / seg",
        "",
        "| Split | v46 majmin | v47 majmin | Δ | v46 seg | v47 seg |",
        "|-------|------------|------------|---|---------|---------|",
        f"| DEV | {dev['v46_summary']['avg_majmin_wcsr']:.3f} | "
        f"{dev['v47_summary']['avg_majmin_wcsr']:.3f} | {dev['delta_majmin_agg']:+.3f} | "
        f"{dev['v46_summary']['avg_seg_score']:.3f} | {dev['v47_summary']['avg_seg_score']:.3f} |",
        f"| TEST | {test['v46_summary']['avg_majmin_wcsr']:.3f} | "
        f"{test['v47_summary']['avg_majmin_wcsr']:.3f} | {test['delta_majmin_agg']:+.3f} | "
        f"{test['v46_summary']['avg_seg_score']:.3f} | {test['v47_summary']['avg_seg_score']:.3f} |",
        "",
        "## Largest v47 gains (corrected ruler)",
        "",
    ]
    for split_name, block in (("DEV", dev), ("TEST", test)):
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append("| id | v46 | v47 | Δ majmin |")
        lines.append("|----|-----|-----|----------|")
        for r in block["tracks"][:8]:
            if r["delta_majmin"] and abs(r["delta_majmin"]) >= 0.01:
                lines.append(
                    f"| {r['id']} | {r['v46_majmin']:.3f} | {r['v47_majmin']:.3f} | {r['delta_majmin']:+.3f} |"
                )
        lines.append("")
    DOC_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true", help="Only diff existing v47 files")
    args = parser.parse_args()

    if not args.skip_eval:
        dev = _run_eval("dev", V47_DEV)
        test = _run_eval("test", V47_TEST)
    else:
        dev = json.loads(V47_DEV.read_text())
        test = json.loads(V47_TEST.read_text())

    diff = {
        "note": "v47 corrected ruler — not comparable to v46 as model improvement",
        "dev": _diff_split("dev", V46_DEV, dev),
        "test": _diff_split("test", V46_TEST, test),
    }
    DIFF_OUT.write_text(json.dumps(diff, indent=2))
    _write_doc(diff)
    print(f"Wrote {V47_DEV}, {V47_TEST}, {DIFF_OUT}, {DOC_OUT}")
    print(
        f"DEV Δmajmin={diff['dev']['delta_majmin_agg']:+.3f} "
        f"TEST Δmajmin={diff['test']['delta_majmin_agg']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
