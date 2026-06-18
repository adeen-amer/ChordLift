#!/usr/bin/env python3
"""Phase 12.5 — v48 re-baseline (pitch-only serving) + diff vs v47."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

V47_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v47.json"
V47_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v47.json"
V48_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v48.json"
V48_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v48.json"
DIFF_OUT = BACKEND / "analysis" / "BASELINE_v47_vs_v48.json"
DOC_OUT = BACKEND / "analysis" / "BASELINE_v48_README.md"


def _run_eval(split: str, out_path: Path) -> dict:
    import os

    os.environ["CHORD_ENGINE"] = "ml"
    os.environ["CHORD_ENGINE_STRICT"] = "1"
    os.environ.setdefault("CHORD_ML_POSTPROCESS", "raw")
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


def _diff_split(name: str, v47_path: Path, v48: dict) -> dict:
    v47 = json.loads(v47_path.read_text())
    by47 = {t["id"]: t for t in v47["tracks"]}
    rows = []
    for t in v48["tracks"]:
        tid = t["id"]
        b = by47.get(tid, {})
        d_mm = round(t["majmin"] - b.get("majmin", 0), 4) if b else None
        rows.append({
            "id": tid,
            "v47_majmin": b.get("majmin"),
            "v48_majmin": t["majmin"],
            "delta_majmin": d_mm,
            "v47_seg": b.get("seg"),
            "v48_seg": t["seg"],
            "delta_seg": round(t["seg"] - b.get("seg", 0), 4) if b else None,
        })
    rows.sort(key=lambda r: -(r["delta_majmin"] or 0))
    return {
        "split": name,
        "v47_summary": v47["summary"],
        "v48_summary": v48["summary"],
        "delta_majmin_agg": round(
            v48["summary"]["avg_majmin_wcsr"] - v47["summary"]["avg_majmin_wcsr"], 4,
        ),
        "tracks": rows,
    }


def _write_doc(diff: dict) -> None:
    dev = diff["dev"]
    test = diff["test"]
    lines = [
        "# Baseline v48 — pitch-only serving on v47 audio",
        "",
        "v48 = `CHORD_ML_POSTPROCESS=raw` (no chroma-align / light-merge) + pitch correction.",
        "Phase 12.5 ablation on identity-verified v47 audio — see `PHASE12_5_POSTPROCESS_ABLATION.md`.",
        "",
        "## Aggregate majmin / seg",
        "",
        "| Split | v47 majmin | v48 majmin | Δ | v47 seg | v48 seg |",
        "|-------|------------|------------|---|---------|---------|",
        f"| DEV | {dev['v47_summary']['avg_majmin_wcsr']:.3f} | "
        f"{dev['v48_summary']['avg_majmin_wcsr']:.3f} | {dev['delta_majmin_agg']:+.3f} | "
        f"{dev['v47_summary']['avg_seg_score']:.3f} | {dev['v48_summary']['avg_seg_score']:.3f} |",
        f"| TEST | {test['v47_summary']['avg_majmin_wcsr']:.3f} | "
        f"{test['v48_summary']['avg_majmin_wcsr']:.3f} | {test['delta_majmin_agg']:+.3f} | "
        f"{test['v47_summary']['avg_seg_score']:.3f} | {test['v48_summary']['avg_seg_score']:.3f} |",
        "",
        "## Largest v48 gains vs v47 (bypass serving)",
        "",
    ]
    for split_name, block in (("DEV", dev), ("TEST", test)):
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append("| id | v47 | v48 | Δ majmin |")
        lines.append("|----|-----|-----|----------|")
        for r in block["tracks"][:8]:
            if r["delta_majmin"] and abs(r["delta_majmin"]) >= 0.01:
                lines.append(
                    f"| {r['id']} | {r['v47_majmin']:.3f} | {r['v48_majmin']:.3f} | {r['delta_majmin']:+.3f} |"
                )
        lines.append("")
    DOC_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not args.skip_eval:
        dev = _run_eval("dev", V48_DEV)
        test = _run_eval("test", V48_TEST)
    else:
        dev = json.loads(V48_DEV.read_text())
        test = json.loads(V48_TEST.read_text())

    diff = {
        "note": "v48 pitch-only serving on v47 audio",
        "dev": _diff_split("dev", V47_DEV, dev),
        "test": _diff_split("test", V47_TEST, test),
    }
    DIFF_OUT.write_text(json.dumps(diff, indent=2))
    _write_doc(diff)
    print(f"Wrote {V48_DEV}, {V48_TEST}, {DIFF_OUT}, {DOC_OUT}")
    print(
        f"DEV Δmajmin={diff['dev']['delta_majmin_agg']:+.3f} "
        f"TEST Δmajmin={diff['test']['delta_majmin_agg']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
