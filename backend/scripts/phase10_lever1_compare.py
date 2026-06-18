#!/usr/bin/env python3
"""Phase 10 Lever 1 — compare full vs bypass post-processing on DEV+TEST."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval_gold_mir import _avg, evaluate_track  # noqa: E402
from engine_verify import prepare_eval_environment  # noqa: E402
from gold_audio_verify import approved_track_ids  # noqa: E402

CATALOG = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
SPLIT = BACKEND / "tests" / "fixtures" / "gold_split_v2.json"
BASELINE_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v45.json"
BASELINE_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v45.json"


def _load_tracks(split_name: str) -> list[dict]:
    catalog = json.loads(CATALOG.read_text())
    split = json.loads(SPLIT.read_text())
    approved = approved_track_ids()
    wanted = {t for t in split[split_name] if t in approved}
    return [t for t in catalog["tracks"] if t["id"] in wanted]


def _run_split(tracks: list[dict], mode: str) -> list[dict]:
    os.environ["CHORD_ML_POSTPROCESS"] = mode
    prepare_eval_environment(no_cache=True)
    reports = []
    for track in tracks:
        reports.append(evaluate_track(track, eval_start=None, eval_end=None))
    return reports


def _summary(reports: list[dict]) -> dict:
    return {
        "majmin": _avg(reports, "majmin"),
        "sevenths": _avg(reports, "sevenths"),
        "seg": _avg(reports, "seg"),
        "root": _avg(reports, "root"),
        "n": sum(1 for r in reports if not r.get("skipped")),
    }


def _compare_table(before: list[dict], after: list[dict]) -> list[dict]:
    by_before = {r["id"]: r for r in before if not r.get("skipped")}
    by_after = {r["id"]: r for r in after if not r.get("skipped")}
    rows = []
    for tid in sorted(by_before):
        b, a = by_before[tid], by_after.get(tid, {})
        rows.append({
            "id": tid,
            "majmin_before": b.get("majmin"),
            "majmin_after": a.get("majmin"),
            "majmin_delta": round((a.get("majmin") or 0) - (b.get("majmin") or 0), 4),
            "seg_before": b.get("seg"),
            "seg_after": a.get("seg"),
            "seg_delta": round((a.get("seg") or 0) - (b.get("seg") or 0), 4),
            "est_before": b.get("est_segments"),
            "est_after": a.get("est_segments"),
        })
    return rows


def _print_table(split_name: str, rows: list[dict], before_s: dict, after_s: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"{split_name.upper()} — full (v45) vs bypass")
    print(
        f"AGG full:   majmin={before_s['majmin']:.3f} seg={before_s['seg']:.3f} | "
        f"bypass: majmin={after_s['majmin']:.3f} seg={after_s['seg']:.3f} | "
        f"Δ majmin={after_s['majmin'] - before_s['majmin']:+.3f} "
        f"Δ seg={after_s['seg'] - before_s['seg']:+.3f}"
    )
    print(f"{'id':<28} {'majmin':>7} {'→':>3} {'bypass':>7} {'Δ':>7} | {'seg':>7} {'→':>3} {'bypass':>7} {'Δ':>7} est")
    for r in rows:
        print(
            f"{r['id']:<28} "
            f"{r['majmin_before']:7.3f} {'→':>3} {r['majmin_after']:7.3f} {r['majmin_delta']:+7.3f} | "
            f"{r['seg_before']:7.3f} {'→':>3} {r['seg_after']:7.3f} {r['seg_delta']:+7.3f} "
            f"{r['est_before']:>3}→{r['est_after']:<3}"
        )


def _check_gate(split_name: str, before_s: dict, after_s: dict, rows: list[dict]) -> bool:
    ok = True
    if after_s["majmin"] <= before_s["majmin"]:
        print(f"GATE FAIL [{split_name}]: majmin did not improve", file=sys.stderr)
        ok = False
    if after_s["seg"] <= before_s["seg"]:
        print(f"GATE FAIL [{split_name}]: seg did not improve", file=sys.stderr)
        ok = False
    for r in rows:
        if r["majmin_delta"] < -0.05:
            print(
                f"GATE FAIL [{split_name}]: {r['id']} majmin regressed {r['majmin_delta']:+.3f}",
                file=sys.stderr,
            )
            ok = False
    if ok:
        print(f"GATE PASS [{split_name}]")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 10 Lever 1 A/B")
    parser.add_argument("--write", type=Path, default=BACKEND / "analysis" / "PHASE10_LEVER1_COMPARE.json")
    args = parser.parse_args()

    os.environ.setdefault("CHORD_ENGINE", "ml")
    os.environ.setdefault("CHORD_ENGINE_STRICT", "1")

    payload = {"splits": {}}
    gate_ok = True

    for split_name in ("dev", "test"):
        tracks = _load_tracks(split_name)
        print(f"\nRunning {split_name} full ({len(tracks)} tracks)...")
        full = _run_split(tracks, "full")
        print(f"Running {split_name} bypass ({len(tracks)} tracks)...")
        bypass = _run_split(tracks, "bypass")

        before_s = _summary(full)
        after_s = _summary(bypass)
        rows = _compare_table(full, bypass)
        _print_table(split_name, rows, before_s, after_s)

        if split_name == "test":
            gate_ok = _check_gate(split_name, before_s, after_s, rows) and gate_ok

        payload["splits"][split_name] = {
            "full": {"summary": before_s, "tracks": full},
            "bypass": {"summary": after_s, "tracks": bypass},
            "per_track": rows,
        }

    payload["gate_pass"] = gate_ok
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.write}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
