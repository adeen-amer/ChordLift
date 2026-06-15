#!/usr/bin/env python3
"""Score duration-weighted timeline agreement from eval_chords.py JSON results."""
import argparse
import json
import sys
from pathlib import Path

from eval_fixture_utils import load_all_overlays, merge_track_enrichment

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def _load_tracks_by_id(fixtures_path):
    data = json.loads(fixtures_path.read_text())
    return {t["id"]: t for t in data.get("tracks", [])}


def _track_has_timeline_fixture(track):
    return bool(
        track.get("reference_timeline")
        or track.get("progression")
    )


def score_timeline_results(results, tracks_by_id, overlays=None):
    overlays = overlays or load_all_overlays()
    scored = []
    mae_rows = []
    for row in results:
        if row.get("skipped"):
            continue
        track = merge_track_enrichment(tracks_by_id.get(row["id"], {}), overlays)
        if not _track_has_timeline_fixture(track):
            continue
        if row.get("timeline_score") is None:
            continue
        scored.append(row)
        if row.get("switch_mae_sec") is not None:
            mae_rows.append(row)

    if not scored:
        return {
            "total": 0,
            "avg_timeline": 0.0,
            "avg_switch_mae": None,
            "rows": [],
        }

    avg_timeline = sum(r["timeline_score"] for r in scored) / len(scored)
    avg_mae = None
    if mae_rows:
        avg_mae = sum(r["switch_mae_sec"] for r in mae_rows) / len(mae_rows)

    return {
        "total": len(scored),
        "avg_timeline": avg_timeline,
        "avg_switch_mae": avg_mae,
        "rows": scored,
    }


def print_timeline_report(summary):
    print("=== Timeline agreement (duration-weighted chord correctness) ===")
    if summary["total"] == 0:
        print("No tracks with progression/reference_timeline in results.")
        return

    print(
        f"Average timeline score: {summary['avg_timeline'] * 100:.1f}% "
        f"over {summary['total']} track(s)"
    )
    if summary["avg_switch_mae"] is not None:
        print(
            f"Average switch timing MAE: {summary['avg_switch_mae']:.2f}s "
            f"(lower is better)"
        )

    for row in sorted(summary["rows"], key=lambda r: r.get("timeline_score", 0)):
        title = row.get("title") or row["id"]
        mae = row.get("switch_mae_sec")
        mae_note = f"  mae={mae:.2f}s" if mae is not None else ""
        print(
            f"  {row['timeline_score']:.3f}  {row['id']} ({title}){mae_note}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Score timeline agreement from eval results JSON",
    )
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument(
        "--min-rate",
        type=float,
        default=None,
        help="Fail if average timeline score is below this (0–1)",
    )
    parser.add_argument(
        "--max-switch-mae",
        type=float,
        default=None,
        help="Fail if average switch MAE exceeds this (seconds)",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"ERROR: results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(args.results_file.read_text())
    results = payload.get("results", payload)
    tracks_by_id = _load_tracks_by_id(args.fixtures)
    overlays = load_all_overlays()
    summary = score_timeline_results(results, tracks_by_id, overlays)
    print_timeline_report(summary)

    if args.min_rate is not None and summary["total"] > 0:
        if summary["avg_timeline"] < args.min_rate:
            print(
                f"\nERROR: timeline score {summary['avg_timeline']:.1%} "
                f"below --min-rate {args.min_rate:.1%}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.max_switch_mae is not None and summary["avg_switch_mae"] is not None:
        if summary["avg_switch_mae"] > args.max_switch_mae:
            print(
                f"\nERROR: switch MAE {summary['avg_switch_mae']:.2f}s "
                f"above --max-switch-mae {args.max_switch_mae:.2f}s",
                file=sys.stderr,
            )
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
