#!/usr/bin/env python3
"""Score boundary alignment from eval_chords.py JSON results (no re-analysis)."""
import argparse
import json
import sys
from pathlib import Path

from eval_fixture_utils import load_all_overlays, merge_track_enrichment

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def _load_tracks_by_id(fixtures_path):
    data = json.loads(fixtures_path.read_text())
    return {t["id"]: t for t in data.get("tracks", [])}


def _track_has_boundary_fixture(track):
    return bool(track.get("reference_changes") or track.get("progression"))


def _is_phase_aligned(row, track):
    return row.get("boundary_method") == "phase_aligned" or track.get(
        "boundary_method", "phase_aligned",
    ) == "phase_aligned"


def score_boundary_results(results, tracks_by_id, overlays=None):
    overlays = overlays or load_all_overlays()
    scored = []
    hand = []
    for row in results:
        if row.get("skipped"):
            continue
        track = merge_track_enrichment(tracks_by_id.get(row["id"], {}), overlays)
        if not _track_has_boundary_fixture(track):
            continue
        if row.get("boundary_score") is None:
            continue
        scored.append({**row, "phase_aligned": _is_phase_aligned(row, track)})
        if _is_phase_aligned(row, track):
            hand.append(row)

    if not scored:
        return {"total": 0, "avg_score": 0.0, "hand_total": 0, "hand_avg_score": 0.0, "rows": []}

    avg = sum(r["boundary_score"] for r in scored) / len(scored)
    hand_avg = sum(r["boundary_score"] for r in hand) / len(hand) if hand else 0.0
    return {
        "total": len(scored),
        "avg_score": avg,
        "hand_total": len(hand),
        "hand_avg_score": hand_avg,
        "rows": scored,
    }


def print_boundary_report(summary):
    print("=== Boundary alignment (progression / reference changes) ===")
    if summary["total"] == 0:
        print("No tracks with progression/reference_changes in results.")
        return

    print(
        f"Average boundary score: {summary['avg_score'] * 100:.1f}% "
        f"over {summary['total']} track(s)"
    )
    if summary.get("hand_total"):
        print(
            f"Phase-aligned (dynamic): {summary['hand_avg_score'] * 100:.1f}% "
            f"over {summary['hand_total']} track(s)"
        )
    for row in sorted(summary["rows"], key=lambda r: r.get("boundary_score", 0)):
        title = row.get("title") or row["id"]
        print(f"  {row['boundary_score']:.3f}  {row['id']} ({title})")


def main():
    parser = argparse.ArgumentParser(description="Score boundary alignment from eval results JSON")
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument(
        "--min-rate",
        type=float,
        default=None,
        help="Fail if average boundary score is below this (0–1)",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"ERROR: results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(args.results_file.read_text())
    results = payload.get("results", payload)
    tracks_by_id = _load_tracks_by_id(args.fixtures)
    overlays = load_all_overlays()
    summary = score_boundary_results(results, tracks_by_id, overlays)
    print_boundary_report(summary)

    if args.min_rate is not None and summary["total"] > 0:
        if summary["avg_score"] < args.min_rate:
            print(
                f"\nERROR: boundary score {summary['avg_score']:.1%} "
                f"below --min-rate {args.min_rate:.1%}",
                file=sys.stderr,
            )
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
