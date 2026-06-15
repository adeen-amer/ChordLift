#!/usr/bin/env python3
"""Score symbol recall from eval_chords.py JSON results (no re-analysis)."""
import argparse
import json
import sys
from pathlib import Path

from eval_fixture_utils import load_all_overlays, merge_track_enrichment

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def _load_tracks_by_id(fixtures_path):
    data = json.loads(fixtures_path.read_text())
    return {t["id"]: t for t in data.get("tracks", [])}


def score_symbol_results(results, tracks_by_id, overlays=None):
    overlays = overlays or load_all_overlays()
    scored = []
    for row in results:
        if row.get("skipped"):
            continue
        track = merge_track_enrichment(tracks_by_id.get(row["id"], {}), overlays)
        if not track.get("expected_symbols") and row.get("symbol_recall") is None:
            continue
        if row.get("symbol_recall") is None:
            continue
        scored.append(row)

    if not scored:
        return {"total": 0, "avg_recall": 0.0, "rows": []}

    avg = sum(r["symbol_recall"] for r in scored) / len(scored)
    return {"total": len(scored), "avg_recall": avg, "rows": scored}


def print_symbol_report(summary):
    print("=== Symbol recall (separate from root recall) ===")
    if summary["total"] == 0:
        print("No tracks with expected_symbols in results.")
        return

    print(
        f"Average symbol recall: {summary['avg_recall'] * 100:.1f}% "
        f"over {summary['total']} track(s)"
    )
    for row in sorted(summary["rows"], key=lambda r: r.get("symbol_recall", 0)):
        title = row.get("title") or row["id"]
        print(f"  {row['symbol_recall']:.3f}  {row['id']} ({title})")


def main():
    parser = argparse.ArgumentParser(description="Score symbol recall from eval results JSON")
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument(
        "--min-rate",
        type=float,
        default=None,
        help="Fail if average symbol recall is below this (0–1)",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"ERROR: results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(args.results_file.read_text())
    results = payload.get("results", payload)
    tracks_by_id = _load_tracks_by_id(args.fixtures)
    overlays = load_all_overlays()
    summary = score_symbol_results(results, tracks_by_id, overlays)
    print_symbol_report(summary)

    if args.min_rate is not None and summary["total"] > 0:
        if summary["avg_recall"] < args.min_rate:
            print(
                f"\nERROR: symbol recall {summary['avg_recall']:.1%} "
                f"below --min-rate {args.min_rate:.1%}",
                file=sys.stderr,
            )
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
