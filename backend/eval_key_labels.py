#!/usr/bin/env python3
"""Score key-label accuracy from eval_chords.py JSON results (no re-analysis)."""
import argparse
import json
import sys
from pathlib import Path

from eval_key_utils import key_labels_match

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def _load_tracks_by_id(fixtures_path):
    data = json.loads(fixtures_path.read_text())
    return {t["id"]: t for t in data.get("tracks", [])}


def recompute_key_matches(results, tracks_by_id):
    """Re-score key_match from stored key labels + fixture flags."""
    for row in results:
        if row.get("skipped") or row.get("key_skipped"):
            continue
        expected = row.get("key_expected")
        predicted = row.get("key")
        if not expected:
            continue
        track = tracks_by_id.get(row["id"], {})
        row["key_match"] = key_labels_match(
            expected,
            predicted or "",
            transpose_invariant=track.get(
                "key_transpose_invariant",
                track.get("transpose_invariant", False),
            ),
            allow_relative=track.get("allow_relative_key", True),
        )
    return results


def score_key_results(results):
    """Return key scoring summary from eval result dicts."""
    scored = []
    for row in results:
        if row.get("skipped"):
            continue
        if row.get("key_skipped"):
            continue
        if row.get("key_expected") is None:
            continue
        scored.append(row)

    if not scored:
        return {
            "total": 0,
            "matched": 0,
            "rate": 0.0,
            "mismatches": [],
            "matches": [],
        }

    matched = [r for r in scored if r.get("key_match")]
    mismatches = [r for r in scored if not r.get("key_match")]

    return {
        "total": len(scored),
        "matched": len(matched),
        "rate": len(matched) / len(scored),
        "mismatches": mismatches,
        "matches": matched,
    }


def print_key_report(summary):
    total = summary["total"]
    matched = summary["matched"]
    rate = summary["rate"]

    print("=== Key label accuracy (separate from root recall) ===")
    if total == 0:
        print("No tracks with expected_key in results.")
        return

    print(f"Matched: {matched}/{total} ({rate * 100:.1f}%)")
    print("(includes relative major/minor + dorian↔minor when enabled)")

    if summary["mismatches"]:
        print("\nKey mismatches:")
        for row in summary["mismatches"]:
            title = row.get("title") or row["id"]
            print(
                f"  - {row['id']} ({title}): "
                f"expected {row.get('key_expected')} → got {row.get('key')}"
            )

    if summary["matches"]:
        print(f"\nKey matches ({len(summary['matches'])}):")
        for row in summary["matches"]:
            title = row.get("title") or row["id"]
            print(f"  ✓ {row['id']} ({title}): {row.get('key')}")


def main():
    parser = argparse.ArgumentParser(description="Score key labels from eval results JSON")
    parser.add_argument(
        "results_file",
        type=Path,
        help="JSON written by eval_chords.py --write-results",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=FIXTURES,
        help="Fixture file for per-track key scoring flags",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=None,
        help="Fail if key match rate is below this (0–1)",
    )
    parser.add_argument(
        "--require-all-keys",
        action="store_true",
        help="Fail if any track with expected_key has a key mismatch",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"ERROR: results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(args.results_file.read_text())
    results = payload.get("results", payload)
    tracks_by_id = _load_tracks_by_id(args.fixtures)
    results = recompute_key_matches(results, tracks_by_id)
    summary = score_key_results(results)
    print_key_report(summary)

    if args.min_rate is not None and summary["total"] > 0:
        if summary["rate"] < args.min_rate:
            print(
                f"\nERROR: key match rate {summary['rate']:.1%} "
                f"below --min-rate {args.min_rate:.1%}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.require_all_keys and summary["mismatches"]:
        print(
            f"\nERROR: {len(summary['mismatches'])} key label mismatch(es)",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
