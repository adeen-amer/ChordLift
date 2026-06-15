#!/usr/bin/env python3
"""Evaluate ML chord engine on Beatles tracks vs Hooktheory references."""
import json
import sys
from pathlib import Path

from analyzer import ANALYZER_VERSION
from chord_engine import CHORD_ENGINE
from eval_chords import evaluate_track
from eval_fixture_utils import load_enrichment

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "beatles_refs.json"
BEATLES_IDS = {"let-it-be", "hey-jude", "yellow-submarine"}


def main():
    base = Path(__file__).parent
    overlays = load_enrichment()

    # Prefer dedicated beatles fixture; fall back to merged chord_refs entries.
    if FIXTURES.exists():
        refs = json.loads(FIXTURES.read_text())
        tracks = refs["tracks"]
    else:
        all_refs = json.loads(
            (base / "tests" / "fixtures" / "chord_refs.json").read_text()
        )
        tracks = [t for t in all_refs["tracks"] if t["id"] in BEATLES_IDS]

    print(f"Beatles eval | analyzer v{ANALYZER_VERSION} | engine={CHORD_ENGINE}\n")

    results = []
    for track in tracks:
        result = evaluate_track(track, base, overlays)
        results.append(result)
        if result.get("skipped"):
            print(f"SKIP {result['id']}: {result['reason']}")
            continue
        status = "PASS" if result["passed"] else "FAIL"
        key_note = ""
        if result.get("key_expected"):
            marker = "✓" if result.get("key_match") else "✗"
            key_note = f" [key {marker}: expected {result['key_expected']}, got {result.get('key')}]"
        print(f"{status} {track['title']}")
        print(f"  recall={result['root_recall']} segments={result['segments']} key={result.get('key')}{key_note}")
        print(f"  roots: {result['unique_roots']}")
        print(f"  top:   {result['top_chords']}")
        print(f"  ref:   {track.get('source', '')}\n")

    failed = [r for r in results if not r.get("skipped") and not r.get("passed")]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
