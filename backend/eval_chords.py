#!/usr/bin/env python3
"""Evaluate chord detection against reference fixtures."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import librosa

from analyzer import ANALYZER_VERSION, _chord_root_name
from chord_engine import CHORD_ENGINE, extract_chords

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def _normalize_root(root):
    """Normalize to sharp spelling for eval comparison."""
    flat_map = {"Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#"}
    return flat_map.get(root, root)


def _expand_enharmonic_roots(roots):
    """Accept both sharp and flat spellings in expected roots."""
    flat_map = {"A#": "Bb", "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab"}
    sharp_map = {v: k for k, v in flat_map.items()}
    expanded = set()
    for r in roots:
        sharp = _normalize_root(r)
        expanded.add(sharp)
        if sharp in sharp_map:
            expanded.add(sharp_map[sharp])
        if r in flat_map:
            expanded.add(flat_map[r])
    return expanded


def _segment_roots(segments):
    roots = []
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        roots.append(_normalize_root(_chord_root_name(chord)))
    return roots


def _sus_fraction(segments):
    if not segments:
        return 0.0
    sus = sum(
        1 for seg in segments
        if "sus" in seg.get("chord", "").lower()
    )
    return sus / len(segments)


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#"}


def _root_to_pc(root):
    root = _normalize_root(root)
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    try:
        return PITCH_CLASSES.index(root)
    except ValueError:
        return None


def _pc_to_root(pc):
    return PITCH_CLASSES[pc % 12]


def _transpose_root(root, semitones):
    pc = _root_to_pc(root)
    if pc is None:
        return root
    return _pc_to_root(pc + semitones)



from eval_fixture_utils import load_all_overlays, load_chord_stamps, prepare_eval_track
from eval_key_utils import key_labels_match as _key_labels_match
from eval_chord_utils import (
    boundary_alignment_score,
    filter_reference_changes,
    filter_reference_timeline,
    resolve_boundary_references,
    switch_timing_mae,
    symbol_recall,
    timeline_agreement_score,
    normalize_chord_symbol,
)


def _segment_symbols(segments):
    symbols = []
    for seg in segments:
        chord = seg.get("chord", "")
        if not chord or chord == "N":
            continue
        symbols.append(normalize_chord_symbol(chord))
    return symbols


def _root_recall(predicted, expected):
    if not predicted:
        return 0.0
    expected_set = _expand_enharmonic_roots(expected)
    hits = sum(1 for r in predicted if _normalize_root(r) in expected_set)
    return hits / len(predicted)


def _root_recall_transpose_invariant(predicted, expected):
    """Best root recall over ±1 semitone (YouTube pitch drift vs Hooktheory)."""
    if not predicted:
        return 0.0
    best = _root_recall(predicted, expected)
    for shift in (-1, 1):
        shifted = [_transpose_root(r, shift) for r in predicted]
        best = max(best, _root_recall(shifted, expected))
    return best


def evaluate_track(track, base_dir, overlays=None, stamps=None, live_boundaries=False):
    track = prepare_eval_track(
        track, overlays, stamps, live_boundaries=live_boundaries,
    )
    using_stamps = bool(track.get("reference_changes")) and not live_boundaries
    audio_path = base_dir / track["file"]
    if not audio_path.exists():
        return {
            "id": track["id"],
            "title": track.get("title"),
            "skipped": True,
            "reason": f"missing {audio_path}",
        }

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    segments, key_info = extract_chords(y, sr)
    predicted = _segment_roots(segments)
    unique = set(predicted)
    recall = _root_recall(predicted, track["expected_roots"])
    if track.get("transpose_invariant"):
        recall = max(recall, _root_recall_transpose_invariant(predicted, track["expected_roots"]))

    root_passed = (
        recall >= track.get("min_root_recall", 0.5)
        and len(unique) >= track.get("min_unique_roots", 2)
    )
    sus_frac = _sus_fraction(segments)
    if track.get("max_sus_fraction") is not None and sus_frac > track["max_sus_fraction"]:
        root_passed = False

    predicted_key = key_info.get("display")
    expected_key = track.get("expected_key")
    key_transpose = track.get(
        "key_transpose_invariant",
        track.get("transpose_invariant", False),
    )
    allow_relative = track.get("allow_relative_key", True)

    result = {
        "id": track["id"],
        "title": track.get("title"),
        "skipped": False,
        "passed": root_passed,
        "segments": len(segments),
        "unique_roots": sorted(unique),
        "root_recall": round(recall, 3),
        "sus_fraction": round(sus_frac, 3),
        "top_chords": Counter(seg["chord"] for seg in segments).most_common(8),
        "key": predicted_key,
    }

    expected_symbols = track.get("expected_symbols")
    if expected_symbols:
        symbols = _segment_symbols(segments)
        sym_recall = symbol_recall(
            symbols,
            expected_symbols,
            tolerant=track.get("symbol_match_power_as_triad", True),
        )
        result["symbol_recall"] = round(sym_recall, 3)
        if track.get("min_symbol_recall") is not None and sym_recall < track["min_symbol_recall"]:
            root_passed = False
            result["passed"] = False

    reference_changes = track.get("reference_changes")
    reference_timeline = track.get("reference_timeline")
    boundary_method = track.get("boundary_method")
    stamp_source = None

    if reference_changes and using_stamps:
        stamp_source = "fixture"
    elif reference_changes:
        stamp_source = "live"

    if not reference_changes and track.get("progression"):
        reference_changes, reference_timeline, boundary_method = resolve_boundary_references(
            y, sr, track,
        )
        stamp_source = "live"

    if reference_changes:
        reference_changes = filter_reference_changes(
            reference_changes,
            eval_start=track.get("boundary_eval_start"),
            eval_end=track.get("boundary_eval_end"),
        )
    if reference_timeline:
        reference_timeline = filter_reference_timeline(
            reference_timeline,
            eval_start=track.get("boundary_eval_start"),
            eval_end=track.get("boundary_eval_end"),
        )

    if reference_changes:
        boundary = boundary_alignment_score(
            segments,
            reference_changes,
            tolerance_sec=track.get("boundary_tolerance", 0.5),
        )
        if boundary is not None:
            result["boundary_score"] = round(boundary, 3)
            if track.get("min_boundary_score") is not None and boundary < track["min_boundary_score"]:
                root_passed = False

        switch_mae = switch_timing_mae(
            segments,
            reference_changes,
            tolerance_sec=track.get("boundary_tolerance", 0.5),
        )
        if switch_mae is not None:
            result["switch_mae_sec"] = round(switch_mae, 3)
        if boundary_method:
            result["boundary_method"] = boundary_method
        if stamp_source:
            result["stamp_source"] = stamp_source

    if reference_timeline:
        timeline = timeline_agreement_score(
            segments,
            reference_timeline,
            tolerant=track.get("symbol_match_power_as_triad", True),
            max_offset_sec=track.get("timeline_max_offset", 2.0),
            eval_start=track.get("boundary_eval_start"),
            eval_end=track.get("boundary_eval_end"),
        )
        if timeline is not None:
            result["timeline_score"] = round(timeline, 3)
            if track.get("min_timeline_score") is not None and timeline < track["min_timeline_score"]:
                root_passed = False

    result["passed"] = root_passed

    if expected_key:
        result["key_expected"] = expected_key
        result["key_match"] = _key_labels_match(
            expected_key,
            predicted_key or "",
            transpose_invariant=key_transpose,
            allow_relative=allow_relative,
        )
        result["key_skipped"] = False
    else:
        result["key_skipped"] = True

    return result


def _print_root_line(track, result):
    status = "PASS" if result["passed"] else "FAIL"
    sus_note = f" sus={result['sus_fraction']}"
    key_note = ""
    if result.get("key_expected") and result.get("key"):
        marker = "✓" if result.get("key_match") else "✗"
        key_note = f" [key {marker}: expected {result['key_expected']}, got {result['key']}]"
    source = track.get("source", "")
    src_note = f"\n  ref:   {source}" if source else ""
    print(
        f"{status} {track['id']}: recall={result['root_recall']} "
        f"segments={result['segments']} key={result['key']}{sus_note}{key_note}"
    )
    if result.get("symbol_recall") is not None:
        print(f"  symbol_recall={result['symbol_recall']}")
    if result.get("boundary_score") is not None:
        print(f"  boundary_score={result['boundary_score']}")
    if result.get("timeline_score") is not None:
        print(f"  timeline_score={result['timeline_score']}")
    if result.get("switch_mae_sec") is not None:
        print(f"  switch_mae_sec={result['switch_mae_sec']}")
    print(f"  roots: {result['unique_roots']}")
    print(f"  top:   {result['top_chords']}{src_note}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChordLift chord detection")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit with failure if any fixture audio file is missing",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated fixture ids to run (default: all tracks in fixtures file)",
    )
    parser.add_argument(
        "--write-results",
        type=Path,
        metavar="PATH",
        help="Write JSON results for eval_key_labels.py (avoids re-running ML in CI)",
    )
    parser.add_argument(
        "--live-boundaries",
        action="store_true",
        help="Recompute reference chord stamps from audio instead of chord_stamp_refs.json",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    refs = json.loads(args.fixtures.read_text())
    tracks = refs["tracks"]
    overlays = load_all_overlays()
    stamps = {} if args.live_boundaries else load_chord_stamps()
    if not args.live_boundaries and not stamps:
        print("Note: chord_stamp_refs.json not found — boundary eval uses live alignment\n")
    if args.ids:
        wanted = {tid.strip() for tid in args.ids.split(",") if tid.strip()}
        tracks = [t for t in tracks if t["id"] in wanted]
        missing_ids = wanted - {t["id"] for t in tracks}
        if missing_ids:
            print(f"Unknown fixture id(s): {', '.join(sorted(missing_ids))}", file=sys.stderr)
            sys.exit(1)

    print(f"Analyzer v{ANALYZER_VERSION} | engine={CHORD_ENGINE}\n")
    print("=== Root recall (CI gate) ===\n")

    results = []
    for track in tracks:
        result = evaluate_track(
            track, base_dir, overlays, stamps, live_boundaries=args.live_boundaries,
        )
        results.append(result)
        if result.get("skipped"):
            print(f"SKIP {result['id']}: {result['reason']}")
            continue
        _print_root_line(track, result)

    skipped = [r for r in results if r.get("skipped")]
    if args.require_all and skipped:
        print(f"ERROR: {len(skipped)} track(s) missing audio (run scripts/download_eval_tracks.py)")
        sys.exit(1)

    root_failed = [r for r in results if not r.get("skipped") and not r.get("passed")]
    root_passed = [r for r in results if not r.get("skipped") and r.get("passed")]
    print(
        f"Root recall: {len(root_passed)}/{len(root_passed) + len(root_failed)} passed"
    )

    symbol_tracks = [
        r for r in results
        if not r.get("skipped") and r.get("symbol_recall") is not None
    ]
    if symbol_tracks:
        avg_sym = sum(r["symbol_recall"] for r in symbol_tracks) / len(symbol_tracks)
        print(
            f"Symbol recall (informational): "
            f"{avg_sym:.1%} avg over {len(symbol_tracks)} track(s) with expected_symbols"
        )

    boundary_tracks = [
        r for r in results
        if not r.get("skipped") and r.get("boundary_score") is not None
    ]
    if boundary_tracks:
        avg_bnd = sum(r["boundary_score"] for r in boundary_tracks) / len(boundary_tracks)
        print(
            f"Boundary score (informational): "
            f"{avg_bnd:.1%} avg over {len(boundary_tracks)} track(s) with progression"
        )

    timeline_tracks = [
        r for r in results
        if not r.get("skipped") and r.get("timeline_score") is not None
    ]
    if timeline_tracks:
        avg_tl = sum(r["timeline_score"] for r in timeline_tracks) / len(timeline_tracks)
        mae_tracks = [r for r in timeline_tracks if r.get("switch_mae_sec") is not None]
        print(
            f"Timeline agreement (informational): "
            f"{avg_tl:.1%} avg over {len(timeline_tracks)} track(s) with progression"
        )
        if mae_tracks:
            avg_mae = sum(r["switch_mae_sec"] for r in mae_tracks) / len(mae_tracks)
            print(
                f"Switch timing MAE (informational): "
                f"{avg_mae:.2f}s avg over {len(mae_tracks)} track(s)"
            )

    if args.write_results:
        payload = {
            "analyzer_version": ANALYZER_VERSION,
            "chord_engine": CHORD_ENGINE,
            "fixtures": str(args.fixtures),
            "results": results,
        }
        args.write_results.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Results written to {args.write_results}")

    if root_failed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
