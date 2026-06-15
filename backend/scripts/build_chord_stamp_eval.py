#!/usr/bin/env python3
"""
Build chord_stamp_refs.json — timestamped reference chord changes for eval.

For each fixture track with a known progression, runs phase-aligned boundary
resolution on the eval audio and writes reference_changes + reference_timeline.
Eval loads these stamps by default so boundary/timeline scoring is stable and
inspectable without recomputing alignment on every run.

Usage:
  python scripts/build_chord_stamp_eval.py
  python scripts/build_chord_stamp_eval.py --ids let-it-be,get-lucky
  python scripts/build_chord_stamp_eval.py --check   # fail if file missing/stale ids
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from analyzer import ANALYZER_VERSION  # noqa: E402
from eval_chord_utils import resolve_boundary_references  # noqa: E402
from eval_fixture_utils import (  # noqa: E402
    DEFAULT_STAMP_REFS,
    load_all_overlays,
    merge_track_enrichment,
)

FIXTURES = BACKEND / "tests" / "fixtures" / "chord_refs.json"


def _track_stamp(track: dict, base_dir: Path) -> dict | None:
    progression = track.get("progression")
    if not progression:
        return None

    audio_path = base_dir / track["file"]
    if not audio_path.exists():
        return None

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr)
    changes, timeline, method = resolve_boundary_references(y, sr, track)

    if not changes:
        return None

    stamp: dict = {
        "title": track.get("title"),
        "source": track.get("source"),
        "progression": progression,
        "beats_per_chord": track.get("beats_per_chord", 4),
        "boundary_cycles": track.get("boundary_cycles", 8),
        "boundary_method": method,
        "duration_sec": round(duration, 3),
        "change_count": len(changes),
        "reference_changes": changes,
        "reference_timeline": timeline,
    }

    if track.get("boundary_eval_start") is not None or track.get("boundary_eval_end") is not None:
        stamp["eval_window"] = {
            "start": track.get("boundary_eval_start"),
            "end": track.get("boundary_eval_end"),
        }

    return stamp


def build_stamps(
    tracks: list[dict],
    base_dir: Path,
    overlays: dict,
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Return (stamps_by_id, exported_ids, skipped_ids)."""
    stamps: dict[str, dict] = {}
    exported: list[str] = []
    skipped: list[str] = []

    for track in tracks:
        merged = merge_track_enrichment(track, overlays)
        tid = merged["id"]
        if not merged.get("progression"):
            continue
        stamp = _track_stamp(merged, base_dir)
        if stamp:
            stamps[tid] = stamp
            exported.append(tid)
            print(f"  {tid}: {stamp['change_count']} changes ({stamp['boundary_method']})")
        else:
            skipped.append(tid)
            print(f"  SKIP {tid}: missing audio or alignment failed")

    return stamps, exported, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chord stamp eval fixtures")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_STAMP_REFS)
    parser.add_argument("--ids", help="Comma-separated track ids (default: all with progression)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if stamp file missing or expected track ids are absent",
    )
    args = parser.parse_args()

    refs = json.loads(args.fixtures.read_text())
    overlays = load_all_overlays()
    tracks = refs["tracks"]

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        tracks = [t for t in tracks if t["id"] in wanted]

    progression_ids = {
        merge_track_enrichment(t, overlays)["id"]
        for t in tracks
        if merge_track_enrichment(t, overlays).get("progression")
    }

    if args.check:
        if not args.out.exists():
            print(f"Missing stamp file: {args.out}", file=sys.stderr)
            return 1
        existing = json.loads(args.out.read_text()).get("tracks", {})
        missing = sorted(progression_ids - set(existing))
        if missing:
            print(f"Stamp file missing {len(missing)} track(s): {', '.join(missing)}", file=sys.stderr)
            return 1
        print(f"OK: {args.out} covers {len(progression_ids)} progression track(s)")
        return 0

    print(f"Building chord stamps (analyzer v{ANALYZER_VERSION})…\n")
    stamps, exported, skipped = build_stamps(tracks, BACKEND, overlays)

    if args.ids and args.out.exists():
        existing = json.loads(args.out.read_text()).get("tracks", {})
        existing.update(stamps)
        stamps = existing

    payload = {
        "description": (
            "Precomputed chord change stamps for boundary/timeline eval. "
            "Regenerate with scripts/build_chord_stamp_eval.py after progression "
            "or alignment changes."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analyzer_version": ANALYZER_VERSION,
        "track_count": len(stamps),
        "tracks": stamps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {len(exported)} track stamp(s) → {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
