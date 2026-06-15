#!/usr/bin/env python3
"""
Export pseudo chord labels from eval fixtures for future autochord fine-tuning.

Generates beat-aligned label files (start, end, autochord-style chord) per track
using progression overlays + audio beat tracking. Not ground truth — Hooktheory
loops approximated to the recording tempo.

Usage:
  python scripts/export_training_labels.py --out training_labels/
  python scripts/export_training_labels.py --ids let-it-be,get-lucky
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval_chord_utils import reference_changes_from_progression  # noqa: E402
from eval_fixture_utils import load_enrichment, merge_track_enrichment  # noqa: E402

FIXTURES = BACKEND / "tests" / "fixtures" / "chord_refs.json"


def _to_autochord_label(chord: str) -> str:
    """Map internal symbol to autochord F:maj / F:min style."""
    if not chord or chord == "N":
        return "N"
    if chord.endswith("m7"):
        root = chord[:-2]
        return f"{root}:min7"
    if chord.endswith("maj7"):
        root = chord[:-4]
        return f"{root}:maj7"
    if chord.endswith("7"):
        root = chord[:-1]
        return f"{root}:7"
    if chord.endswith("m"):
        root = chord[:-1]
        return f"{root}:min"
    if chord.endswith("5"):
        root = chord[:-1]
        return f"{root}:maj"
    return f"{chord}:maj"


def _changes_to_segments(changes: list[dict], duration: float) -> list[tuple[float, float, str]]:
    if not changes:
        return []
    segments = []
    for i, change in enumerate(changes):
        start = change["time"]
        end = changes[i + 1]["time"] if i + 1 < len(changes) else duration
        label = _to_autochord_label(change["chord"])
        segments.append((start, end, label))
    return segments


def export_track(track: dict, base_dir: Path, out_dir: Path) -> dict | None:
    audio_path = base_dir / track["file"]
    if not audio_path.exists():
        return None

    progression = track.get("progression")
    if not progression:
        return None

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr)

    changes = reference_changes_from_progression(
        y,
        sr,
        progression,
        beats_per_chord=track.get("beats_per_chord", 4),
        cycles=track.get("boundary_cycles", 16),
        skip_beats=track.get("boundary_skip_beats", 0),
    )
    if not changes:
        return None

    segments = _changes_to_segments(changes, duration)
    payload = {
        "id": track["id"],
        "title": track.get("title"),
        "source": track.get("source"),
        "audio_file": track["file"],
        "duration": duration,
        "progression": progression,
        "labels": [
            {"start": s, "end": e, "chord": c} for s, e, c in segments
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{track['id']}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"id": track["id"], "path": str(out_path), "segments": len(segments)}


def main():
    parser = argparse.ArgumentParser(description="Export training label JSON from fixture progressions")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, default=BACKEND / "training_labels")
    parser.add_argument("--ids", help="Comma-separated track ids (default: all with progression)")
    args = parser.parse_args()

    refs = json.loads(args.fixtures.read_text())
    overlays = load_enrichment()
    tracks = [merge_track_enrichment(t, overlays) for t in refs["tracks"]]

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        tracks = [t for t in tracks if t["id"] in wanted]

    base_dir = BACKEND
    exported = []
    skipped = 0
    for track in tracks:
        if not track.get("progression"):
            skipped += 1
            continue
        result = export_track(track, base_dir, args.out)
        if result:
            exported.append(result)
            print(f"  {result['id']}: {result['segments']} segments → {result['path']}")
        else:
            skipped += 1
            print(f"  SKIP {track['id']}: missing audio or beat track")

    print(f"\nExported {len(exported)} label file(s) to {args.out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
