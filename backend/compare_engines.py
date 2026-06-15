#!/usr/bin/env python3
"""Compare classic vs ML chord engines on eval fixtures."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import librosa

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "chord_refs.json"


def run_engine(engine, audio_path):
    os.environ["CHORD_ENGINE"] = engine
    # Reload dispatcher each run
    import importlib
    import chord_engine
    importlib.reload(chord_engine)

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    return chord_engine.extract_chords(y, sr)


def summarize(segments):
    return Counter(seg["chord"] for seg in segments).most_common(6)


def main():
    base = Path(__file__).parent
    refs = json.loads(FIXTURES.read_text())
    tracks = refs["tracks"]

    print("Engine comparison (classic vs ml)\n")
    for track in tracks:
        path = base / track["file"]
        if not path.exists():
            print(f"SKIP {track['id']}: missing file")
            continue
        print(f"=== {track['id']} ===")
        for engine in ("classic", "ml"):
            try:
                segments, key = run_engine(engine, path)
                print(f"  {engine:8} key={key.get('display')} segs={len(segments)} top={summarize(segments)}")
            except Exception as exc:
                print(f"  {engine:8} ERROR: {exc}")
        print()


if __name__ == "__main__":
    main()
