#!/usr/bin/env python3
"""Scan a data dir for <stem>.(mp3|wav|m4a) + <stem>.lab pairs, write a
training manifest (`<abs_audio_path>\\t<abs_lab_path>` per line), then gate
it against gold-holdout leakage via scripts/verify_training_leakage.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = BACKEND / "scripts" / "verify_training_leakage.py"
AUDIO_EXTS = (".mp3", ".wav", ".m4a")


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for audio_path in sorted(data_dir.iterdir()):
        if audio_path.suffix.lower() not in AUDIO_EXTS:
            continue
        lab_path = audio_path.with_suffix(".lab")
        if lab_path.exists():
            pairs.append((audio_path.resolve(), lab_path.resolve()))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a chord training manifest")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-pairs", type=int, default=10)
    args = ap.parse_args()

    pairs = find_pairs(args.data_dir)
    lines = [f"{audio}\t{lab}" for audio, lab in pairs]
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""))

    leak_rc = subprocess.run([sys.executable, str(VERIFY_SCRIPT), str(args.out)]).returncode
    leak_status = "OK" if leak_rc == 0 else "LEAK"
    print(f"pairs={len(pairs)} leakage={leak_status}")

    if leak_rc != 0 or len(pairs) < args.min_pairs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
