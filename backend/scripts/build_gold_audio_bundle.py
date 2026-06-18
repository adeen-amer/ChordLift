#!/usr/bin/env python3
"""Build gold_audio_v2.tar.gz from local identity-verified mp3s (run on your machine, not CI)."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from gold_audio_bundle import BUNDLE_NAME, RELEASE_TAG, build_bundle, manifest_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack gold eval audio for CI/release")
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND / BUNDLE_NAME,
        help=f"Output tarball (default: backend/{BUNDLE_NAME})",
    )
    parser.add_argument("--print-sha256", action="store_true", help="Print tarball sha256")
    args = parser.parse_args()

    fp = manifest_fingerprint()
    print(f"Manifest fingerprint: {fp}")
    meta = build_bundle(args.output)
    print(f"Wrote {args.output} ({meta['track_count']} tracks, {args.output.stat().st_size // 1024} KiB)")
    print(f"Upload to private release tag '{RELEASE_TAG}':")
    print(f"  gh release upload {RELEASE_TAG} {args.output} --clobber")
    if args.print_sha256:
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        print(f"sha256:{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
