#!/usr/bin/env python3
"""Fetch and extract cached gold eval audio for CI (no YouTube / spotdl)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from gold_audio_bundle import (  # noqa: E402
    BUNDLE_NAME,
    RELEASE_TAG,
    extract_bundle,
    manifest_fingerprint,
    verify_extracted_bundle,
)


def _github_repo() -> str:
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not repo:
        raise RuntimeError("GITHUB_REPOSITORY is not set")
    return repo


def _download_release_asset(dest: Path) -> None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to download the private gold audio release")

    repo = _github_repo()
    api = f"https://api.github.com/repos/{repo}/releases/tags/{RELEASE_TAG}"
    request = urllib.request.Request(
        api,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ChordLift-gold-audio-fetch",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            release = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"GitHub release '{RELEASE_TAG}' not found ({exc.code}). "
            f"Build locally with scripts/build_gold_audio_bundle.py and upload:\n"
            f"  gh release create {RELEASE_TAG} --title 'Gold eval audio v2' --notes 'CI bundle'\n"
            f"  gh release upload {RELEASE_TAG} backend/{BUNDLE_NAME} --clobber"
        ) from exc

    asset = next((a for a in release.get("assets", []) if a.get("name") == BUNDLE_NAME), None)
    if not asset:
        names = [a.get("name") for a in release.get("assets", [])]
        raise RuntimeError(
            f"Release '{RELEASE_TAG}' has no asset named {BUNDLE_NAME}. Found: {names}"
        )

    asset_api = asset["url"]
    request = urllib.request.Request(
        asset_api,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
            "User-Agent": "ChordLift-gold-audio-fetch",
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=600) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=BACKEND / BUNDLE_NAME,
        help="Where to store the downloaded tarball",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract/verify an existing archive (for local testing)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify extracted downloads/ against gold_audio_identity.json (no download)",
    )
    args = parser.parse_args()

    fp = manifest_fingerprint()
    print(f"Expected manifest fingerprint: {fp}")

    if args.verify_only:
        errors = verify_extracted_bundle()
        if errors:
            print("Gold audio verification failed:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print("Gold audio cache verified against identity manifest")
        return 0

    if not args.skip_download:
        print(f"Downloading {BUNDLE_NAME} from release {RELEASE_TAG}...")
        _download_release_asset(args.archive)

    if not args.archive.is_file():
        print(f"Missing archive: {args.archive}", file=sys.stderr)
        return 1

    print(f"Extracting {args.archive}...")
    meta = extract_bundle(args.archive)
    if meta.get("fingerprint") != fp:
        print(
            f"WARNING: bundle fingerprint {meta.get('fingerprint')} != repo {fp}",
            file=sys.stderr,
        )

    errors = verify_extracted_bundle(expected_fingerprint=meta.get("fingerprint"))
    if errors:
        print("Gold audio verification failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Gold audio ready: {meta.get('track_count')} tracks verified against identity manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
