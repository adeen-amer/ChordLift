#!/usr/bin/env python3
"""Extract Isophonics Harte chord .lab files into tests/fixtures/gold/lab/."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
OUT_DIR = BACKEND / "tests" / "fixtures" / "gold" / "lab"

ARCHIVE_URLS = {
    "beatles": "http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
    "queen": "http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
}


def _load_archive(name: str, local: Path | None) -> tarfile.TarFile:
    if local and local.is_file():
        return tarfile.open(local, "r:gz")
    url = ARCHIVE_URLS[name]
    tmp = Path(tempfile.gettempdir()) / f"isophonics_{name}.tar.gz"
    if not tmp.is_file():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tmp)
    return tarfile.open(tmp, "r:gz")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Isophonics gold chord labs")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--beatles-archive", type=Path, help="Local Beatles Annotations.tar.gz")
    parser.add_argument("--queen-archive", type=Path, help="Local Queen Annotations.tar.gz")
    parser.add_argument("--ids", help="Comma-separated track ids (default: all in catalog)")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    tracks = catalog["tracks"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        tracks = [t for t in tracks if t["id"] in wanted]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    open_archives: dict[str, tarfile.TarFile] = {}

    try:
        for track in tracks:
            archive = track.get("archive", "beatles")
            member = track["isophonics_path"]
            if archive not in open_archives:
                local = args.beatles_archive if archive == "beatles" else args.queen_archive
                open_archives[archive] = _load_archive(archive, local)

            tf = open_archives[archive]
            extracted = tf.extractfile(member)
            if extracted is None:
                print(f"MISSING {member}", file=sys.stderr)
                return 1
            dest = OUT_DIR / f"{track['id']}.lab"
            dest.write_bytes(extracted.read())
            print(f"✓ {dest.name}")
    finally:
        for tf in open_archives.values():
            tf.close()

    print(f"\nDone. {len(tracks)} labs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
