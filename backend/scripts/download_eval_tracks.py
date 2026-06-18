#!/usr/bin/env python3
"""Download audio for eval fixture JSON files (local dev / non-gold fixtures).

Gold v2 CI uses scripts/fetch_gold_audio_bundle.py — not this script.
"""
import asyncio
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloader import download_audio

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
SKIP_FIXTURES = frozenset({"gold_audio_identity.json", "gold_split_v2.json"})


def _track_source_url(track):
    yt_id = track.get("youtube_id")
    if yt_id:
        return f"https://www.youtube.com/watch?v={yt_id}"
    if track.get("url"):
        return track["url"]
    file_path = track.get("file", "")
    spotify_match = re.match(r"downloads/spotify-([A-Za-z0-9]+)\.mp3$", file_path)
    if spotify_match:
        return f"https://open.spotify.com/track/{spotify_match.group(1)}"
    yt_match = re.match(r"downloads/([A-Za-z0-9_-]{11})\.mp3$", file_path)
    if yt_match:
        return f"https://www.youtube.com/watch?v={yt_match.group(1)}"
    return None


def _collect_tracks():
    tracks_by_id: dict[str, dict] = {}
    for name in sorted(os.listdir(FIXTURES_DIR)):
        if not name.endswith(".json") or name in SKIP_FIXTURES:
            continue
        path = os.path.join(FIXTURES_DIR, name)
        data = json.loads(open(path, encoding="utf-8").read())
        raw = data.get("tracks", [])
        if isinstance(raw, dict):
            items = [{"id": tid, **track} for tid, track in raw.items()]
        else:
            items = raw
        for track in items:
            if not isinstance(track, dict):
                continue
            tid = track.get("id")
            if tid:
                tracks_by_id[tid] = track
    return list(tracks_by_id.values())


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download eval fixture audio")
    parser.add_argument("--force-ids", help="Comma-separated track ids to re-download")
    args = parser.parse_args()
    force = {x.strip() for x in (args.force_ids or "").split(",") if x.strip()}

    os.chdir(BACKEND_DIR)
    os.makedirs("downloads", exist_ok=True)

    missing = []
    for track in _collect_tracks():
        rel_path = track.get("file", "")
        if not rel_path:
            continue
        if os.path.exists(rel_path) and track["id"] not in force:
            print(f"✓ {track['id']}")
            continue
        source = _track_source_url(track)
        if not source:
            print(f"✗ {track['id']}: no source URL for {rel_path}", file=sys.stderr)
            missing.append(track["id"])
            continue
        print(f"↓ {track['id']} ({track.get('title', '')})")
        try:
            result = await download_audio(source)
            dest = track.get("file")
            if dest and result.get("audio_path") and result["audio_path"] != dest:
                shutil.copy2(result["audio_path"], dest)
                print(f"  → {dest} (from {result['audio_path']})")
            else:
                print(f"  → {result['audio_path']}")
        except Exception as exc:
            print(f"  ✗ failed: {exc}", file=sys.stderr)
            missing.append(track["id"])

    if missing:
        print(f"\nFailed or unresolved: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("\nAll eval tracks ready.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
