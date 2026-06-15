#!/usr/bin/env python3
"""Download audio for all eval fixture tracks that are missing locally."""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloader import download_audio

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures")
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))


def _track_source_url(track):
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
    seen = set()
    tracks = []
    for name in sorted(os.listdir(FIXTURES_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(FIXTURES_DIR, name)
        data = json.loads(open(path, encoding="utf-8").read())
        for track in data.get("tracks", []):
            tid = track.get("id")
            if tid in seen:
                continue
            seen.add(tid)
            tracks.append(track)
    return tracks


async def main():
    os.chdir(BACKEND_DIR)
    os.makedirs("downloads", exist_ok=True)

    missing = []
    for track in _collect_tracks():
        rel_path = track.get("file", "")
        if not rel_path:
            continue
        if os.path.exists(rel_path):
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
