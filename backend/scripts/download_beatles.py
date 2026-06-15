#!/usr/bin/env python3
"""Download Beatles eval tracks via Spotify → YouTube (same as app pipeline)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloader import download_audio, extract_source_id

BEATLES = [
    ("Let It Be", "https://open.spotify.com/track/5V1AHQugSTASVez5ffJtFo"),
    ("Hey Jude", "https://open.spotify.com/track/1eT2CjXwFXNx6oY5ydvzKU"),
    ("Yellow Submarine", "https://open.spotify.com/track/0Dk6z7iOJzFYywJvxjNltW"),
]


async def main():
    for title, url in BEATLES:
        sid = extract_source_id(url)
        path = f"downloads/{sid}.mp3"
        if os.path.exists(path):
            print(f"✓ {title} already at {path}")
            continue
        print(f"Downloading {title}...")
        try:
            result = await download_audio(url)
            print(f"  → {result['audio_path']}")
        except Exception as exc:
            print(f"  ✗ failed: {exc}", file=sys.stderr)
            return 1
    print("\nDone. Run: ./venv2/bin/python eval_beatles.py")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
