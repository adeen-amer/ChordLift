#!/usr/bin/env python3
"""Re-download gold audio via spotdl (YT Music metadata match) for Phase 11.5 Step 2."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import yt_dlp

BACKEND = Path(__file__).resolve().parent.parent
CATALOG = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
OVERRIDES = BACKEND / "tests" / "fixtures" / "gold_youtube_overrides.json"
TMP = BACKEND / "downloads" / "_spotdl_tmp"

sys.path.insert(0, str(BACKEND))
from spotify_metadata import verify_audio_matches_spotify  # noqa: E402


def _load_overrides() -> dict[str, str]:
    if not OVERRIDES.is_file():
        return {}
    data = json.loads(OVERRIDES.read_text())
    return dict(data.get("overrides", {}))


def _download_youtube(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = dest.with_suffix("")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    produced = dest.with_suffix(".mp3")
    if not produced.is_file():
        mp3s = list(dest.parent.glob(f"{out.name}*.mp3"))
        if not mp3s:
            raise FileNotFoundError(f"no mp3 produced for {url}")
        produced = mp3s[0]
    if produced != dest:
        shutil.copy2(produced, dest)


def _download_spotdl(url: str, track_tmp: Path) -> Path:
    track_tmp.mkdir(parents=True, exist_ok=True)
    for old in track_tmp.glob("*.mp3"):
        old.unlink()
    subprocess.run(
        [sys.executable, "-m", "spotdl", "download", url, "--output", str(track_tmp), "--format", "mp3"],
        check=True,
        cwd=str(BACKEND),
    )
    mp3s = list(track_tmp.glob("*.mp3"))
    if not mp3s:
        raise FileNotFoundError("no mp3 produced")
    return mp3s[0]


def _verify_spotify(url: str, audio_path: Path) -> dict:
    duration = float(librosa.get_duration(path=str(audio_path)))
    return verify_audio_matches_spotify(url, duration, tolerance_sec=1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated track ids (default: all with spotify url)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text())
    overrides = _load_overrides()
    wanted = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    tracks = [t for t in catalog["tracks"] if t.get("url", "").startswith("https://open.spotify.com")]
    if wanted:
        tracks = [t for t in tracks if t["id"] in wanted]

    TMP.mkdir(parents=True, exist_ok=True)
    failed = []

    for track in tracks:
        tid = track["id"]
        dest = BACKEND / track["file"]
        url = track["url"]
        print(f"↓ {tid}")
        if args.dry_run:
            if tid in overrides:
                print(f"  override: {overrides[tid]}")
            continue
        try:
            if tid in overrides:
                print(f"  override YT: {overrides[tid]}")
                _download_youtube(overrides[tid], dest)
            else:
                track_tmp = TMP / tid
                src = _download_spotdl(url, track_tmp)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            check = _verify_spotify(url, dest)
            if check.get("spotify_duration_status") == "fail":
                raise RuntimeError(
                    f"Spotify duration mismatch: audio vs official "
                    f"{check.get('spotify_duration_sec')}s "
                    f"(Δ={check.get('spotify_delta_sec'):.2f}s)"
                )
            print(f"  → {dest} (spotify Δ={check.get('spotify_delta_sec', 0):.2f}s)")
        except Exception as exc:
            print(f"  ✗ {exc}", file=sys.stderr)
            failed.append(tid)

    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nDone: {len(tracks)} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
