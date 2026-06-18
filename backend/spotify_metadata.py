"""Spotify official track metadata (duration) for identity verification."""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_RE = re.compile(
    r"open\.spotify\.com/(?:intl-[^/]+/)?track/([a-zA-Z0-9]+)",
)


def extract_spotify_track_id(url: str) -> str | None:
    match = SPOTIFY_TRACK_RE.search(url or "")
    return match.group(1) if match else None


@lru_cache(maxsize=256)
def fetch_spotify_duration_sec(track_id: str) -> float | None:
    """Official Spotify track duration in seconds (duration_ms precision when available)."""
    duration = _duration_via_spotipy(track_id)
    if duration is not None:
        return duration
    duration = _duration_via_spotipy_free(track_id)
    if duration is not None:
        return duration
    return _duration_via_ytmusic_search(track_id)


def _duration_via_spotipy(track_id: str) -> float | None:
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials

        sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            ),
        )
        meta = sp.track(track_id)
        ms = meta.get("duration_ms")
        return float(ms) / 1000.0 if ms else None
    except Exception as exc:
        logger.warning("Spotify API duration failed for %s: %s", track_id, exc)
        return None


def _duration_via_spotipy_free(track_id: str) -> float | None:
    """Tokenless Spotify metadata (bundled with spotdl)."""
    try:
        from SpotipyFree import Spotify as FreeSpotify

        meta = FreeSpotify().track(track_id)
        ms = meta.get("duration_ms")
        return float(ms) / 1000.0 if ms else None
    except Exception as exc:
        logger.warning("SpotipyFree duration failed for %s: %s", track_id, exc)
        return None


def _duration_via_ytmusic_search(track_id: str) -> float | None:
    """Last resort: YT Music search using Spotify oembed title (second precision)."""
    url = f"https://open.spotify.com/track/{track_id}"
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get("https://open.spotify.com/oembed", params={"url": url})
            resp.raise_for_status()
            title = (resp.json().get("title") or "").strip()
    except Exception as exc:
        logger.warning("Spotify oembed failed for %s: %s", track_id, exc)
        return None
    if not title:
        return None
    try:
        from ytmusicapi import YTMusic

        yt = YTMusic()
        results = yt.search(title, filter="songs", limit=5)
        for hit in results:
            dur = hit.get("duration_seconds")
            if dur and int(dur) > 0:
                return float(dur)
            length = hit.get("length")
            if length and ":" in str(length):
                parts = str(length).split(":")
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
        return None
    except Exception as exc:
        logger.warning("YTMusic duration fallback failed for %s: %s", track_id, exc)
        return None


def verify_audio_matches_spotify(url: str, audio_duration_sec: float, *, tolerance_sec: float = 1.0) -> dict:
    """Compare downloaded audio length to Spotify official duration."""
    track_id = extract_spotify_track_id(url)
    if not track_id:
        return {
            "spotify_track_id": None,
            "spotify_duration_sec": None,
            "spotify_delta_sec": None,
            "spotify_duration_status": "skip",
        }
    official = fetch_spotify_duration_sec(track_id)
    if official is None:
        return {
            "spotify_track_id": track_id,
            "spotify_duration_sec": None,
            "spotify_delta_sec": None,
            "spotify_duration_status": "unknown",
        }
    delta = abs(audio_duration_sec - official)
    status = "pass" if delta <= tolerance_sec else "fail"
    return {
        "spotify_track_id": track_id,
        "spotify_duration_sec": round(official, 3),
        "spotify_delta_sec": round(delta, 3),
        "spotify_duration_status": status,
    }
