"""Spotify official track metadata (duration) for identity verification."""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_RE = re.compile(
    r"open\.spotify\.com/(?:intl-[^/]+/)?track/([a-zA-Z0-9]+)",
)


def extract_spotify_track_id(url: str) -> str | None:
    match = SPOTIFY_TRACK_RE.search(url or "")
    return match.group(1) if match else None


def _spotify_credentials() -> tuple[str, str] | None:
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return client_id, client_secret
    return None


@lru_cache(maxsize=1)
def _spotipy_client():
    creds = _spotify_credentials()
    if not creds:
        return None
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id, client_secret = creds
    return spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        ),
    )


@lru_cache(maxsize=256)
def fetch_spotify_track(track_id: str) -> dict[str, Any] | None:
    """Official Spotify Web API track payload (requires client-credentials)."""
    sp = _spotipy_client()
    if sp is None:
        return None
    try:
        return sp.track(track_id)
    except Exception as exc:
        logger.warning("Spotify API track lookup failed for %s: %s", track_id, exc)
        return None


@lru_cache(maxsize=256)
def fetch_spotify_duration_sec(track_id: str) -> float | None:
    """Official Spotify track duration in seconds (duration_ms precision when available)."""
    meta = fetch_spotify_track(track_id)
    if meta:
        ms = meta.get("duration_ms")
        if ms:
            return float(ms) / 1000.0
    return _duration_via_oembed(track_id)


def fetch_spotify_display_meta(url: str) -> dict[str, str | None] | None:
    """Title/artist from official API; None when credentials are missing."""
    track_id = extract_spotify_track_id(url)
    if not track_id:
        return None
    meta = fetch_spotify_track(track_id)
    if not meta:
        return None
    artists = ", ".join(a["name"] for a in meta.get("artists", []) if a.get("name"))
    images = meta.get("album", {}).get("images") or []
    art = images[0]["url"] if images else None
    return {
        "title": (meta.get("name") or "").strip() or "Unknown track",
        "artist": artists,
        "album_art_url": art,
    }


def _duration_via_oembed(track_id: str) -> float | None:
    """Second-precision fallback when Spotify API credentials are not configured."""
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
    return _duration_via_ytmusic_search(title)


def _duration_via_ytmusic_search(title: str) -> float | None:
    """Last resort: YT Music search using a public title string (second precision)."""
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
        logger.warning("YTMusic duration fallback failed for %r: %s", title, exc)
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
