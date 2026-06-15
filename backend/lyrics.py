"""Fetch synced lyrics and attach them to chord timeline segments."""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LYRICS_VERSION = "5"
LRCLIB_BASE = "https://lrclib.net/api"

LRC_TIMESTAMP = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]([^\n\r\[]*)")

TITLE_SUFFIX_RE = re.compile(
    r"\s*[\(\[](official|video|audio|lyric|hd|4k|remaster|visualizer)[^\)\]]*[\)\]]\s*$",
    re.I,
)


def split_artist_title(song: dict[str, Any] | None) -> tuple[str, str]:
    """Best-effort artist + title from song metadata."""
    if not song:
        return "", ""

    artist = (song.get("artist") or "").strip()
    title = (song.get("title") or "").strip()

    if artist and artist.lower() not in ("unknown", "unknown track"):
        return artist, TITLE_SUFFIX_RE.sub("", title).strip()

    if " - " in title:
        left, right = title.split(" - ", 1)
        left, right = left.strip(), right.strip()
        if left and right:
            return left, TITLE_SUFFIX_RE.sub("", right).strip()

    return artist, TITLE_SUFFIX_RE.sub("", title).strip()


def parse_lrc(content: str) -> list[dict[str, Any]]:
    """Parse LRC synced lyrics into [{time, text}, ...] (seconds)."""
    if not content:
        return []

    lines: list[dict[str, Any]] = []
    for match in LRC_TIMESTAMP.finditer(content):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac = match.group(3) or "0"
        text = (match.group(4) or "").strip()
        frac_val = int(frac) / (100.0 if len(frac) <= 2 else 1000.0)
        t = minutes * 60 + seconds + frac_val

        if text and not text.startswith("["):
            lines.append({"time": float(t), "text": text})

    lines.sort(key=lambda row: row["time"])
    return lines


def _parse_vtt(content: str) -> list[dict[str, Any]]:
    """Minimal WebVTT → synced lines."""
    lines: list[dict[str, Any]] = []
    block_time = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})",
    )
    current_time: float | None = None
    buffer: list[str] = []

    def flush():
        nonlocal buffer, current_time
        if current_time is not None and buffer:
            text = " ".join(buffer).strip()
            text = re.sub(r"<[^>]+>", "", text)
            if text:
                lines.append({"time": current_time, "text": text})
        buffer = []

    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if block_time.match(line):
            flush()
            m = block_time.match(line)
            assert m is not None
            current_time = (
                int(m.group(1)) * 3600
                + int(m.group(2)) * 60
                + int(m.group(3))
                + int(m.group(4)) / 1000.0
            )
            continue
        if line.isdigit():
            continue
        buffer.append(line)

    flush()
    lines.sort(key=lambda row: row["time"])
    return lines


def fetch_lrclib(
    artist: str,
    title: str,
    duration_sec: float | None = None,
) -> tuple[list[dict[str, Any]], str | None, str]:
    """
    Fetch synced or plain lyrics from LRCLIB.

    Returns (synced_lines, plain_text, source_label).
    """
    if not title:
        return [], None, ""

    params: dict[str, str | int] = {
        "track_name": title,
        "artist_name": artist or "",
    }
    if duration_sec and duration_sec > 1:
        params["duration"] = int(round(duration_sec))

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{LRCLIB_BASE}/get", params=params)
            if resp.status_code == 404:
                search = client.get(
                    f"{LRCLIB_BASE}/search",
                    params={"q": f"{artist} {title}".strip()},
                )
                if search.status_code == 200:
                    hits = search.json()
                    if hits:
                        best = hits[0]
                        resp = client.get(
                            f"{LRCLIB_BASE}/get",
                            params={
                                "track_name": best.get("trackName", title),
                                "artist_name": best.get("artistName", artist),
                                **(
                                    {"duration": int(round(duration_sec))}
                                    if duration_sec
                                    else {}
                                ),
                            },
                        )
            if resp.status_code != 200:
                return [], None, ""

            payload = resp.json()
            synced_raw = payload.get("syncedLyrics") or ""
            plain = payload.get("plainLyrics") or ""
            synced = parse_lrc(synced_raw) if synced_raw else []
            source = "lrclib"
            if synced:
                return synced, plain or None, source
            return [], plain or None, source
    except Exception as exc:
        logger.warning("LRCLIB fetch failed: %s", exc)
        return [], None, ""


def fetch_youtube_captions(youtube_id: str) -> list[dict[str, Any]]:
    """Try YouTube manual/auto captions via yt-dlp (VTT)."""
    if not youtube_id or len(youtube_id) != 11:
        return []

    try:
        import yt_dlp
    except ImportError:
        return []

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.debug("YouTube caption info failed: %s", exc)
        return []

    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        for lang in ("en", "en-US", "en-GB", "a.en"):
            for fmt in tracks.get(lang, tracks.get("en", [])):
                if fmt.get("ext") not in ("vtt", "srv3", "ttml"):
                    continue
                sub_url = fmt.get("url")
                if not sub_url:
                    continue
                try:
                    with httpx.Client(timeout=20.0) as client:
                        body = client.get(sub_url).text
                    parsed = _parse_vtt(body)
                    if parsed:
                        return parsed
                except Exception as exc:
                    logger.debug("Caption download failed: %s", exc)
        if tracks:
            break

    return []


def distribute_plain_lyrics(
    timeline: list[dict[str, Any]],
    plain_text: str,
) -> list[dict[str, Any]]:
    """Assign unsynced lyrics lines across chord segments in order."""
    lines = [
        ln.strip()
        for ln in plain_text.splitlines()
        if ln.strip() and not ln.strip().startswith("[")
    ]
    if not lines or not timeline:
        return timeline

    out = [dict(seg) for seg in timeline]
    seg_count = len(out)
    if seg_count == 0:
        return out

    per_seg = max(1, len(lines) // seg_count)
    idx = 0
    for i, seg in enumerate(out):
        if i == seg_count - 1:
            chunk = lines[idx:]
        else:
            chunk = lines[idx: idx + per_seg]
            idx += per_seg
        seg["lyrics"] = " ".join(chunk).strip()
    return out


def _with_line_end_times(lines: list[dict[str, Any]], duration_sec: float) -> list[dict[str, Any]]:
    """Add end_time to each synced line for playback scrubbing."""
    if not lines:
        return lines
    out = []
    for i, line in enumerate(lines):
        row = dict(line)
        if i + 1 < len(lines):
            row["end_time"] = float(lines[i + 1]["time"])
        else:
            row["end_time"] = min(float(row["time"]) + 6.0, duration_sec)
        out.append(row)
    return out


def align_synced_lyrics(
    timeline: list[dict[str, Any]],
    lyric_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map synced lyric lines into chord segments by timestamp overlap."""
    if not timeline:
        return timeline

    out = [dict(seg) for seg in timeline]
    if not lyric_lines:
        return out

    for seg in out:
        t0 = float(seg.get("time", 0))
        t1 = float(seg.get("end_time", t0 + 1.0))
        parts: list[str] = []
        for line in lyric_lines:
            t = float(line["time"])
            if t0 <= t < t1:
                parts.append(str(line["text"]))
        seg["lyrics"] = " ".join(parts).strip()

    return out


def attach_lyrics_to_timeline(
    timeline: list[dict[str, Any]],
    song: dict[str, Any] | None,
    duration_sec: float,
    *,
    youtube_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Fetch lyrics and attach a ``lyrics`` string to each chord segment.

    Returns (updated_timeline, lyrics_meta).
    """
    meta: dict[str, Any] = {"lyrics_version": LYRICS_VERSION, "source": None, "synced": False}

    if not timeline:
        return timeline, meta

    artist, title = split_artist_title(song)
    synced, plain, source = fetch_lrclib(artist, title, duration_sec)

    if not synced and youtube_id:
        synced = fetch_youtube_captions(youtube_id)
        if synced:
            source = "youtube_captions"

    if synced:
        synced = _with_line_end_times(synced, duration_sec)
        meta["source"] = source
        meta["synced"] = True
        meta["lines"] = synced
        timeline = align_synced_lyrics(timeline, synced)
        return timeline, meta

    if plain:
        meta["source"] = source or "lrclib_plain"
        meta["synced"] = False
        meta["plain_text"] = plain
        return distribute_plain_lyrics(timeline, plain), meta

    return [dict(seg) for seg in timeline], meta
