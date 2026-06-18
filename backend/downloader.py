import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from typing import List, Dict, Optional
from urllib.parse import urlparse

import httpx
import yt_dlp

DOWNLOAD_DIR = "downloads"
UPLOAD_DIR = "uploads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "")
PIPED_INSTANCES = [
    inst.strip()
    for inst in os.getenv(
        "PIPED_INSTANCES",
        "https://pipedapi.kavin.rocks,https://pipedapi.in.projectsegfau.lt,https://pipedapi.leptons.xyz",
    ).split(",")
    if inst.strip()
]
INVIDIOUS_INSTANCES = [
    inst.strip()
    for inst in os.getenv(
        "INVIDIOUS_INSTANCES",
        "https://inv.nadeko.net,https://invidious.jing.rocks,https://yewtu.be",
    ).split(",")
    if inst.strip()
]

YOUTUBE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
LOCAL_URL_PREFIX = "chordlift://local/"

logger = logging.getLogger(__name__)


def _bitrate_value(stream: dict) -> int:
    try:
        return int(stream.get("bitrate") or 0)
    except (TypeError, ValueError):
        return 0


def is_local_source(url: str) -> bool:
    return url.startswith(LOCAL_URL_PREFIX)


def local_source_id(url: str) -> str:
    from safe_paths import validate_source_id

    source_id = url[len(LOCAL_URL_PREFIX):]
    return validate_source_id(source_id)


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(
        r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([a-zA-Z0-9_-]{11})",
        url,
    )
    if match and YOUTUBE_ID_RE.match(match.group(1)):
        return match.group(1)
    return None


def extract_spotify_id(url: str) -> Optional[str]:
    match = re.search(r"open\.spotify\.com/(?:track|album|playlist)/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None


def extract_source_id(url: str) -> str:
    if is_local_source(url):
        return local_source_id(url)

    yt_id = extract_youtube_id(url)
    if yt_id:
        return yt_id

    spotify_id = extract_spotify_id(url)
    if spotify_id:
        return f"spotify-{spotify_id}"

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host:
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        slug = re.sub(r"[^a-z0-9]+", "-", host.split(".")[0].lower()).strip("-") or "link"
        return f"{slug}-{digest}"

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:12]
        except Exception:
            return hashlib.sha1(url.encode()).hexdigest()[:12]


def _detect_browsers():
    import platform

    system = platform.system()
    if system == "Darwin":
        return ["chrome", "safari", "firefox", "brave", "edge"]
    if system == "Windows":
        return ["chrome", "edge", "firefox", "brave"]
    return ["chrome", "firefox", "brave", "chromium"]


def _base_ytdlp_opts(output_template: str) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    return opts


def _youtube_strategies(output_template: str) -> List[Dict]:
    base = _base_ytdlp_opts(output_template)
    strategies = []

    # Current recommended clients (Oct 2025+). Works without browser cookies.
    strategies.append({
        **base,
        "extractor_args": {
            "youtube": {
                "player_client": ["web_safari", "tv", "android_vr"],
                "player_js_version": "actual",
            }
        },
        "_label": "web_safari+tv+android_vr",
    })

    strategies.append({
        **base,
        "extractor_args": {
            "youtube": {
                "player_client": ["tv_embedded", "web_creator"],
                "player_js_version": "actual",
            }
        },
        "_label": "tv_embedded+web_creator",
    })

    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        strategies.append({
            **base,
            "extractor_args": {
                "youtube": {
                    "player_client": ["default", "web_safari"],
                    "player_js_version": "actual",
                }
            },
            "_label": "cookies_file+web_safari",
        })

    for browser in _detect_browsers():
        strategies.append({
            **base,
            "cookiesfrombrowser": (browser,),
            "extractor_args": {
                "youtube": {
                    "player_client": ["web_safari", "tv", "android_vr"],
                    "player_js_version": "actual",
                }
            },
            "_label": f"cookies({browser})+web_safari",
        })

    return strategies


def _run_ytdlp(url: str, output_template: str, strategies: List[Dict]) -> None:
    last_error = None
    for strategy in strategies:
        label = strategy.pop("_label", "unknown")
        try:
            logger.info(f"Trying download strategy: {label}")
            with yt_dlp.YoutubeDL(strategy) as ydl:
                ydl.download([url])
            return
        except Exception as exc:
            last_error = exc
            logger.info(f"Strategy '{label}' failed: {exc}")
    raise RuntimeError(f"All yt-dlp strategies failed. Last error: {last_error}")


def _ffmpeg_to_mp3(input_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")


def _download_stream_to_mp3(stream_url: str, output_path: str) -> None:
    temp_path = output_path.replace(".mp3", ".stream")
    with httpx.stream("GET", stream_url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    try:
        _ffmpeg_to_mp3(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _download_youtube_via_piped(video_id: str, output_path: str) -> bool:
    for instance in PIPED_INSTANCES:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{instance.rstrip('/')}/streams/{video_id}")
                if response.status_code != 200:
                    continue
                payload = response.json()
            streams = payload.get("audioStreams") or []
            if not streams:
                continue
            best = sorted(streams, key=_bitrate_value, reverse=True)[0]
            stream_url = best.get("url")
            if not stream_url:
                continue
            logger.info("Piped download via %s", instance)
            _download_stream_to_mp3(stream_url, output_path)
            return True
        except Exception as exc:
            logger.warning("Piped instance %s failed: %s", instance, exc)
    return False


def _download_youtube_via_invidious(video_id: str, output_path: str) -> bool:
    for instance in INVIDIOUS_INSTANCES:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{instance.rstrip('/')}/api/v1/videos/{video_id}")
                if response.status_code != 200:
                    continue
                payload = response.json()
            formats = [
                fmt for fmt in (payload.get("adaptiveFormats") or [])
                if fmt.get("type", "").startswith("audio/")
            ]
            if not formats:
                continue
            best = sorted(formats, key=_bitrate_value, reverse=True)[0]
            stream_url = best.get("url")
            if not stream_url:
                continue
            logger.info("Invidious download via %s", instance)
            _download_stream_to_mp3(stream_url, output_path)
            return True
        except Exception as exc:
            logger.warning("Invidious instance %s failed: %s", instance, exc)
    return False


def _clean_spotify_title(title: str) -> str:
    for suffix in (
        " - Remastered 2009",
        " - Remastered",
        " - Radio Edit",
        " - Single Version",
        " - Album Version",
    ):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
    return title.strip()


def _spotify_youtube_query(title: str) -> str:
    """Build a YouTube search query from a Spotify track title."""
    title = _clean_spotify_title(title)
    known = {
        "Viva La Vida": "Coldplay Viva La Vida lyrics",
        "Let It Be": "The Beatles Let It Be official audio",
        "Hey Jude": "The Beatles Hey Jude official audio",
        "Yellow Submarine": "The Beatles Yellow Submarine official audio",
        "Wonderwall": "Oasis Wonderwall official audio",
        "With or Without You": "U2 With or Without You official audio",
        "Someone Like You": "Adele Someone Like You official audio",
        "Stand By Me": "Ben E King Stand By Me official audio",
        "Counting Stars": "OneRepublic Counting Stars official audio",
        "Ho Hey": "The Lumineers Ho Hey official audio",
        "Creep": "Radiohead Creep official audio",
        "Mr. Brightside": "The Killers Mr Brightside official audio",
        "Boulevard of Broken Dreams": "Green Day Boulevard of Broken Dreams official audio",
        "Shape of You": "Ed Sheeran Shape of You official audio",
        "bad guy": "Billie Eilish bad guy official audio",
        "Hotel California - 2013 Remaster": "Eagles Hotel California official audio",
        "Hotel California": "Eagles Hotel California official audio",
        "Sweet Child O' Mine": "Guns N Roses Sweet Child O Mine official audio",
        "Thinking out Loud": "Ed Sheeran Thinking Out Loud official audio",
        "Radioactive": "Imagine Dragons Radioactive official audio",
        "Hallelujah": "Leonard Cohen Hallelujah official audio",
        "Get Lucky (feat. Pharrell Williams and Nile Rodgers)": "Daft Punk Get Lucky official audio",
        "Smells Like Teen Spirit": "Nirvana Smells Like Teen Spirit official audio",
        "Imagine": "John Lennon Imagine official audio",
        "Rolling in the Deep": "Adele Rolling in the Deep official audio",
        "Love Story": "Taylor Swift Love Story official audio",
        "Don't Stop Believin'": "Journey Dont Stop Believin official audio",
        "Africa": "Toto Africa official audio",
        "Seven Nation Army": "White Stripes Seven Nation Army official audio",
        "Demons": "Imagine Dragons Demons official audio",
        "Iris": "Goo Goo Dolls Iris official audio",
        "Zombie": "Cranberries Zombie official audio",
    }
    search = known.get(title, f"{title} official audio")
    return f"ytsearch1:{search}"


# Fallback when Spotify oembed is unavailable (common for some track IDs/regions).
SPOTIFY_ID_SEARCH = {
    "5ghIIDpPgdJcK7B5X6JDlF": "Nirvana Smells Like Teen Spirit official audio",
    "3EJS5LyBJHfyR7lFls1e38": "John Lennon Imagine official audio",
    "1c8gkexLLshvpDHY6npDE7": "Adele Rolling in the Deep official audio",
    "3CeCwYWvdfXbZLl4KB3Cnr": "Taylor Swift Love Story official audio",
    "77NNZQSxLN5jjLw2G2eh69": "Journey Dont Stop Believin official audio",
    "7FRW07ElWFdbC5nTr2U8iO": "Toto Africa official audio",
    "2IH8tNQAz0QZDAe2U8Q0Hf": "White Stripes Seven Nation Army official audio",
    "0sW7NKeUXGuXBluxZmSJVW": "Imagine Dragons Demons official audio",
    "5oA6Ta6bZEVmeacEp8DnbS": "Goo Goo Dolls Iris official audio",
    "6Qyc6fS7Ds6B4k4h1sH9Le": "Cranberries Zombie official audio",
}


def _download_spotify_via_spotdl(url: str, expected_path: str) -> None:
    """YT-Music metadata-matched download + Spotify duration verification."""
    import shutil
    import subprocess
    import sys
    import tempfile

    import librosa

    from spotify_metadata import verify_audio_matches_spotify

    tmp = tempfile.mkdtemp(prefix="spotdl_")
    try:
        subprocess.run(
            [sys.executable, "-m", "spotdl", "download", url, "--output", tmp, "--format", "mp3"],
            check=True,
        )
        mp3s = [f for f in os.listdir(tmp) if f.endswith(".mp3")]
        if not mp3s:
            raise RuntimeError("spotdl produced no mp3")
        src = os.path.join(tmp, mp3s[0])
        duration = float(librosa.get_duration(path=src))
        check = verify_audio_matches_spotify(url, duration, tolerance_sec=1.0)
        if check.get("spotify_duration_status") == "fail":
            raise RuntimeError(
                f"Downloaded audio {duration:.1f}s does not match Spotify official "
                f"{check.get('spotify_duration_sec')}s (Δ={check.get('spotify_delta_sec'):.2f}s)"
            )
        shutil.copy2(src, expected_path)
        if check.get("spotify_duration_status") == "pass":
            logger.info(
                f"Spotify duration verified: {duration:.1f}s "
                f"(official {check.get('spotify_duration_sec')}s)"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve_spotify_to_youtube_search(url: str) -> str:
    from spotify_metadata import extract_spotify_track_id, fetch_spotify_track

    spotify_id = extract_spotify_id(url)
    if spotify_id:
        meta = fetch_spotify_track(spotify_id)
        if meta:
            title = (meta.get("name") or "").strip()
            artists = meta.get("artists") or []
            if artists and artists[0].get("name"):
                title = f"{artists[0]['name']} {title}"
            if title:
                query = _spotify_youtube_query(title)
                logger.info("Spotify API resolved to search: %s", query)
                return query

    if spotify_id and spotify_id in SPOTIFY_ID_SEARCH:
        query = f"ytsearch1:{SPOTIFY_ID_SEARCH[spotify_id]}"
        logger.info(f"Spotify ID fallback search: {query}")
        return query

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                "https://open.spotify.com/oembed",
                params={"url": url},
            )
            response.raise_for_status()
            title = response.json().get("title", "")
    except Exception as exc:
        if spotify_id and spotify_id in SPOTIFY_ID_SEARCH:
            query = f"ytsearch1:{SPOTIFY_ID_SEARCH[spotify_id]}"
            logger.info(f"Spotify oembed failed ({exc}); using ID fallback: {query}")
            return query
        raise RuntimeError(f"Could not resolve Spotify track: {exc}") from exc

    if not title:
        raise RuntimeError("Could not resolve Spotify track title.")

    query = _spotify_youtube_query(title)
    logger.info(f"Spotify resolved to search: {query}")
    return query


def _fetch_oembed_metadata(oembed_url: str, params: dict) -> dict:
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        response = client.get(oembed_url, params=params)
        response.raise_for_status()
        data = response.json()

    title = (data.get("title") or "").strip()
    artist = (data.get("author_name") or "").strip()
    thumbnail = data.get("thumbnail_url")

    return {
        "title": title or "Unknown track",
        "artist": artist,
        "album_art_url": thumbnail,
    }


def fetch_track_metadata(url: str) -> dict:
    """Fetch display metadata (title, artist, album art) for a source URL."""
    if is_local_source(url):
        source_id = local_source_id(url)
        meta_path = os.path.join(DOWNLOAD_DIR, f"{source_id}.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"title": "Uploaded track", "artist": "", "album_art_url": None}

    spotify_id = extract_spotify_id(url)
    if spotify_id:
        from spotify_metadata import fetch_spotify_display_meta

        official = fetch_spotify_display_meta(url)
        if official:
            return official
        try:
            return _fetch_oembed_metadata(
                "https://open.spotify.com/oembed",
                {"url": url},
            )
        except Exception as exc:
            logger.info(f"Spotify metadata failed: {exc}")

    yt_id = extract_youtube_id(url)
    if yt_id:
        try:
            return _fetch_oembed_metadata(
                "https://www.youtube.com/oembed",
                {"url": f"https://www.youtube.com/watch?v={yt_id}", "format": "json"},
            )
        except Exception as exc:
            logger.info(f"YouTube oembed failed: {exc}")

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title") or "Unknown track",
            "artist": info.get("uploader") or info.get("channel") or "",
            "album_art_url": info.get("thumbnail"),
        }
    except Exception as exc:
        logger.info(f"yt-dlp metadata fallback failed: {exc}")

    return {"title": "Unknown track", "artist": "", "album_art_url": None}


def save_uploaded_file(file_bytes: bytes, original_name: str) -> dict:
    source_id = f"upload-{uuid.uuid4().hex[:12]}"
    ext = os.path.splitext(original_name)[1].lower() or ".mp3"
    raw_path = os.path.join(UPLOAD_DIR, f"{source_id}{ext}")
    mp3_path = os.path.join(DOWNLOAD_DIR, f"{source_id}.mp3")

    with open(raw_path, "wb") as handle:
        handle.write(file_bytes)

    if ext == ".mp3":
        shutil.copy(raw_path, mp3_path)
    else:
        _ffmpeg_to_mp3(raw_path, mp3_path)

    title = os.path.splitext(original_name)[0]
    meta = {"title": title, "artist": "", "album_art_url": None}
    with open(os.path.join(DOWNLOAD_DIR, f"{source_id}.meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)

    return {
        "source_id": source_id,
        "url": f"{LOCAL_URL_PREFIX}{source_id}",
        "audio_path": mp3_path,
        "title": title,
        "artist": "",
        "album_art_url": None,
    }


async def download_audio(url: str) -> dict:
    import asyncio

    if is_local_source(url):
        source_id = local_source_id(url)
        mp3_path = os.path.join(DOWNLOAD_DIR, f"{source_id}.mp3")
        if not os.path.exists(mp3_path):
            raise FileNotFoundError("Uploaded audio file not found. Please upload again.")
        return {"video_id": source_id, "audio_path": mp3_path, "source": "upload"}

    source_id = extract_source_id(url)
    output_template = os.path.join(DOWNLOAD_DIR, f"{source_id}.%(ext)s")
    expected_path = os.path.join(DOWNLOAD_DIR, f"{source_id}.mp3")

    if os.path.exists(expected_path):
        return {"video_id": source_id, "audio_path": expected_path, "source": "cache"}

    def _download():
        spotify_id = extract_spotify_id(url)
        yt_id = extract_youtube_id(url)

        if spotify_id and not yt_id:
            try:
                logger.info(f"Spotify link → spotdl YT-Music match: {url}")
                _download_spotify_via_spotdl(url, expected_path)
                return
            except Exception as spotdl_err:
                logger.info(f"spotdl failed ({spotdl_err}); falling back to ytsearch")

        download_url = url
        if spotify_id and not yt_id:
            download_url = _resolve_spotify_to_youtube_search(url)

        if yt_id or spotify_id or "youtube.com" in url or "youtu.be" in url:
            strategies = _youtube_strategies(output_template)
            try:
                _run_ytdlp(download_url, output_template, strategies)
            except Exception as primary_error:
                if yt_id:
                    logger.info(f"yt-dlp failed, trying proxy fallbacks: {primary_error}")
                    if _download_youtube_via_piped(yt_id, expected_path):
                        return
                    if _download_youtube_via_invidious(yt_id, expected_path):
                        return
                raise primary_error
        else:
            generic = {**_base_ytdlp_opts(output_template), "_label": "generic"}
            _run_ytdlp(download_url, output_template, [generic])

    await asyncio.to_thread(_download)

    if os.path.exists(expected_path):
        return {"video_id": source_id, "audio_path": expected_path, "source": "download"}

    raise RuntimeError("Audio file was not created. Ensure ffmpeg is installed.")


# Backwards-compatible alias used elsewhere in the project
extract_video_id = extract_source_id
