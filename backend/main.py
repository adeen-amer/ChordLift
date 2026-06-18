from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
import logging
from pydantic import BaseModel

from downloader import (
    download_audio,
    extract_source_id,
    extract_youtube_id,
    fetch_track_metadata,
    save_uploaded_file,
)
from analyzer import analyze_audio, is_cache_valid, ANALYZER_VERSION
from chord_engine import CHORD_ENGINE, log_chord_engine_status
from ml_env import summarize_ml_readiness
from safe_paths import InvalidSourceIdError, resolve_cache_json, resolve_download_mp3, validate_source_id
from analysis_runtime import run_analysis_deduped
from model_cache import preload_ml_models

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

ALLOWED_UPLOAD_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}

app = FastAPI(title="ChordLift API")

@app.on_event("startup")
async def startup():
    log_chord_engine_status()
    await asyncio.to_thread(preload_ml_models)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    url: str
    force_reanalyze: bool = False

def _client_error_message(exc: Exception) -> str:
    """Safe user-facing message — no stack traces or internal paths."""
    if isinstance(exc, InvalidSourceIdError):
        return "Invalid track id."
    if isinstance(exc, FileNotFoundError):
        return "Audio file not found. Try uploading again or re-analyze."
    if isinstance(exc, RuntimeError):
        return "Download failed. Try another link or upload a file."
    logger.exception("Request failed")
    return "Analysis failed. Please try again."

def _enrich_analysis(data: dict, url: str) -> dict:
    if not data.get("song"):
        data["song"] = fetch_track_metadata(url)
    yt_id = extract_youtube_id(url)
    if yt_id:
        song = dict(data.get("song") or {})
        song["youtube_id"] = yt_id
        data["song"] = song
    lyrics = data.get("lyrics")
    needs_lyrics = not lyrics
    needs_sync_lines = bool(
        lyrics and lyrics.get("synced") and not lyrics.get("lines")
    )
    if (needs_lyrics or needs_sync_lines) and data.get("timeline"):
        from lyrics import attach_lyrics_to_timeline
        timeline, lyrics_meta = attach_lyrics_to_timeline(
            data["timeline"],
            data.get("song"),
            _timeline_duration(data["timeline"]),
            youtube_id=yt_id,
        )
        data["timeline"] = timeline
        data["lyrics"] = lyrics_meta
    return data


def _timeline_duration(timeline: list) -> float:
    if not timeline:
        return 0.0
    last = timeline[-1]
    return float(last.get("end_time") or last.get("time", 0))

@app.get("/api/health")
async def health():
    ml = summarize_ml_readiness()
    return {
        "status": "ok",
        "analyzer_version": ANALYZER_VERSION,
        "chord_engine": CHORD_ENGINE,
        "ml_ready": ml["ml_deps_ok"] if CHORD_ENGINE == "ml" else None,
        "ml_setup_hint": ml.get("setup_hint"),
    }

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use MP3, WAV, M4A, FLAC, or OGG.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File too large (max 50 MB).")
        chunks.append(chunk)
    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = await asyncio.to_thread(save_uploaded_file, contents, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_client_error_message(exc)) from exc

    return {
        "status": "ready",
        "source_id": result["source_id"],
        "url": result["url"],
        "title": result["title"],
        "artist": result.get("artist", ""),
        "album_art_url": result.get("album_art_url"),
    }

@app.post("/api/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    source_id = extract_source_id(req.url)
    try:
        validate_source_id(source_id)
        cache_file = resolve_cache_json(source_id)
    except InvalidSourceIdError as exc:
        raise HTTPException(status_code=400, detail="Invalid source.") from exc

    if not req.force_reanalyze and cache_file.is_file():
        with open(cache_file, "r") as f:
            data = json.load(f)
        if is_cache_valid(data):
            data = _enrich_analysis(data, req.url)
            return {"status": "cached", "data": data, "video_id": source_id}

    return {"status": "processing", "video_id": source_id, "url": req.url}

@app.get("/api/progress/{video_id}")
async def progress_endpoint(video_id: str, url: str, force_reanalyze: bool = False):
    source_id = extract_source_id(url)
    try:
        validate_source_id(source_id)
        if video_id != source_id:
            video_id = source_id
    except InvalidSourceIdError as exc:
        raise HTTPException(status_code=400, detail="Invalid track id.") from exc

    async def event_generator():
        try:
            yield f"data: {json.dumps({'stage': 'Downloading...'})}\n\n"

            download_info = await download_audio(url)
            audio_path = download_info["audio_path"]
            canonical_id = download_info["video_id"]
            song_metadata = await asyncio.to_thread(fetch_track_metadata, url)
            yt_id = extract_youtube_id(url)
            if yt_id and song_metadata is not None:
                song_metadata = dict(song_metadata)
                song_metadata["youtube_id"] = yt_id

            q = asyncio.Queue()

            def progress_cb(msg):
                q.put_nowait(msg)

            async def pump_progress(task: asyncio.Task):
                while not task.done():
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=0.1)
                        yield f"data: {json.dumps({'stage': msg})}\n\n"
                    except asyncio.TimeoutError:
                        pass
                while not q.empty():
                    msg = q.get_nowait()
                    yield f"data: {json.dumps({'stage': msg})}\n\n"

            analyze_task = asyncio.create_task(
                run_analysis_deduped(
                    canonical_id,
                    lambda: analyze_audio(
                        audio_path,
                        canonical_id,
                        progress_cb,
                        force_reanalyze,
                        song_metadata,
                    ),
                )
            )

            async for chunk in pump_progress(analyze_task):
                yield chunk

            result = await analyze_task
            result = _enrich_analysis(result, url)
            yield f"data: {json.dumps({'stage': 'Complete', 'result': result})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'error': _client_error_message(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/audio/{video_id}")
async def get_audio(video_id: str):
    try:
        audio_path = resolve_download_mp3(video_id)
    except InvalidSourceIdError as exc:
        raise HTTPException(status_code=400, detail="Invalid track id.") from exc

    if audio_path.is_file():
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
