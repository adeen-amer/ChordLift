from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
from pydantic import BaseModel

from downloader import (
    download_audio,
    extract_source_id,
    extract_youtube_id,
    fetch_track_metadata,
    save_uploaded_file,
    DOWNLOAD_DIR,
)
from analyzer import analyze_audio, CACHE_DIR, is_cache_valid

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

ALLOWED_UPLOAD_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}

app = FastAPI(title="ChordLift API")

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
    return {"status": "ok"}

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

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")

    try:
        result = await asyncio.to_thread(save_uploaded_file, contents, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
    cache_file = os.path.join(CACHE_DIR, f"{source_id}.json")

    if not req.force_reanalyze and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        if is_cache_valid(data):
            data = _enrich_analysis(data, req.url)
            return {"status": "cached", "data": data, "video_id": source_id}

    return {"status": "processing", "video_id": source_id, "url": req.url}

@app.get("/api/progress/{video_id}")
async def progress_endpoint(video_id: str, url: str, force_reanalyze: bool = False):
    async def event_generator():
        try:
            yield f"data: {json.dumps({'stage': 'Downloading...'})}\n\n"

            download_info = await download_audio(url)
            audio_path = download_info["audio_path"]
            song_metadata = await asyncio.to_thread(fetch_track_metadata, url)
            yt_id = extract_youtube_id(url)
            if yt_id and song_metadata is not None:
                song_metadata = dict(song_metadata)
                song_metadata["youtube_id"] = yt_id

            q = asyncio.Queue()

            def progress_cb(msg):
                q.put_nowait(msg)

            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(
                None,
                lambda: analyze_audio(
                    audio_path, video_id, progress_cb, force_reanalyze, song_metadata,
                ),
            )

            while not task.done():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.1)
                    yield f"data: {json.dumps({'stage': msg})}\n\n"
                except asyncio.TimeoutError:
                    pass

            result = task.result()
            result = _enrich_analysis(result, url)
            yield f"data: {json.dumps({'stage': 'Complete', 'result': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
    audio_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.mp3")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
