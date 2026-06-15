# ChordLift 🎸

ChordLift is an AI-powered guitar practice tool that analyzes YouTube videos to detect chords, strumming patterns, and guitar solos. It features a real-time, high-performance sync engine that ties the audio playback directly to an SVG-rendered chord sequence and an interactive fretboard for solos.

## Features
- **Multi-Source Audio**: YouTube, Spotify (via search match), SoundCloud, or direct file upload (MP3/WAV/M4A).
- **Resilient YouTube Downloads**: Uses current yt-dlp client workarounds, optional cookies file, and Piped/Invidious fallbacks.
- **Chord Detection**: Harmonic-percussive separation, smoothed chroma analysis, and extended chord types (maj/min, 7, maj7, m7, sus2, sus4).
- **Solo Fretboard**: Pinpoints individual guitar notes using `basic-pitch` and renders them on an interactive SVG fretboard synced to playback.
- **Real-Time Sync**: Uses a `requestAnimationFrame` loop to guarantee buttery-smooth 60 FPS highlighting of chords and notes without React re-render lag.

---

## Prerequisites

- **Python 3.10+** (Required for the AI backend)
- **Node.js 18+** (Required for the React frontend)
- **FFmpeg** (Required system-level dependency for yt-dlp audio extraction)
  - **Mac**: `brew install ffmpeg`
  - **Ubuntu**: `sudo apt install ffmpeg`
  - **Windows**: Download via `winget install ffmpeg` or use a pre-built binary.

---

## Setup & Running (Local Development)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # optional
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env   # optional — leave VITE_API_BASE empty to use Vite proxy
npm run dev
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`, so no CORS configuration is needed locally.

### 3. Usage

1. Open the frontend in your browser (usually `http://localhost:5173`).
2. Paste a **YouTube**, **Spotify**, or **SoundCloud** URL — or click **Upload audio file**.
3. Click **Analyze**.
4. Wait for the pipeline to download the audio, extract chords, and detect solos.
5. Use the Audio Player to play the track and watch the chords and solo notes highlight in real-time!

### YouTube 403 errors?

YouTube frequently blocks downloads. ChordLift tries several workarounds automatically. If a YouTube link still fails:

1. **Upload an MP3/WAV** of the song directly (most reliable).
2. Paste a **Spotify link** instead — ChordLift resolves the track and finds matching audio.
3. Export browser cookies to `cookies.txt` and set `YTDLP_COOKIES_FILE` in `backend/.env`.
4. Update yt-dlp: `pip install -U yt-dlp`

---

## Deployment (Docker)

The simplest way to run ChordLift in production is with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

- **Frontend**: http://localhost:3000 (nginx serves the React app and proxies `/api` to the backend)
- **Backend API**: http://localhost:8000 (direct access, optional)

### Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | backend | Comma-separated CORS origins (include your public frontend URL) |
| `PORT` | backend | API port (default `8000`) |
| `VITE_API_BASE` | frontend build | API base URL. Leave empty for same-origin nginx proxy |

### Production Notes

- Persisted volumes store analysis cache and downloaded audio between restarts.
- The backend health check is available at `GET /api/health`.
- For a custom domain, put a reverse proxy (Caddy, Traefik, nginx) in front of the frontend container on port 80/443.
- YouTube downloads may require browser cookies on the host machine when running locally; in Docker, the iOS/web client fallbacks are used automatically.

---

## Architecture Notes

- The backend uses harmonic-percussive separation and median-filtered chroma for chord detection, plus Spotify's `basic-pitch` (ONNX) for polyphonic note detection.
- Analysis results are cached with a version stamp; bumping `ANALYZER_VERSION` in `analyzer.py` invalidates stale cache entries.
- The frontend circumvents React-state bottlenecks during playback by reading `audio.currentTime` in a `requestAnimationFrame` loop and updating DOM/SVG nodes directly.
