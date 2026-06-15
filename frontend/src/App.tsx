import { useState, useRef } from 'react';
import type { AnalysisData } from './types';
import { AudioPlayer } from './components/AudioPlayer';
import { ChordSequence } from './components/ChordSequence';
import { LyricsPanel, plainLinesFromTimeline } from './components/LyricsPanel';
import { FretboardViewer } from './components/FretboardViewer';
import { ScalePractice } from './components/ScalePractice';
import { useSyncEngine } from './hooks/useSyncEngine';
import { Music, ArrowRight, Upload, Guitar } from 'lucide-react';
import { apiUrl } from './config';

type AppView = 'analyze' | 'practice';

function App() {
  const [view, setView] = useState<AppView>('analyze');
  const [url, setUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [improvEnabled, setImprovEnabled] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const chordContainerRef = useRef<HTMLDivElement | null>(null);
  const lyricsPanelRef = useRef<HTMLDivElement | null>(null);
  const fretboardRef = useRef<SVGSVGElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useSyncEngine({
    audioRef,
    data,
    chordContainerRef,
    lyricsPanelRef,
    fretboardRef,
    improvEnabled,
  });

  const runAnalysis = async (sourceUrl: string, forceReanalyze = false) => {
    setIsAnalyzing(true);
    setStage(forceReanalyze ? "Re-analyzing..." : "Starting...");
    setError(null);
    if (forceReanalyze) setData(null);

    try {
      const res = await fetch(apiUrl('/api/analyze'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: sourceUrl, force_reanalyze: forceReanalyze })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const initData = await res.json();
      const videoId = initData.video_id;

      if (initData.status === "cached") {
        setData(initData.data);
        setIsAnalyzing(false);
        setStage(null);
        return;
      }

      const sseUrl = `${apiUrl(`/api/progress/${videoId}`)}?url=${encodeURIComponent(sourceUrl)}&force_reanalyze=${forceReanalyze}`;
      const eventSource = new EventSource(sseUrl);

      eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.error) {
          setError(payload.error);
          setIsAnalyzing(false);
          setStage(null);
          eventSource.close();
        } else if (payload.stage === 'Complete') {
          setData(payload.result);
          setIsAnalyzing(false);
          setStage(null);
          eventSource.close();
        } else {
          setStage(payload.stage);
        }
      };

      eventSource.onerror = () => {
        setError("Connection to analysis server lost.");
        setIsAnalyzing(false);
        setStage(null);
        eventSource.close();
      };
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setIsAnalyzing(false);
      setStage(null);
    }
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    await runAnalysis(url);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsAnalyzing(true);
    setStage("Uploading...");
    setError(null);
    setData(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(apiUrl('/api/upload'), {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const uploadData = await res.json();
      setUrl(uploadData.url);
      await runAnalysis(uploadData.url);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setIsAnalyzing(false);
      setStage(null);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>ChordLift</h1>
        <p>AI-Powered Guitar Tab & Chord Synchronization</p>
        <nav className="app-nav" aria-label="Main">
          <button
            type="button"
            className={`app-nav-btn${view === 'analyze' ? ' active' : ''}`}
            onClick={() => setView('analyze')}
          >
            <Music size={18} />
            Analyze
          </button>
          <button
            type="button"
            className={`app-nav-btn${view === 'practice' ? ' active' : ''}`}
            onClick={() => setView('practice')}
          >
            <Guitar size={18} />
            Scale practice
          </button>
        </nav>
      </header>

      {view === 'practice' && <ScalePractice />}

      {view === 'analyze' && (
      <>
      <div className="glass-panel">
        <form className="input-form" onSubmit={handleAnalyze}>
          <input
            type="url"
            placeholder="YouTube, Spotify, or SoundCloud URL..."
            value={url}
            onChange={e => setUrl(e.target.value)}
            disabled={isAnalyzing}
          />
          <button type="submit" className="btn-primary" disabled={isAnalyzing || !url}>
            {isAnalyzing ? "Analyzing" : "Analyze"}
            {!isAnalyzing && <ArrowRight size={20} />}
          </button>
        </form>

        <div className="upload-row">
          <span className="upload-divider">or</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg"
            onChange={handleFileUpload}
            disabled={isAnalyzing}
            hidden
          />
          <button
            type="button"
            className="btn-secondary"
            disabled={isAnalyzing}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={18} />
            Upload audio file
          </button>
        </div>

        {error && (
          <div className="error-banner">
            <p>{error}</p>
            <p className="error-hint">
              YouTube blocked? Try a Spotify link, SoundCloud URL, or upload an MP3/WAV file directly.
            </p>
          </div>
        )}
      </div>

      {isAnalyzing && (
        <div className="loader-container">
          <div className="spinner"></div>
          <div style={{ color: 'var(--accent-secondary)', fontSize: '1.2rem', fontWeight: 600 }}>
            {stage}
          </div>
        </div>
      )}

      {data && !isAnalyzing && (
        <>
          <audio
            ref={audioRef}
            src={apiUrl(`/api/audio/${data.video_id}`)}
            crossOrigin="anonymous"
          />

          <AudioPlayer
            audioRef={audioRef}
            src={apiUrl(`/api/audio/${data.video_id}`)}
            song={data.song}
            songKey={data.key}
            analyzerVersion={data.analyzer_version}
            chordEngine={data.chord_engine}
          />

          <div className="reanalyze-row">
            <button
              type="button"
              className="btn-secondary"
              disabled={isAnalyzing || !url}
              onClick={() => runAnalysis(url, true)}
            >
              Re-analyze chords
            </button>
          </div>

          <div className="playback-main">
            <div className="glass-panel chord-panel">
              <ChordSequence timeline={data.timeline} ref={chordContainerRef} />
            </div>

            {(data.lyrics?.lines?.length || data.lyrics?.plain_text || data.timeline.some(s => s.lyrics)) && (
              <div className="glass-panel lyrics-glass">
                <LyricsPanel
                  ref={lyricsPanelRef}
                  lines={data.lyrics?.lines}
                  synced={Boolean(data.lyrics?.synced && data.lyrics?.lines?.length)}
                  plainLines={
                    data.lyrics?.plain_text
                      ? data.lyrics.plain_text.split(/\n+/).map(l => l.trim()).filter(Boolean)
                      : plainLinesFromTimeline(data.timeline)
                  }
                />
              </div>
            )}
          </div>

          <div className="glass-panel fretboard-panel">
            <FretboardViewer
              solos={data.solos}
              songKey={data.key}
              improvEnabled={improvEnabled}
              onImprovToggle={setImprovEnabled}
              ref={fretboardRef}
            />
          </div>
        </>
      )}

      {!data && !isAnalyzing && !error && (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '40px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <Music size={64} opacity={0.2} />
          <p>Paste a YouTube, Spotify, or SoundCloud link — or upload an audio file — to extract chords and solos.</p>
        </div>
      )}
      </>
      )}
    </div>
  );
}

export default App;
