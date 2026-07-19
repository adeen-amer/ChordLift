import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import type { AnalysisData, AudioLoadState, ChordEvent } from '../types';
import { AudioPlayer } from '../components/AudioPlayer';
import { ChordSequence } from '../components/ChordSequence';
import { LyricsPanel } from '../components/LyricsPanel';
import { plainLinesFromTimeline } from '../utils/lyricsSync';
import { applyCorrections, loadCorrections, saveCorrection } from '../utils/chordCorrections';
import {
  loadPresentationMode,
  savePresentationMode,
  type PresentationMode,
} from '../utils/presentation';
import { FretboardViewer } from '../components/FretboardViewer';
import { ScalePractice } from '../components/ScalePractice';
import { useSyncEngine } from '../hooks/useSyncEngine';
import { Music, ArrowRight, Upload, Guitar } from 'lucide-react';
import { apiUrl, UPLOAD_ONLY } from '../config';

const SOLO_ENABLED = import.meta.env.VITE_ENABLE_SOLO === 'true';

type AppView = 'analyze' | 'practice';

function HomePage() {
  const [view, setView] = useState<AppView>('analyze');
  const [url, setUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [presentationMode, setPresentationMode] = useState<PresentationMode>(loadPresentationMode);
  const [correctionCache, setCorrectionCache] = useState<{ videoId: string; map: ReturnType<typeof loadCorrections> } | null>(null);
  const [audioLoadState, setAudioLoadState] = useState<AudioLoadState>('idle');
  const [audioTrackId, setAudioTrackId] = useState<string | undefined>();

  const videoId = data?.video_id;

  if (videoId !== audioTrackId) {
    setAudioTrackId(videoId);
    setAudioLoadState(videoId ? 'loading' : 'idle');
  }

  const corrections = useMemo(() => {
    if (!videoId) return {};
    if (correctionCache?.videoId === videoId) return correctionCache.map;
    return loadCorrections(videoId);
  }, [videoId, correctionCache]);

  const effectivePresentationMode = useMemo((): PresentationMode => {
    if (presentationMode === 'raw' && data && !data.model_timeline?.length) {
      return 'synced';
    }
    return presentationMode;
  }, [presentationMode, data]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const chordContainerRef = useRef<HTMLDivElement | null>(null);
  const lyricsPanelRef = useRef<HTMLDivElement | null>(null);
  const fretboardRef = useRef<SVGSVGElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const analysisGenRef = useRef(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const cancelInFlightAnalysis = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  useEffect(() => () => cancelInFlightAnalysis(), [cancelInFlightAnalysis]);

  const baseTimeline = useMemo((): ChordEvent[] => {
    if (!data) return [];
    if (effectivePresentationMode === 'raw' && data.model_timeline?.length) {
      return data.model_timeline;
    }
    return data.timeline;
  }, [data, effectivePresentationMode]);

  const displayTimeline = useMemo(
    () => applyCorrections(baseTimeline, corrections),
    [baseTimeline, corrections],
  );

  const syncData = useMemo((): AnalysisData | null => {
    if (!data) return null;
    return { ...data, timeline: displayTimeline };
  }, [data, displayTimeline]);

  useSyncEngine({
    audioRef,
    data: syncData,
    chordContainerRef,
    lyricsPanelRef,
    fretboardRef,
    improvEnabled: false,
  });

  const handleCorrectChord = useCallback((time: number, chord: string) => {
    if (!data?.video_id) return;
    saveCorrection(data.video_id, time, chord);
    setCorrectionCache({ videoId: data.video_id, map: loadCorrections(data.video_id) });
  }, [data]);

  const handlePresentationToggle = (mode: PresentationMode) => {
    setPresentationMode(mode);
    savePresentationMode(mode);
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !videoId) return;

    const onReady = () => setAudioLoadState('ready');
    const onError = () => {
      setAudioLoadState('error');
      setError('Audio failed to load. Try re-analyzing or upload a file.');
    };

    audio.addEventListener('loadedmetadata', onReady);
    audio.addEventListener('canplay', onReady);
    audio.addEventListener('error', onError);

    if (audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
      onReady();
    }

    return () => {
      audio.removeEventListener('loadedmetadata', onReady);
      audio.removeEventListener('canplay', onReady);
      audio.removeEventListener('error', onError);
    };
  }, [videoId]);

  const runAnalysis = async (sourceUrl: string, forceReanalyze = false) => {
    cancelInFlightAnalysis();
    const gen = ++analysisGenRef.current;
    const controller = new AbortController();
    abortRef.current = controller;

    setIsAnalyzing(true);
    setStage(forceReanalyze ? "Re-analyzing..." : "Starting...");
    setError(null);
    setAudioLoadState('idle');
    if (forceReanalyze) setData(null);

    try {
      const res = await fetch(apiUrl('/api/analyze'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: sourceUrl, force_reanalyze: forceReanalyze }),
        signal: controller.signal,
      });

      if (gen !== analysisGenRef.current) return;

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
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        if (gen !== analysisGenRef.current) return;

        const payload = JSON.parse(event.data);
        if (payload.error) {
          setError(payload.error);
          setIsAnalyzing(false);
          setStage(null);
          eventSource.close();
          eventSourceRef.current = null;
        } else if (payload.stage === 'Complete') {
          setData(payload.result);
          setIsAnalyzing(false);
          setStage(null);
          eventSource.close();
          eventSourceRef.current = null;
        } else {
          setStage(payload.stage);
        }
      };

      eventSource.onerror = () => {
        if (gen !== analysisGenRef.current) return;
        setError("Connection to analysis server lost.");
        setIsAnalyzing(false);
        setStage(null);
        eventSource.close();
        eventSourceRef.current = null;
      };
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (gen !== analysisGenRef.current) return;
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

    cancelInFlightAnalysis();
    const gen = ++analysisGenRef.current;
    const controller = new AbortController();
    abortRef.current = controller;

    setIsAnalyzing(true);
    setStage("Uploading...");
    setError(null);
    setData(null);
    setAudioLoadState('idle');

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(apiUrl('/api/upload'), {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (gen !== analysisGenRef.current) return;

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const uploadData = await res.json();
      setUrl(uploadData.url);
      await runAnalysis(uploadData.url);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (gen !== analysisGenRef.current) return;
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
        {!UPLOAD_ONLY && (
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
        )}

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
            key={data.video_id}
            audioRef={audioRef}
            src={apiUrl(`/api/audio/${data.video_id}`)}
            song={data.song}
            songKey={data.key}
            capo={data.capo}
            analyzerVersion={data.analyzer_version}
            chordEngine={data.chord_engine}
            loadState={audioLoadState}
          />

          <div className="presentation-row">
            <span className="presentation-label">Chord view:</span>
            <button
              type="button"
              className={`btn-secondary presentation-btn${presentationMode === 'synced' ? ' active' : ''}`}
              onClick={() => handlePresentationToggle('synced')}
              aria-pressed={presentationMode === 'synced'}
            >
              Synced (recommended)
            </button>
            <button
              type="button"
              className={`btn-secondary presentation-btn${presentationMode === 'raw' ? ' active' : ''}`}
              onClick={() => handlePresentationToggle('raw')}
              disabled={!data.model_timeline?.length}
              aria-pressed={presentationMode === 'raw'}
            >
              Raw model
            </button>
          </div>

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
              <ChordSequence
                key={`${data.video_id}-${effectivePresentationMode}`}
                timeline={displayTimeline}
                downbeatTimes={data.beats?.downbeat_times}
                ref={chordContainerRef}
                onCorrectChord={handleCorrectChord}
              />
            </div>

            {(data.lyrics?.lines?.length || data.lyrics?.plain_text || displayTimeline.some(s => s.lyrics)) && (
              <div className="glass-panel lyrics-glass">
                <LyricsPanel
                  ref={lyricsPanelRef}
                  lines={data.lyrics?.lines}
                  synced={Boolean(data.lyrics?.synced && data.lyrics?.lines?.length)}
                  plainLines={
                    data.lyrics?.plain_text
                      ? data.lyrics.plain_text.split(/\n+/).map(l => l.trim()).filter(Boolean)
                      : plainLinesFromTimeline(displayTimeline)
                  }
                />
              </div>
            )}
          </div>

          {SOLO_ENABLED && (
          <div className="glass-panel fretboard-panel">
            <FretboardViewer
              solos={data.solos}
              songKey={data.key}
              improvEnabled={false}
              onImprovToggle={() => {}}
              ref={fretboardRef}
            />
          </div>
          )}
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

export default HomePage;
