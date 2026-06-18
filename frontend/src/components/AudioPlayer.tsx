import { useState, useEffect, type RefObject } from 'react';
import { Play, Pause, Disc3, Music2, Guitar } from 'lucide-react';
import type { CapoInfo, KeyInfo, SongInfo, AudioLoadState } from '../types';

interface AudioPlayerProps {
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string | null;
  song?: SongInfo;
  songKey?: KeyInfo;
  capo?: CapoInfo;
  analyzerVersion?: string;
  chordEngine?: string;
  loadState?: AudioLoadState;
}

export const AudioPlayer = ({
  audioRef,
  src,
  song,
  songKey,
  capo,
  analyzerVersion,
  chordEngine,
  loadState = 'idle',
}: AudioPlayerProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const title = song?.title?.trim() || 'Unknown track';
  const artist = song?.artist?.trim();
  const artUrl = song?.album_art_url;
  const keyDisplay = songKey?.display;
  const capoDisplay = capo?.display;
  const engineLabel = chordEngine === 'ml' ? 'ML' : chordEngine === 'classic' ? 'Classic' : chordEngine;
  const metaParts = [
    engineLabel && `Engine: ${engineLabel}`,
    analyzerVersion && `v${analyzerVersion}`,
  ].filter(Boolean);

  const canPlay = Boolean(src) && loadState === 'ready';

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateProgress = () => {
      setProgress(audio.currentTime);
    };

    const updateDuration = () => {
      setDuration(audio.duration);
    };

    const handleEnded = () => setIsPlaying(false);
    const handlePause = () => setIsPlaying(false);
    const handlePlay = () => setIsPlaying(true);

    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('play', handlePlay);

    return () => {
      audio.removeEventListener('timeupdate', updateProgress);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('play', handlePlay);
    };
  }, [audioRef, src]);

  const togglePlay = async () => {
    if (!audioRef.current || !canPlay) return;

    if (isPlaying) {
      audioRef.current.pause();
      return;
    }

    try {
      await audioRef.current.play();
    } catch {
      setIsPlaying(false);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    audioRef.current.currentTime = percent * duration;
    setProgress(percent * duration);
  };

  const formatTime = (time: number) => {
    if (isNaN(time)) return "0:00";
    const m = Math.floor(time / 60);
    const s = Math.floor(time % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const playDisabled = !canPlay;
  const playHint =
    loadState === 'loading'
      ? 'Loading audio…'
      : loadState === 'error'
        ? 'Audio unavailable'
        : undefined;

  return (
    <div className="audio-player glass-panel">
      <div className="player-top">
        <div className="song-art">
          {artUrl ? (
            <img src={artUrl} alt={`${title} album art`} className="song-art-img" />
          ) : (
            <div className="song-art-placeholder" aria-hidden="true">
              <Music2 size={28} />
            </div>
          )}
        </div>

        <div className="song-meta">
          <h2 className="song-title">{title}</h2>
          {artist && <p className="song-artist">{artist}</p>}
          {keyDisplay && (
            <div className="song-key">
              <Disc3 size={14} />
              <span>Key: <strong>{keyDisplay}</strong></span>
            </div>
          )}
          {capoDisplay && (
            <div className="song-capo">
              <Guitar size={14} />
              <span><strong>{capoDisplay}</strong> (open shapes)</span>
            </div>
          )}
          {metaParts.length > 0 && (
            <p className="analysis-meta">{metaParts.join(' · ')}</p>
          )}
          {playHint && <p className="analysis-meta">{playHint}</p>}
        </div>

        <button
          className="play-btn"
          onClick={togglePlay}
          disabled={playDisabled}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause size={24} /> : <Play size={24} fill="currentColor" />}
        </button>
      </div>

      <div className="progress-container">
        <span className="time">{formatTime(progress)}</span>
        <div className="progress-bar" onClick={handleSeek}>
          <div
            className="progress-fill"
            style={{ width: `${duration ? (progress / duration) * 100 : 0}%` }}
          />
        </div>
        <span className="time">{formatTime(duration)}</span>
      </div>
    </div>
  );
};
