import { useState, useEffect, type RefObject } from 'react';
import { Play, Pause, Disc3, Music2 } from 'lucide-react';
import type { KeyInfo, SongInfo } from '../types';

interface AudioPlayerProps {
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string | null;
  song?: SongInfo;
  songKey?: KeyInfo;
  analyzerVersion?: string;
  chordEngine?: string;
}

export const AudioPlayer = ({ audioRef, src, song, songKey, analyzerVersion, chordEngine }: AudioPlayerProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const title = song?.title?.trim() || 'Unknown track';
  const artist = song?.artist?.trim();
  const artUrl = song?.album_art_url;
  const keyDisplay = songKey?.display;
  const engineLabel = chordEngine === 'ml' ? 'ML' : chordEngine === 'classic' ? 'Classic' : chordEngine;
  const metaParts = [
    engineLabel && `Engine: ${engineLabel}`,
    analyzerVersion && `v${analyzerVersion}`,
  ].filter(Boolean);

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

    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateProgress);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioRef]);

  const togglePlay = () => {
    if (!audioRef.current || !src) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
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
          {metaParts.length > 0 && (
            <p className="analysis-meta">{metaParts.join(' · ')}</p>
          )}
        </div>

        <button className="play-btn" onClick={togglePlay} disabled={!src} aria-label={isPlaying ? 'Pause' : 'Play'}>
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
