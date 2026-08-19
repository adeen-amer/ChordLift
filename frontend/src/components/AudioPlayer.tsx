import { useState, useEffect, useRef, type RefObject } from 'react';
import { Play, Pause, Disc3, Music2, Guitar, Minus, Plus, Repeat, X } from 'lucide-react';
import type { CapoInfo, KeyInfo, SongInfo, AudioLoadState } from '../types';
import {
  DEFAULT_SPEED,
  formatSpeed,
  loopSeekTarget,
  normalizeLoop,
  snapToDownbeat,
  stepSpeed,
  SPEED_STEPS,
} from '../utils/practice';

interface AudioPlayerProps {
  audioRef: RefObject<HTMLAudioElement | null>;
  src: string | null;
  song?: SongInfo;
  songKey?: KeyInfo;
  capo?: CapoInfo;
  analyzerVersion?: string;
  chordEngine?: string;
  loadState?: AudioLoadState;
  downbeatTimes?: number[];
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
  downbeatTimes,
}: AudioPlayerProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState<number>(DEFAULT_SPEED);
  const [loopA, setLoopA] = useState<number | null>(null);
  const [loopB, setLoopB] = useState<number | null>(null);

  const loop = normalizeLoop(loopA, loopB, duration);

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

  useEffect(() => {
    setLoopA(null);
    setLoopB(null);
    setSpeed(DEFAULT_SPEED);
  }, [src]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.playbackRate = speed;
    // Without this a slowed-down track is also pitched down, which makes it
    // useless for playing along to.
    audio.preservesPitch = true;
  }, [audioRef, speed, src, loadState]);

  // Two clocks on purpose. rAF is precise (~60Hz) but browsers stop it
  // entirely in a hidden/non-compositing tab, while audio keeps playing -- so
  // on its own the loop silently breaks the moment you switch tabs. The
  // `timeupdate` event is coarse (~4Hz, so it can overshoot a fraction of a
  // second) but keeps firing when hidden. Running both means the loop is tight
  // in the foreground and still holds in the background.
  const loopRef = useRef(loop);
  loopRef.current = loop;
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !loop || !isPlaying) return;

    const enforce = () => {
      const target = loopSeekTarget(audio.currentTime, loopRef.current);
      if (target !== null) audio.currentTime = target;
    };

    let frame = 0;
    const tick = () => {
      enforce();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    audio.addEventListener('timeupdate', enforce);

    return () => {
      cancelAnimationFrame(frame);
      audio.removeEventListener('timeupdate', enforce);
    };
  }, [audioRef, isPlaying, loop?.start, loop?.end]);

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
          {loop && duration > 0 && (
            <div
              className="loop-region"
              style={{
                left: `${(loop.start / duration) * 100}%`,
                width: `${((loop.end - loop.start) / duration) * 100}%`,
              }}
              aria-hidden="true"
            />
          )}
          <div
            className="progress-fill"
            style={{ width: `${duration ? (progress / duration) * 100 : 0}%` }}
          />
        </div>
        <span className="time">{formatTime(duration)}</span>
      </div>

      <div className="practice-bar">
        <div className="practice-group" role="group" aria-label="Playback speed">
          <button
            className="practice-btn"
            onClick={() => setSpeed((s) => stepSpeed(s, -1))}
            disabled={!canPlay || speed === SPEED_STEPS[0]}
            aria-label="Slower"
          >
            <Minus size={14} />
          </button>
          <span className="practice-value" aria-live="polite">
            {formatSpeed(speed)}
          </span>
          <button
            className="practice-btn"
            onClick={() => setSpeed((s) => stepSpeed(s, 1))}
            disabled={!canPlay || speed === SPEED_STEPS[SPEED_STEPS.length - 1]}
            aria-label="Faster"
          >
            <Plus size={14} />
          </button>
        </div>

        <div className="practice-group" role="group" aria-label="Loop section">
          <Repeat size={14} className="practice-icon" aria-hidden="true" />
          <button
            className={`practice-btn practice-btn-wide${loopA !== null ? ' is-set' : ''}`}
            onClick={() => setLoopA(snapToDownbeat(progress, downbeatTimes))}
            disabled={!canPlay}
            aria-label={
              loopA === null ? 'Set loop start at playhead' : `Loop start ${formatTime(loopA)}`
            }
          >
            A{loopA !== null && ` ${formatTime(loopA)}`}
          </button>
          <button
            className={`practice-btn practice-btn-wide${loopB !== null ? ' is-set' : ''}`}
            onClick={() => setLoopB(snapToDownbeat(progress, downbeatTimes))}
            disabled={!canPlay}
            aria-label={
              loopB === null ? 'Set loop end at playhead' : `Loop end ${formatTime(loopB)}`
            }
          >
            B{loopB !== null && ` ${formatTime(loopB)}`}
          </button>
          <button
            className="practice-btn"
            onClick={() => {
              setLoopA(null);
              setLoopB(null);
            }}
            disabled={loopA === null && loopB === null}
            aria-label="Clear loop"
          >
            <X size={14} />
          </button>
          {downbeatTimes && downbeatTimes.length > 0 && (
            <span className="practice-note">snaps to bars</span>
          )}
        </div>
      </div>
    </div>
  );
};
