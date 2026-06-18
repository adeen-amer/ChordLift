import { forwardRef } from 'react';
import type { LyricLine } from '../types';

interface LyricsPanelProps {
  lines?: LyricLine[];
  /** Unsynced lyrics shown as static text when no LRC timestamps exist. */
  plainLines?: string[];
  synced?: boolean;
}

export const LyricsPanel = forwardRef<HTMLDivElement, LyricsPanelProps>(
  ({ lines = [], plainLines = [], synced = false }, ref) => {
  const hasSynced = synced && lines.length > 0;
  const hasPlain = !hasSynced && plainLines.length > 0;

  if (!hasSynced && !hasPlain) {
    return null;
  }

  return (
    <div className="lyrics-panel-wrap">
      <div className="lyrics-panel-header">
        <span className="lyrics-panel-title">Lyrics</span>
        {hasSynced ? (
          <span className="lyrics-panel-badge">Synced</span>
        ) : (
          <span className="lyrics-panel-badge muted">Not synced</span>
        )}
      </div>

      {hasSynced ? (
        <div className="lyrics-panel" aria-label="Synced lyrics">
          <div className="lyrics-panel-scroll" ref={ref}>
            {lines.map((line, idx) => (
              <p
                key={`lyric-${idx}`}
                className="lyrics-line"
                data-index={idx}
              >
                {line.text}
              </p>
            ))}
          </div>
        </div>
      ) : (
        <div className="lyrics-panel lyrics-panel-plain" aria-label="Lyrics">
          <div className="lyrics-panel-scroll">
            {plainLines.map((text, idx) => (
              <p key={`plain-lyric-${idx}`} className="lyrics-line plain">
                {text}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

LyricsPanel.displayName = 'LyricsPanel';
