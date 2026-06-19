import { forwardRef, useMemo } from 'react';
import type { KeyInfo, SoloSection } from '../types';
import { getImprovGuideData } from '../utils/pentatonic';
import { improvGuideToScaleGuide } from '../utils/scales';
import { PentatonicFretboard } from './PentatonicFretboard';
import { stringFretToXY } from '../utils/fretboardLayout';

interface FretboardViewerProps {
  solos: SoloSection[];
  songKey?: KeyInfo;
  improvEnabled: boolean;
  onImprovToggle: (enabled: boolean) => void;
}

export const FretboardViewer = forwardRef<SVGSVGElement, FretboardViewerProps>(
  ({ solos, songKey, improvEnabled, onImprovToggle }, ref) => {
    const guideData = useMemo(
      () =>
        improvEnabled && songKey
          ? improvGuideToScaleGuide(getImprovGuideData(songKey, 18), songKey.display)
          : null,
      [improvEnabled, songKey],
    );

    const soloNotes = solos.flatMap((solo, soloIdx) =>
      solo.notes.map((note, noteIdx) => {
        const { x, y } = stringFretToXY(note.string, note.fret);
        return (
          <g key={`solo-${soloIdx}-note-${noteIdx}`}>
                <circle
                  id={`dot-${soloIdx}-${noteIdx}`}
                  className="note-dot"
                  data-solo={soloIdx}
                  cx={x}
                  cy={y}
                  r={0}
                  opacity={0}
                  fill="var(--accent-primary)"
                />
                <text
                  id={`text-${soloIdx}-${noteIdx}`}
                  className="note-text"
                  data-solo={soloIdx}
                  x={x}
                  y={y}
                  opacity={0}
                >
                  {note.note.replace(/[0-9]/g, '')}
                </text>
              </g>
        );
      }),
    );

    return (
      <div className="fretboard-section">
        <div className="fretboard-header">
          <div className="fretboard-header-text">
            <h3 className="fretboard-title">Improvisation guide</h3>
            {improvEnabled && songKey && (
              <span className="fretboard-key-hint">{songKey.display} · 5 CAGED boxes</span>
            )}
          </div>
          <label className="improv-toggle">
            <input
              type="checkbox"
              checked={improvEnabled}
              onChange={(e) => onImprovToggle(e.target.checked)}
              disabled={!songKey}
            />
            <span>Show pentatonic shapes</span>
          </label>
        </div>

        <div
          className={`fretboard-host${improvEnabled ? ' improv-on' : ''}${!improvEnabled && solos.length ? ' solo-only' : ''}`}
          data-has-solos={solos.length > 0 ? 'true' : 'false'}
        >
          {!improvEnabled && solos.length > 0 && (
            <div className="fretboard-overlay-text">SOLO SECTION</div>
          )}

          <PentatonicFretboard
            ref={ref}
            guideData={improvEnabled ? guideData : null}
            showLegend={false}
          >
            {soloNotes}
          </PentatonicFretboard>
        </div>

        {improvEnabled && !songKey && (
          <p className="fretboard-hint">Key not detected — re-analyze to enable the guide.</p>
        )}
      </div>
    );
  },
);

FretboardViewer.displayName = 'FretboardViewer';
