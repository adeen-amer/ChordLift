import { forwardRef, useMemo, type ReactNode } from 'react';
import {
  FRETBOARD,
  boundsToSvgRect,
  fretboardSpacing,
  fretLabelX,
  stringFretToXY,
} from '../utils/fretboardLayout';
import type { ImprovGuideData } from '../utils/pentatonic';
import { positionKey } from '../utils/pentatonic';

const FRET_LABELS = [3, 5, 7, 9, 12, 15, 17];

interface PentatonicFretboardProps {
  guideData?: ImprovGuideData | null;
  highlightBox?: number | null;
  showLegend?: boolean;
  className?: string;
  children?: ReactNode;
}

export const PentatonicFretboard = forwardRef<SVGSVGElement, PentatonicFretboardProps>(
  ({ guideData, highlightBox = null, showLegend = true, className = '', children }, ref) => {
    const { numFrets, numStrings, width, height, fretLabelY } = FRETBOARD;
    const { fretSpacing, stringSpacing, xOffset, yOffset } = fretboardSpacing();

    const boxRects = useMemo(
      () =>
        guideData?.boxes.map((box) => ({
          number: box.number,
          rect: boundsToSvgRect(box.bounds),
        })) ?? [],
      [guideData],
    );

    const visibleBoxes = highlightBox
      ? boxRects.filter((b) => b.number === highlightBox)
      : boxRects;

    const visibleDots = highlightBox
      ? guideData?.dots.filter((d) => d.boxes.includes(highlightBox)) ?? []
      : guideData?.dots ?? [];

    return (
      <div className={`fretboard-wrapper${guideData ? ' improv-active' : ''} ${className}`.trim()}>
        <svg className="fretboard-svg" viewBox={`0 0 ${width} ${height}`} ref={ref}>
          {Array.from({ length: numFrets + 1 }).map((_, i) => (
            <line
              key={`fret-${i}`}
              x1={xOffset + i * fretSpacing}
              y1={yOffset}
              x2={xOffset + i * fretSpacing}
              y2={height - yOffset - 14}
              strokeWidth={i === 0 ? 6 : 2}
              stroke={i === 0 ? '#ffffff' : '#4a4a4a'}
            />
          ))}

          {Array.from({ length: numStrings }).map((_, i) => (
            <line
              key={`string-${i}`}
              className="string"
              x1={xOffset}
              y1={yOffset + i * stringSpacing}
              x2={width}
              y2={yOffset + i * stringSpacing}
              strokeWidth={1 + i * 0.5}
            />
          ))}

          {FRET_LABELS.map((fret) => (
            <text
              key={`fret-label-${fret}`}
              className="fret-number-label"
              x={fretLabelX(fret)}
              y={fretLabelY}
            >
              {fret}
            </text>
          ))}

          {[3, 5, 7, 9, 15, 17].map((fret) => (
            <circle
              key={`marker-${fret}`}
              className="fret-marker"
              cx={xOffset + (fret - 0.5) * fretSpacing}
              cy={yOffset + 2.5 * stringSpacing}
              r={8}
            />
          ))}
          <circle
            className="fret-marker"
            cx={xOffset + 11.5 * fretSpacing}
            cy={yOffset + 2 * stringSpacing}
            r={8}
          />
          <circle
            className="fret-marker"
            cx={xOffset + 11.5 * fretSpacing}
            cy={yOffset + 3 * stringSpacing}
            r={8}
          />

          {guideData && (
            <g className="pentatonic-layer">
              {visibleBoxes.map(({ number, rect }) => (
                <g
                  key={`caged-box-${number}`}
                  id={`caged-box-${number}`}
                  className="caged-box-group"
                >
                  <rect
                    className="caged-box-outline"
                    data-box={number}
                    x={rect.x}
                    y={rect.y}
                    width={rect.width}
                    height={rect.height}
                  />
                  <text className="caged-box-number" x={rect.x + 8} y={rect.y + 14}>
                    {number}
                  </text>
                </g>
              ))}

              <rect id="pent-box-highlight" className="pent-box-highlight" opacity={0} />
              <text id="pent-chord-label" className="pent-chord-label" opacity={0}>
                {' '}
              </text>

              {visibleDots.map(({ string, fret, isKeyRoot, noteName, boxes }) => {
                const { x, y } = stringFretToXY(string, fret);
                const key = positionKey(string, fret);
                const radius = isKeyRoot ? 11 : 10;

                return (
                  <g
                    key={`pent-${key}`}
                    id={`pent-${key}`}
                    className={`pent-dot-group${isKeyRoot ? ' pent-root-dot' : ''}`}
                    data-string={string}
                    data-fret={fret}
                    data-is-root={isKeyRoot ? 'true' : 'false'}
                    data-boxes={boxes.join(',')}
                  >
                    <circle className="pent-scale-dot" cx={x} cy={y} r={radius} />
                    <text className="pent-dot-label" x={x} y={y}>
                      {noteName.replace('#', '♯')}
                    </text>
                  </g>
                );
              })}
            </g>
          )}

          {children}
        </svg>

        {showLegend && guideData && (
          <div className="pent-legend pent-legend-below" aria-hidden="true">
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-scale" />
              Scale
            </span>
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-root" />
              Key root
            </span>
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-active" />
              Chord box
            </span>
          </div>
        )}
      </div>
    );
  },
);

PentatonicFretboard.displayName = 'PentatonicFretboard';
