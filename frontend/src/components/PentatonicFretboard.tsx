import { forwardRef, useMemo, type ReactNode } from 'react';
import {
  FRETBOARD,
  OPEN_STRING_LABELS,
  boundsToSvgRect,
  fretboardSpacing,
  fretLabelX,
  stringFretToXY,
} from '../utils/fretboardLayout';
import type { ScaleGuideData } from '../utils/scales';
import type { LabelMode } from '../utils/scales';

const FRET_MARKERS: { fret: number; strings: number[] }[] = [
  { fret: 3, strings: [3] },
  { fret: 5, strings: [3] },
  { fret: 7, strings: [3] },
  { fret: 9, strings: [3] },
  { fret: 12, strings: [2, 4] },
  { fret: 15, strings: [3] },
  { fret: 17, strings: [3] },
];

interface PentatonicFretboardProps {
  guideData?: ScaleGuideData | null;
  highlightPosition?: number | null;
  labelMode?: LabelMode;
  showLegend?: boolean;
  className?: string;
  children?: ReactNode;
}

export const PentatonicFretboard = forwardRef<SVGSVGElement, PentatonicFretboardProps>(
  (
    {
      guideData,
      highlightPosition = null,
      labelMode = 'note',
      showLegend = true,
      className = '',
      children,
    },
    ref,
  ) => {
    const { numFrets, numStrings, width, height, fretLabelY, stringLabelX } = FRETBOARD;
    const { fretSpacing, stringSpacing, xOffset, yOffset } = fretboardSpacing();

    const boxRects = useMemo(
      () =>
        guideData?.boxes.map((box) => ({
          number: box.number,
          rect: boundsToSvgRect(box.bounds),
        })) ?? [],
      [guideData],
    );

    const activePosition = highlightPosition && highlightPosition >= 1 ? highlightPosition : null;

    const visibleDots = useMemo(() => guideData?.dots ?? [], [guideData]);

    const fretNumbers = useMemo(
      () => Array.from({ length: numFrets }, (_, i) => i + 1),
      [numFrets],
    );

    return (
      <div className={`fretboard-scroll${guideData ? ' improv-active' : ''} ${className}`.trim()}>
        <div className={`fretboard-wrapper${guideData ? ' improv-active' : ''}`}>
          <svg className="fretboard-svg" viewBox={`0 0 ${width} ${height}`} ref={ref}>
            {/* Fret wire lines */}
            {Array.from({ length: numFrets + 1 }).map((_, i) => (
              <line
                key={`fret-${i}`}
                className={i === 0 ? 'fret-wire nut-wire' : 'fret-wire'}
                x1={xOffset + i * fretSpacing}
                y1={yOffset - 6}
                x2={xOffset + i * fretSpacing}
                y2={height - yOffset - 8}
              />
            ))}

            {/* Strings — thicker toward bass (bottom) */}
            {Array.from({ length: numStrings }).map((_, i) => (
              <line
                key={`string-${i}`}
                className="string"
                x1={xOffset}
                y1={yOffset + i * stringSpacing}
                x2={width - 16}
                y2={yOffset + i * stringSpacing}
                strokeWidth={1.1 + i * 0.45}
              />
            ))}

            {/* Open-string labels in left margin */}
            {OPEN_STRING_LABELS.map((label, i) => (
              <text
                key={`open-label-${i}`}
                className="open-string-label"
                x={stringLabelX}
                y={yOffset + i * stringSpacing}
              >
                {label}
              </text>
            ))}

            {/* Fret number row */}
            {fretNumbers.map((fret) => (
              <text
                key={`fret-label-${fret}`}
                className="fret-number-label"
                x={fretLabelX(fret)}
                y={fretLabelY}
              >
                {fret}
              </text>
            ))}

            {/* Standard inlay markers */}
            {FRET_MARKERS.flatMap(({ fret, strings }) =>
              strings.map((stringNum) => {
                const stringIdx = stringNum - 1;
                return (
                  <circle
                    key={`marker-${fret}-${stringNum}`}
                    className="fret-marker"
                    cx={xOffset + (fret - 0.5) * fretSpacing}
                    cy={yOffset + stringIdx * stringSpacing}
                    r={5}
                  />
                );
              }),
            )}

            {guideData && (
              <g className="scale-layer">
                {boxRects.map(({ number, rect }) => (
                  <g
                    key={`region-${number}`}
                    id={`caged-box-${number}`}
                    className={`scale-region-group caged-box-group${activePosition === number ? ' active-region active-chord-box' : ''}`}
                  >
                    <rect
                      className="scale-region-outline caged-box-outline"
                      data-box={number}
                      x={rect.x}
                      y={rect.y}
                      width={rect.width}
                      height={rect.height}
                    />
                    {guideData.isPentatonic && (
                      <text className="scale-region-number caged-box-number" x={rect.x + 8} y={rect.y + 14}>
                        {number}
                      </text>
                    )}
                  </g>
                ))}

                <rect id="pent-box-highlight" className="pent-box-highlight scale-box-highlight" opacity={0} />
                <text id="pent-chord-label" className="pent-chord-label scale-chord-label" opacity={0}>
                  {' '}
                </text>

                {visibleDots.map(({ string, fret, isKeyRoot, noteName, degreeLabel, dimmed, boxes }) => {
                  const { x, y } = stringFretToXY(string, fret);
                  const key = `${string}-${fret}`;
                  const displayLabel =
                    labelMode === 'degree'
                      ? degreeLabel
                      : noteName.replace('#', '♯');

                  return (
                    <g
                      key={`dot-${key}`}
                      id={`pent-${key}`}
                      className={`scale-dot-group pent-dot-group${isKeyRoot ? ' scale-root-dot pent-root-dot' : ''}${dimmed ? ' dimmed' : ''}`}
                      data-string={string}
                      data-fret={fret}
                      data-is-root={isKeyRoot ? 'true' : 'false'}
                      data-boxes={boxes.join(',')}
                    >
                      {isKeyRoot && <circle className="scale-root-ring" cx={x} cy={y} r={14} />}
                      <circle className="scale-tone-dot pent-scale-dot" cx={x} cy={y} r={isKeyRoot ? 12 : 10} />
                      <text className="scale-dot-label pent-dot-label" x={x} y={y}>
                        {displayLabel}
                      </text>
                    </g>
                  );
                })}
              </g>
            )}

            {children}
          </svg>
        </div>

        {showLegend && guideData && (
          <div className="pent-legend pent-legend-below" aria-hidden="true">
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-root" />
              Root
            </span>
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-scale" />
              Scale tone
            </span>
            <span className="pent-legend-item">
              <span className="pent-legend-swatch pent-legend-active" />
              Position
            </span>
          </div>
        )}
      </div>
    );
  },
);

PentatonicFretboard.displayName = 'PentatonicFretboard';
