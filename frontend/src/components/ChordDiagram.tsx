import { prepareShape } from '../utils/chordShapes';

interface ChordDiagramProps {
  chordName: string;
}

export const ChordDiagram = ({ chordName }: ChordDiagramProps) => {
  const shape = prepareShape(chordName);

  const width = 100;
  const height = 120;
  const xOffset = 20;
  const yOffset = 30;
  const stringSpacing = (width - 2 * xOffset) / 5;
  const fretSpacing = (height - yOffset - 10) / shape.fretCount;

  if (shape.unknown) {
    const label = chordName === 'N' ? '—' : chordName;
    return (
      <svg
        className="chord-diagram"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-label={chordName === 'N' ? 'No chord' : `No diagram for ${chordName}`}
      >
        <text
          x={width / 2}
          y={height / 2}
          fill="var(--text-muted)"
          fontSize="11"
          textAnchor="middle"
          dominantBaseline="middle"
        >
          {label}
        </text>
      </svg>
    );
  }

  const showNut = shape.baseFret === 1;

  return (
    <svg
      className="chord-diagram"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-label={`${chordName} chord diagram`}
    >
      {/* Nut or first fret line */}
      <line
        x1={xOffset}
        y1={yOffset}
        x2={width - xOffset}
        y2={yOffset}
        stroke="var(--text-main)"
        strokeWidth={showNut ? 4 : 2}
      />

      {shape.baseFret > 1 && (
        <text
          x={8}
          y={yOffset + fretSpacing / 2}
          fill="var(--text-muted)"
          fontSize="10"
          dominantBaseline="middle"
        >
          {shape.baseFret}fr
        </text>
      )}

      {/* Strings (low E → high e) */}
      {[0, 1, 2, 3, 4, 5].map((s) => (
        <line
          key={`string-${s}`}
          x1={xOffset + s * stringSpacing}
          y1={yOffset}
          x2={xOffset + s * stringSpacing}
          y2={yOffset + shape.fretCount * fretSpacing}
          stroke="var(--text-muted)"
          strokeWidth={1}
        />
      ))}

      {/* Fret wires */}
      {Array.from({ length: shape.fretCount }, (_, i) => i + 1).map((f) => (
        <line
          key={`fret-${f}`}
          x1={xOffset}
          y1={yOffset + f * fretSpacing}
          x2={width - xOffset}
          y2={yOffset + f * fretSpacing}
          stroke="var(--text-muted)"
          strokeWidth={1}
        />
      ))}

      {/* Finger dots / mutes / open strings */}
      {shape.frets.map((fret, stringIdx) => {
        const x = xOffset + stringIdx * stringSpacing;

        if (fret === -1) {
          return (
            <text
              key={`mark-${stringIdx}`}
              x={x}
              y={yOffset - 10}
              fill="var(--accent-primary)"
              fontSize="12"
              textAnchor="middle"
              fontWeight="bold"
            >
              ×
            </text>
          );
        }

        if (fret === 0) {
          return (
            <circle
              key={`mark-${stringIdx}`}
              cx={x}
              cy={yOffset - 12}
              r={3}
              fill="none"
              stroke="var(--text-main)"
              strokeWidth={1}
            />
          );
        }

        const relativeFret = fret - shape.baseFret + 1;
        if (relativeFret < 1 || relativeFret > shape.fretCount) {
          return null;
        }

        const y = yOffset + (relativeFret - 0.5) * fretSpacing;
        return (
          <circle
            key={`mark-${stringIdx}`}
            cx={x}
            cy={y}
            r={5}
            fill="var(--accent-secondary)"
          />
        );
      })}
    </svg>
  );
};
