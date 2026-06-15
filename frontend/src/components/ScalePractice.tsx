import { useMemo, useState } from 'react';
import {
  PENTATONIC_ROOTS,
  getImprovGuideData,
  keyInfoFromSelection,
  type PentatonicMode,
} from '../utils/pentatonic';
import { PentatonicFretboard } from './PentatonicFretboard';

const MODES: { value: PentatonicMode; label: string }[] = [
  { value: 'major', label: 'Major pentatonic' },
  { value: 'minor', label: 'Minor pentatonic' },
];

const BOX_OPTIONS = [
  { value: 0, label: 'All boxes' },
  { value: 1, label: 'Box 1' },
  { value: 2, label: 'Box 2' },
  { value: 3, label: 'Box 3' },
  { value: 4, label: 'Box 4' },
  { value: 5, label: 'Box 5' },
];

export function ScalePractice() {
  const [root, setRoot] = useState('A');
  const [mode, setMode] = useState<PentatonicMode>('minor');
  const [boxFilter, setBoxFilter] = useState(0);

  const key = useMemo(() => keyInfoFromSelection(root, mode), [root, mode]);

  const guideData = useMemo(
    () => getImprovGuideData(key, 18, boxFilter || null),
    [key, boxFilter],
  );

  return (
    <div className="scale-practice">
      <div className="glass-panel scale-practice-panel">
        <div className="scale-practice-header">
          <div>
            <h2 className="scale-practice-title">Scale practice</h2>
            <p className="scale-practice-subtitle">
              CAGED pentatonic shapes — no song required. Pick a key and box to drill.
            </p>
          </div>
        </div>

        <div className="scale-practice-controls">
          <label className="scale-control">
            <span>Key</span>
            <select value={root} onChange={(e) => setRoot(e.target.value)}>
              {PENTATONIC_ROOTS.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </label>

          <label className="scale-control">
            <span>Mode</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as PentatonicMode)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </label>

          <label className="scale-control">
            <span>Box</span>
            <select
              value={boxFilter}
              onChange={(e) => setBoxFilter(Number(e.target.value))}
            >
              {BOX_OPTIONS.map((b) => (
                <option key={b.value} value={b.value}>{b.label}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="scale-practice-meta">
          <span className="scale-practice-key">{key.display}</span>
          <span className="scale-practice-count">
            {guideData.dots.length} notes · {guideData.boxes.length} box
            {guideData.boxes.length !== 1 ? 'es' : ''}
          </span>
        </div>

        <PentatonicFretboard
          guideData={guideData}
          highlightBox={boxFilter || null}
          showLegend
        />
      </div>
    </div>
  );
}
