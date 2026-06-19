import { useMemo, useState } from 'react';
import {
  PENTATONIC_ROOTS,
  SCALE_REGISTRY,
  buildScaleGuideData,
  isPentatonicScaleId,
  positionLabel,
  type LabelMode,
  type ScaleId,
} from '../utils/scales';
import { PentatonicFretboard } from './PentatonicFretboard';

const DEFAULT_SCALE: ScaleId = 'minor-penta';

export function ScalePractice() {
  const [root, setRoot] = useState('A');
  const [scaleId, setScaleId] = useState<ScaleId>(DEFAULT_SCALE);
  const [positionFilter, setPositionFilter] = useState(0);
  const [labelMode, setLabelMode] = useState<LabelMode>('note');

  const isPentatonic = isPentatonicScaleId(scaleId);

  const positionOptions = useMemo(() => {
    const all = { value: 0, label: 'All (full neck)' };
    const numbered = [1, 2, 3, 4, 5].map((n) => ({
      value: n,
      label: isPentatonic ? `Box ${n}` : `Position ${n}`,
    }));
    return [all, ...numbered];
  }, [isPentatonic]);

  const guideData = useMemo(
    () => buildScaleGuideData(root, scaleId, 18, positionFilter),
    [root, scaleId, positionFilter],
  );

  const metaLine = `${root} ${guideData.scaleLabel} · ${guideData.noteCount} notes · ${positionLabel(positionFilter, isPentatonic)}`;

  const pentatonicScales = SCALE_REGISTRY.filter((s) => s.group === 'pentatonic');
  const modeScales = SCALE_REGISTRY.filter((s) => s.group === 'modes');

  return (
    <div className="scale-practice">
      <div className="glass-panel scale-practice-panel">
        <div className="scale-practice-header">
          <div>
            <h2 className="scale-practice-title">Scale practice</h2>
            <p className="scale-practice-subtitle">
              Pentatonic CAGED boxes and seven diatonic modes — pick a key, scale, and neck position.
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
            <span>Scale</span>
            <select
              value={scaleId}
              onChange={(e) => {
                setScaleId(e.target.value as ScaleId);
                setPositionFilter(0);
              }}
            >
              <optgroup label="Pentatonic">
                {pentatonicScales.map((s) => (
                  <option key={s.id} value={s.id}>{s.label}</option>
                ))}
              </optgroup>
              <optgroup label="Modes">
                {modeScales.map((s) => (
                  <option key={s.id} value={s.id}>{s.label}</option>
                ))}
              </optgroup>
            </select>
          </label>

          <label className="scale-control">
            <span>Position</span>
            <select
              value={positionFilter}
              onChange={(e) => setPositionFilter(Number(e.target.value))}
            >
              {positionOptions.map((b) => (
                <option key={b.value} value={b.value}>{b.label}</option>
              ))}
            </select>
          </label>

          <label className="scale-control">
            <span>Labels</span>
            <select
              value={labelMode}
              onChange={(e) => setLabelMode(e.target.value as LabelMode)}
            >
              <option value="note">Note names</option>
              <option value="degree">Scale degrees</option>
            </select>
          </label>
        </div>

        <div className="scale-practice-meta">
          <span className="scale-practice-key">{metaLine}</span>
          <span className="scale-practice-count">
            {guideData.dots.filter((d) => !d.dimmed).length} visible · {guideData.dots.length} total
          </span>
        </div>

        <PentatonicFretboard
          guideData={guideData}
          highlightPosition={positionFilter || null}
          labelMode={labelMode}
          showLegend
        />
      </div>
    </div>
  );
}
