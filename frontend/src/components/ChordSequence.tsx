import { forwardRef, useState } from 'react';
import type { ChordEvent } from '../types';
import { ChordDiagram } from './ChordDiagram';
import { segmentTimeKey } from '../utils/chordCorrections';

interface ChordSequenceProps {
  timeline: ChordEvent[];
  onCorrectChord?: (time: number, chord: string) => void;
}

export const ChordSequence = forwardRef<HTMLDivElement, ChordSequenceProps>(
  ({ timeline, onCorrectChord }, ref) => {
    const [editingKey, setEditingKey] = useState<string | null>(null);
    const [draft, setDraft] = useState('');

    const startEdit = (seg: ChordEvent) => {
      if (!onCorrectChord) return;
      setEditingKey(segmentTimeKey(seg.time));
      setDraft(seg.chord);
    };

    const commitEdit = (seg: ChordEvent) => {
      if (!onCorrectChord) return;
      onCorrectChord(seg.time, draft.trim());
      setEditingKey(null);
      setDraft('');
    };

    return (
      <div className="chord-sequence-wrap">
        <div className="chord-container" ref={ref}>
          {timeline.map((chord) => {
            const segKey = segmentTimeKey(chord.time);
            const tier = chord.user_corrected
              ? 'high'
              : (chord.confidence_tier ?? (chord.is_low_confidence ? 'low' : 'high'));
            const tierClass = tier !== 'high' ? ` confidence-${tier}` : '';
            const adjustedClass = chord.display_adjusted ? ' display-adjusted' : '';
            const correctedClass = chord.user_corrected ? ' user-corrected' : '';

            return (
              <div
                key={segKey}
                className={`chord-card${tierClass}${adjustedClass}${correctedClass}`}
                id={`chord-${segKey}`}
                data-time={segKey}
                title={
                  chord.model_chord && chord.model_chord !== chord.chord
                    ? `Model: ${chord.model_chord}`
                    : undefined
                }
              >
                {editingKey === segKey ? (
                  <input
                    className="chord-edit-input"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => commitEdit(chord)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitEdit(chord);
                      if (e.key === 'Escape') setEditingKey(null);
                    }}
                    autoFocus
                    aria-label="Edit chord"
                  />
                ) : (
                  <div
                    className="chord-name"
                    role="button"
                    tabIndex={onCorrectChord ? 0 : undefined}
                    onDoubleClick={() => startEdit(chord)}
                    onKeyDown={(e) => {
                      if (onCorrectChord && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault();
                        startEdit(chord);
                      }
                    }}
                  >
                    {chord.chord}
                    {tier === 'low' && (
                      <span className="low-confidence" title="Low confidence detection">?</span>
                    )}
                    {tier === 'medium' && (
                      <span className="medium-confidence" title="Medium confidence">~</span>
                    )}
                    {chord.user_corrected && (
                      <span className="user-corrected-badge" title="Your correction">✎</span>
                    )}
                  </div>
                )}
                <ChordDiagram chordName={chord.chord} />
                {chord.strumming && chord.strumming_is_heuristic && (
                  <div
                    className="strumming"
                    title="Rhythm hint from onset density — not true strumming detection"
                  >
                    {chord.strumming}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {onCorrectChord && (
          <p className="chord-edit-hint">Double-click a chord to correct it (saved locally).</p>
        )}
      </div>
    );
  },
);

ChordSequence.displayName = 'ChordSequence';
