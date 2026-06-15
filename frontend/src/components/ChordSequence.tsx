import { forwardRef } from 'react';
import type { ChordEvent } from '../types';
import { ChordDiagram } from './ChordDiagram';

interface ChordSequenceProps {
  timeline: ChordEvent[];
}

export const ChordSequence = forwardRef<HTMLDivElement, ChordSequenceProps>(
  ({ timeline }, ref) => (
    <div className="chord-sequence-wrap">
      <div className="chord-container" ref={ref}>
        {timeline.map((chord, idx) => (
          <div
            key={idx}
            className={`chord-card${chord.is_low_confidence ? ' low-confidence' : ''}`}
            id={`chord-${idx}`}
          >
            <div className="chord-name">
              {chord.chord}
              {chord.is_low_confidence && (
                <span className="low-confidence" title="Low confidence detection">?</span>
              )}
            </div>
            <ChordDiagram chordName={chord.chord} />
            <div className="strumming">{chord.strumming || 'D DU UDU'}</div>
          </div>
        ))}
      </div>
    </div>
  ),
);

ChordSequence.displayName = 'ChordSequence';
