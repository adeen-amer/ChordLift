import { describe, expect, it, vi } from 'vitest';
import {
  applyCorrections,
  loadCorrections,
  saveCorrection,
  segmentTimeKey,
} from './chordCorrections';
import type { ChordEvent } from '../types';

describe('segmentTimeKey', () => {
  it('rounds to 100ms', () => {
    expect(segmentTimeKey(0.04)).toBe('0.0');
    expect(segmentTimeKey(2.06)).toBe('2.1');
  });
});

describe('chord corrections', () => {
  it('keys by time so re-ordered timeline still matches', () => {
    const storage: Record<string, string> = {};
    const mock = {
      getItem: (k: string) => storage[k] ?? null,
      setItem: (k: string, v: string) => { storage[k] = v; },
    };
    vi.stubGlobal('localStorage', mock);

    saveCorrection('vid1', 2.06, 'Em');
    const reordered: ChordEvent[] = [
      { time: 2.06, end_time: 4, chord: 'G', confidence: 0.8, is_low_confidence: false },
      { time: 0.04, end_time: 2, chord: 'C', confidence: 0.3, is_low_confidence: true },
    ];
    const out = applyCorrections(reordered, loadCorrections('vid1'));
    expect(out[0].chord).toBe('Em');
    expect(out[0].user_corrected).toBe(true);
    expect(out[0].confidence_tier).toBe('high');
    expect(out[0].is_low_confidence).toBe(false);

    vi.unstubAllGlobals();
  });
});
