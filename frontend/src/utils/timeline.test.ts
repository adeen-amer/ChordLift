import { describe, expect, it } from 'vitest';
import { activeLyricIndex } from './lyricsSync';
import { activeLyricIndexBinary, activeRangeIndex, activeSegmentIndex } from './timeline';

describe('activeSegmentIndex', () => {
  const segments = [
    { time: 0, end_time: 2 },
    { time: 2, end_time: 5 },
    { time: 5, end_time: 10 },
  ];

  it('returns -1 before first segment', () => {
    expect(activeSegmentIndex(segments, -1)).toBe(-1);
  });

  it('finds active segment', () => {
    expect(activeSegmentIndex(segments, 2.5)).toBe(1);
    expect(activeSegmentIndex(segments, 5)).toBe(2);
  });

  it('returns -1 in gaps between segments', () => {
    expect(activeSegmentIndex([{ time: 0, end_time: 1 }, { time: 3, end_time: 4 }], 2)).toBe(-1);
  });
});

describe('activeRangeIndex', () => {
  it('finds solo-style ranges', () => {
    const solos = [{ start: 10, end: 20 }, { start: 30, end: 40 }];
    expect(activeRangeIndex(solos, 15)).toBe(0);
    expect(activeRangeIndex(solos, 35)).toBe(1);
  });
});

describe('activeLyricIndexBinary', () => {
  const lines = [
    { time: 0, text: 'a' },
    { time: 10, text: 'b' },
    { time: 20, text: 'c' },
  ];

  it('matches linear scan', () => {
    for (const t of [-1, 0, 5, 10, 15, 20, 99]) {
      expect(activeLyricIndexBinary(lines, t)).toBe(activeLyricIndex(lines, t));
    }
  });
});
