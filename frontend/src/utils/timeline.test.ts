import { describe, expect, it } from 'vitest';
import { activeLyricIndex } from './lyricsSync';
import {
  activeLyricIndexBinary,
  activeRangeIndex,
  activeSegmentIndex,
  barIndexForTime,
  barNumbersForTimeline,
} from './timeline';

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

describe('barIndexForTime', () => {
  const downbeats = [0, 2, 4, 6];

  it('returns 0 before the first downbeat', () => {
    expect(barIndexForTime(downbeats, -1)).toBe(0);
  });

  it('finds the containing bar', () => {
    expect(barIndexForTime(downbeats, 0)).toBe(0);
    expect(barIndexForTime(downbeats, 1.9)).toBe(0);
    expect(barIndexForTime(downbeats, 2)).toBe(1);
    expect(barIndexForTime(downbeats, 5.5)).toBe(2);
  });

  it('clamps to the last bar past the final downbeat', () => {
    expect(barIndexForTime(downbeats, 100)).toBe(3);
  });

  it('returns 0 for an empty downbeat list', () => {
    expect(barIndexForTime([], 5)).toBe(0);
  });
});

describe('barNumbersForTimeline', () => {
  it('maps each chord to its bar index', () => {
    const downbeats = [0, 2, 4];
    const timeline = [{ time: 0 }, { time: 1 }, { time: 2.1 }, { time: 5 }];
    expect(barNumbersForTimeline(timeline, downbeats)).toEqual([0, 0, 1, 2]);
  });

  it('returns all zeros with no downbeat data', () => {
    const timeline = [{ time: 0 }, { time: 3 }];
    expect(barNumbersForTimeline(timeline, [])).toEqual([0, 0]);
  });
});
