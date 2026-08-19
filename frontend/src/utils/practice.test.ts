import { describe, it, expect } from 'vitest';
import {
  canSeek,
  DEFAULT_SPEED,
  MIN_LOOP_SEC,
  SPEED_STEPS,
  formatSpeed,
  loopSeekTarget,
  normalizeLoop,
  snapToDownbeat,
  stepSpeed,
} from './practice';

describe('normalizeLoop', () => {
  it('orders the points regardless of which was set first', () => {
    expect(normalizeLoop(30, 10, 100)).toEqual({ start: 10, end: 30 });
    expect(normalizeLoop(10, 30, 100)).toEqual({ start: 10, end: 30 });
  });

  it('returns null until both points are set', () => {
    expect(normalizeLoop(null, 30, 100)).toBeNull();
    expect(normalizeLoop(10, null, 100)).toBeNull();
    expect(normalizeLoop(null, null, 100)).toBeNull();
  });

  it('clamps to the track bounds', () => {
    expect(normalizeLoop(-5, 30, 100)).toEqual({ start: 0, end: 30 });
    expect(normalizeLoop(80, 500, 100)).toEqual({ start: 80, end: 100 });
  });

  it('rejects a range shorter than MIN_LOOP_SEC', () => {
    expect(normalizeLoop(10, 10 + MIN_LOOP_SEC / 2, 100)).toBeNull();
    expect(normalizeLoop(10, 10, 100)).toBeNull();
  });

  it('accepts a range exactly at the minimum', () => {
    expect(normalizeLoop(10, 10 + MIN_LOOP_SEC, 100)).toEqual({
      start: 10,
      end: 10 + MIN_LOOP_SEC,
    });
  });

  it('returns null for an unloaded track (duration 0 or NaN)', () => {
    expect(normalizeLoop(10, 30, 0)).toBeNull();
    expect(normalizeLoop(10, 30, NaN)).toBeNull();
  });
});

describe('loopSeekTarget', () => {
  const loop = { start: 10, end: 30 };

  it('leaves playback alone inside the range', () => {
    expect(loopSeekTarget(10, loop)).toBeNull();
    expect(loopSeekTarget(20, loop)).toBeNull();
    expect(loopSeekTarget(29.9, loop)).toBeNull();
  });

  it('wraps to the start at the end of the range', () => {
    expect(loopSeekTarget(30, loop)).toBe(10);
    expect(loopSeekTarget(45, loop)).toBe(10);
  });

  it('pulls playback back in when the user seeks before the range', () => {
    expect(loopSeekTarget(5, loop)).toBe(10);
  });

  it('does nothing when no loop is set', () => {
    expect(loopSeekTarget(20, null)).toBeNull();
  });
});

describe('stepSpeed', () => {
  it('steps down and up through the ladder', () => {
    expect(stepSpeed(1, -1)).toBe(0.85);
    expect(stepSpeed(1, 1)).toBe(1.1);
  });

  it('clamps at both ends', () => {
    expect(stepSpeed(SPEED_STEPS[0], -1)).toBe(SPEED_STEPS[0]);
    expect(stepSpeed(SPEED_STEPS[SPEED_STEPS.length - 1], 1)).toBe(
      SPEED_STEPS[SPEED_STEPS.length - 1],
    );
  });

  it('snaps an unknown speed back to the default instead of drifting', () => {
    expect(stepSpeed(0.93, 1)).toBe(stepSpeed(DEFAULT_SPEED, 1));
  });
});

describe('formatSpeed', () => {
  it('renders whole and fractional rates readably', () => {
    expect(formatSpeed(1)).toBe('1x');
    expect(formatSpeed(0.5)).toBe('0.5x');
    expect(formatSpeed(0.65)).toBe('0.65x');
  });
});

describe('snapToDownbeat', () => {
  const downbeats = [0, 2, 4, 6, 8];

  it('snaps to the nearest bar line', () => {
    expect(snapToDownbeat(2.1, downbeats)).toBe(2);
    expect(snapToDownbeat(3.9, downbeats)).toBe(4);
    expect(snapToDownbeat(5.4, downbeats)).toBe(6);
  });

  it('keeps an exact bar line unchanged', () => {
    expect(snapToDownbeat(4, downbeats)).toBe(4);
  });

  it('snaps backwards on an exact tie', () => {
    expect(snapToDownbeat(3, downbeats)).toBe(2);
  });

  it('clamps to the first and last bar outside the grid', () => {
    expect(snapToDownbeat(-5, downbeats)).toBe(0);
    expect(snapToDownbeat(100, downbeats)).toBe(8);
  });

  it('leaves the point alone without downbeat data', () => {
    expect(snapToDownbeat(3.7, [])).toBe(3.7);
    expect(snapToDownbeat(3.7, undefined)).toBe(3.7);
  });
});

describe('canSeek', () => {
  it('is true for a normal seekable range', () => {
    expect(canSeek(332.12)).toBe(true);
  });

  it('is false when the server does not support byte ranges', () => {
    // Browser reports seekable as [0, 0] -- what ChordLift's own backend did
    // before the Range fix.
    expect(canSeek(0)).toBe(false);
    expect(canSeek(null)).toBe(false);
  });

  it('is false for a non-finite end', () => {
    expect(canSeek(NaN)).toBe(false);
    expect(canSeek(Infinity)).toBe(false);
  });
});
