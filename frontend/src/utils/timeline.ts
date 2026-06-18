/** Half-open interval [start, end). */
export interface RangeSegment {
  start: number;
  end: number;
}

/**
 * Index of the range segment active at `t` for half-open intervals [start, end).
 */
export function activeRangeIndex(segments: RangeSegment[], t: number): number {
  if (!segments.length) return -1;

  let lo = 0;
  let hi = segments.length - 1;
  let candidate = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].start <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (candidate < 0) return -1;
  const seg = segments[candidate];
  if (t >= seg.start && t < seg.end) return candidate;
  return -1;
}

/** Time-sorted chord segment with time/end_time. */
export interface TimedSegment {
  time: number;
  end_time: number;
}

/** Index of the chord segment active at `t` (no per-call allocation). */
export function activeSegmentIndex(segments: TimedSegment[], t: number): number {
  if (!segments.length) return -1;

  let lo = 0;
  let hi = segments.length - 1;
  let candidate = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].time <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (candidate < 0) return -1;
  const seg = segments[candidate];
  return t >= seg.time && t < seg.end_time ? candidate : -1;
}

export function activeLyricIndexBinary(
  lines: { time: number }[],
  t: number,
): number {
  if (!lines.length) return -1;

  let lo = 0;
  let hi = lines.length - 1;
  let candidate = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lines[mid].time <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return candidate;
}
