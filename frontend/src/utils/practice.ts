/**
 * Practice-mode helpers: playback speed and A/B section looping.
 *
 * Pure functions only — this repo has no component-rendering test infra, so
 * anything worth asserting lives here and is covered by practice.test.ts.
 */

export interface LoopRange {
  start: number;
  end: number;
}

/** Slow-down steps for working out a part, plus a couple above 1x. */
export const SPEED_STEPS = [0.5, 0.65, 0.75, 0.85, 1, 1.1, 1.25] as const;

export const DEFAULT_SPEED = 1;

/** Minimum loop length. Below this a loop just stutters and is unusable. */
export const MIN_LOOP_SEC = 0.5;

/**
 * Validate a pair of user-set loop points into an ordered, in-bounds range.
 *
 * Accepts the points in either order — the UI lets you set B before A — and
 * returns null when the range is unusable (missing point, out of bounds,
 * or shorter than MIN_LOOP_SEC) so callers can treat "no loop" as one case.
 */
export function normalizeLoop(
  a: number | null,
  b: number | null,
  duration: number,
): LoopRange | null {
  if (a === null || b === null) return null;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(duration)) return null;
  if (duration <= 0) return null;

  const start = Math.max(0, Math.min(a, b));
  const end = Math.min(duration, Math.max(a, b));

  if (end - start < MIN_LOOP_SEC) return null;
  return { start, end };
}

/**
 * Where playback should jump to, or null to leave it alone.
 *
 * Wraps at the end of the loop, and also pulls playback in when the user
 * seeks outside the range — without that, scrubbing away from an active loop
 * silently escapes it and the loop appears broken.
 */
export function loopSeekTarget(currentTime: number, loop: LoopRange | null): number | null {
  if (!loop) return null;
  if (currentTime >= loop.end || currentTime < loop.start) return loop.start;
  return null;
}

/** Step the speed one notch through SPEED_STEPS, clamped at both ends. */
export function stepSpeed(current: number, direction: 1 | -1): number {
  const idx = SPEED_STEPS.indexOf(current as (typeof SPEED_STEPS)[number]);
  // An unknown speed (e.g. restored from an older session) snaps to 1x rather
  // than getting stuck, since indexOf would return -1 and drift from there.
  const from = idx === -1 ? SPEED_STEPS.indexOf(DEFAULT_SPEED) : idx;
  const next = Math.max(0, Math.min(SPEED_STEPS.length - 1, from + direction));
  return SPEED_STEPS[next];
}

export function formatSpeed(rate: number): string {
  return `${Number.isInteger(rate) ? rate : rate.toFixed(2).replace(/0$/, '')}x`;
}
