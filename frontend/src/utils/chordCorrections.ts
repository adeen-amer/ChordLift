import type { ChordEvent } from '../types';

const STORAGE_PREFIX = 'chordlift-corrections:';
const MAX_CORRECTIONS_PER_SONG = 200;

export function correctionsKey(videoId: string): string {
  return `${STORAGE_PREFIX}${videoId}`;
}

/** Stable key across re-analyze (timeline length/order may change). */
export function segmentTimeKey(time: number): string {
  return (Math.round(time * 10) / 10).toFixed(1);
}

export type CorrectionsMap = Record<string, string>;

function parseStored(raw: string): CorrectionsMap {
  const parsed = JSON.parse(raw) as Record<string, string>;
  const out: CorrectionsMap = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (v.trim()) out[k] = v.trim();
  }
  return out;
}

export function loadCorrections(videoId: string): CorrectionsMap {
  try {
    const raw = localStorage.getItem(correctionsKey(videoId));
    if (!raw) return {};
    return parseStored(raw);
  } catch {
    return {};
  }
}

function pruneCorrections(all: CorrectionsMap): CorrectionsMap {
  const entries = Object.entries(all);
  if (entries.length <= MAX_CORRECTIONS_PER_SONG) return all;
  return Object.fromEntries(entries.slice(-MAX_CORRECTIONS_PER_SONG));
}

function persistCorrections(videoId: string, all: CorrectionsMap): void {
  const pruned = pruneCorrections(all);
  try {
    localStorage.setItem(correctionsKey(videoId), JSON.stringify(pruned));
  } catch {
    const keys = Object.keys(pruned);
    for (let n = keys.length; n > 0; n -= 10) {
      try {
        localStorage.setItem(
          correctionsKey(videoId),
          JSON.stringify(Object.fromEntries(keys.slice(-n).map((k) => [k, pruned[k]]))),
        );
        return;
      } catch {
        /* quota — drop oldest */
      }
    }
  }
}

export function saveCorrection(videoId: string, time: number, chord: string): void {
  const key = segmentTimeKey(time);
  const all = loadCorrections(videoId);
  if (chord.trim()) {
    all[key] = chord.trim();
  } else {
    delete all[key];
  }
  persistCorrections(videoId, all);
}

export function applyCorrections(
  timeline: ChordEvent[],
  corrections: CorrectionsMap,
): ChordEvent[] {
  if (!Object.keys(corrections).length) return timeline;
  return timeline.map((seg) => {
    const corrected = corrections[segmentTimeKey(seg.time)];
    if (!corrected || corrected === seg.chord) return seg;
    return {
      ...seg,
      chord: corrected,
      user_corrected: true,
      model_chord: seg.model_chord ?? seg.chord,
      confidence: 1,
      confidence_tier: 'high' as const,
      is_low_confidence: false,
    };
  });
}
