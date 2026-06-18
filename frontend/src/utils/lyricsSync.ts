import type { LyricLine } from '../types';

/** Index of the LRC line active at playback time (-1 if before first line). */
export function activeLyricIndex(lines: LyricLine[], currentTime: number): number {
  if (!lines.length) return -1;
  let idx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].time <= currentTime) idx = i;
    else break;
  }
  return idx;
}

export type LyricLineState = 'past' | 'active' | 'upcoming';

export function lyricLineState(
  lineIndex: number,
  activeIndex: number,
): LyricLineState {
  if (activeIndex < 0) return 'upcoming';
  if (lineIndex < activeIndex) return 'past';
  if (lineIndex === activeIndex) return 'active';
  return 'upcoming';
}

/** Collect ordered unique lyric snippets from chord segments (unsynced fallback). */
export function plainLinesFromTimeline(
  timeline: { lyrics?: string }[],
): string[] {
  const out: string[] = [];
  for (const seg of timeline) {
    const text = seg.lyrics?.trim();
    if (!text) continue;
    const parts = text.split(/\s{2,}|\n+/).map((p) => p.trim()).filter(Boolean);
    for (const part of parts) {
      if (out[out.length - 1] !== part) out.push(part);
    }
  }
  return out;
}
