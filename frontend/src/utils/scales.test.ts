import { describe, expect, it } from 'vitest';
import {
  SCALE_REGISTRY,
  buildScaleGuideData,
  scalePitchClasses,
} from './scales';

const OPEN_STRING_PC: Record<number, number> = {
  1: 4, 2: 11, 3: 7, 4: 2, 5: 9, 6: 4,
};

describe('scales registry', () => {
  it('has 9 scales with exact interval counts', () => {
    expect(SCALE_REGISTRY).toHaveLength(9);
    expect(SCALE_REGISTRY.find((s) => s.id === 'dorian')?.intervals).toEqual([0, 2, 3, 5, 7, 9, 10]);
  });

  it('every rendered dot belongs to the selected scale', () => {
    for (const scale of SCALE_REGISTRY) {
      const data = buildScaleGuideData('A', scale.id, 18, 0);
      const allowed = new Set(scalePitchClasses('A', scale));
      const rootPc = 9; // A

      for (const dot of data.dots) {
        const pc = (OPEN_STRING_PC[dot.string] + dot.fret) % 12;
        expect(allowed.has(pc)).toBe(true);
      }

      const rootDots = data.dots.filter((d) => d.isKeyRoot);
      for (const dot of rootDots) {
        expect((OPEN_STRING_PC[dot.string] + dot.fret) % 12).toBe(rootPc);
      }
    }
  });
});
