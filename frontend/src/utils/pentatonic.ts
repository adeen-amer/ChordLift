import type { KeyInfo } from '../types';

const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

const FLAT_TO_SHARP: Record<string, string> = {
  Db: 'C#', Eb: 'D#', Gb: 'F#', Ab: 'G#', Bb: 'A#',
};

/** Open-string pitch classes: string 1 = high e … string 6 = low E. */
const OPEN_STRING_PC: Record<number, number> = {
  1: 4, 2: 11, 3: 7, 4: 2, 5: 9, 6: 4,
};

const MINOR_PENTA = [0, 3, 5, 7, 10];
const MAJOR_PENTA = [0, 2, 4, 7, 9];

type BoxPattern = Record<number, number[]>;

/**
 * CAGED minor pentatonic shapes (fret offsets from each box anchor on string 6).
 * Cross-checked against standard Am @ 5th-fret reference diagrams.
 */
const MINOR_BOX_PATTERNS: BoxPattern[] = [
  { 6: [0, 3], 5: [0, 2], 4: [0, 2], 3: [0, 2], 2: [0, 3], 1: [0, 3] },
  { 6: [0, 2], 5: [-1, 2], 4: [-3, -1], 3: [-1, 1], 2: [0, 2], 1: [0, 2] },
  { 6: [0, 2], 5: [0, 2], 4: [0, 2], 3: [-1, 2], 2: [0, 3], 1: [0, 2] },
  { 6: [0, 3], 5: [0, 3], 4: [0, 2], 3: [0, 2], 2: [-2, 1], 1: [0, 3] },
  { 6: [0, 2], 5: [0, 2], 4: [-1, 2], 3: [-1, 2], 2: [0, 2], 1: [0, 2] },
];

/** Major boxes reuse minor shapes rotated one position (same note pool, different root emphasis). */
const MAJOR_PATTERN_FROM_MINOR = [1, 2, 3, 4, 0] as const;

const MINOR_BOX_ANCHOR_OFFSETS = [0, 3, 5, 7, -2] as const;
const MAJOR_BOX_ANCHOR_OFFSETS = [0, 2, 4, 7, -3] as const;

export type PentatonicMode = 'major' | 'minor';

export const PENTATONIC_ROOTS = [
  'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
] as const;

export interface FretPosition {
  string: number;
  fret: number;
}

export interface ScalePosition extends FretPosition {
  isKeyRoot: boolean;
  noteName: string;
}

export interface BoxBounds {
  minFret: number;
  maxFret: number;
  minString: number;
  maxString: number;
}

export interface CagedBox {
  number: 1 | 2 | 3 | 4 | 5;
  anchorFret: number;
  positions: ScalePosition[];
  bounds: BoxBounds;
}

export interface ImprovGuideDot extends ScalePosition {
  boxes: number[];
}

export interface ImprovGuideData {
  boxes: CagedBox[];
  dots: ImprovGuideDot[];
}

export interface ParsedChord {
  root: string;
  rootPc: number;
  useMinorPenta: boolean;
}

export function normalizeRoot(root: string): string {
  return FLAT_TO_SHARP[root] ?? root;
}

export function rootToPc(root: string): number | null {
  const r = normalizeRoot(root);
  const idx = PITCH_CLASSES.indexOf(r as (typeof PITCH_CLASSES)[number]);
  return idx >= 0 ? idx : null;
}

export function noteNameAt(string: number, fret: number): string {
  const pc = (OPEN_STRING_PC[string] + fret) % 12;
  return PITCH_CLASSES[pc];
}

export function parseChord(chord: string): ParsedChord | null {
  if (!chord || chord === 'N') return null;
  const match = chord.match(/^([A-G](?:#|b)?)(.*)$/);
  if (!match) return null;
  const root = normalizeRoot(match[1]);
  const suffix = match[2] || '';
  const rootPc = rootToPc(root);
  if (rootPc === null) return null;

  const useMinorPenta =
    suffix.startsWith('m') && !suffix.startsWith('maj');

  return { root, rootPc, useMinorPenta };
}

export function modeUsesMinorPentatonic(mode: PentatonicMode | KeyInfo['mode']): boolean {
  return mode === 'minor' || mode === 'dorian';
}

export function keyUsesMinorPentatonic(key?: KeyInfo): boolean {
  if (!key) return false;
  return modeUsesMinorPentatonic(key.mode);
}

export function keyInfoFromSelection(root: string, mode: PentatonicMode): KeyInfo {
  const label = mode === 'major' ? 'Major' : 'Minor';
  return { root: normalizeRoot(root), mode, display: `${normalizeRoot(root)} ${label}` };
}

export function getKeyPentatonicPcs(key?: KeyInfo): number[] {
  if (!key) return [];
  const rootPc = rootToPc(key.root);
  if (rootPc === null) return [];

  const intervals = keyUsesMinorPentatonic(key) ? MINOR_PENTA : MAJOR_PENTA;
  return intervals.map((iv) => (rootPc + iv) % 12);
}

export function getPentatonicPitchClasses(root: string, mode: PentatonicMode): number[] {
  const rootPc = rootToPc(root);
  if (rootPc === null) return [];
  const intervals = modeUsesMinorPentatonic(mode) ? MINOR_PENTA : MAJOR_PENTA;
  return intervals.map((iv) => (rootPc + iv) % 12);
}

function lowestRootFretOnString6(rootPc: number): number {
  const open = OPEN_STRING_PC[6];
  for (let fret = 0; fret <= 12; fret++) {
    if ((open + fret) % 12 === rootPc) return fret;
  }
  return 0;
}

function buildBoxPositions(
  pattern: BoxPattern,
  anchorFret: number,
  maxFret: number,
  keyRootPc: number | null,
): ScalePosition[] {
  const positions: ScalePosition[] = [];

  for (let string = 1; string <= 6; string++) {
    for (const offset of pattern[string] ?? []) {
      const fret = anchorFret + offset;
      if (fret >= 0 && fret <= maxFret) {
        const pc = (OPEN_STRING_PC[string] + fret) % 12;
        positions.push({
          string,
          fret,
          isKeyRoot: keyRootPc !== null && pc === keyRootPc,
          noteName: PITCH_CLASSES[pc],
        });
      }
    }
  }

  return positions;
}

export function getBoxBounds(positions: FretPosition[]): BoxBounds | null {
  if (positions.length === 0) return null;
  let minFret = 99;
  let maxFret = 0;
  let minString = 6;
  let maxString = 1;
  for (const { string, fret } of positions) {
    minFret = Math.min(minFret, fret);
    maxFret = Math.max(maxFret, fret);
    minString = Math.min(minString, string);
    maxString = Math.max(maxString, string);
  }
  return { minFret, maxFret, minString, maxString };
}

function getBoxesForMode(
  root: string,
  mode: PentatonicMode,
  maxFret: number,
): CagedBox[] {
  const rootPc = rootToPc(root);
  if (rootPc === null) return [];

  const useMinor = modeUsesMinorPentatonic(mode);
  const baseAnchor = lowestRootFretOnString6(rootPc);
  const anchorOffsets = useMinor ? MINOR_BOX_ANCHOR_OFFSETS : MAJOR_BOX_ANCHOR_OFFSETS;

  return anchorOffsets.map((offset, index) => {
    const patternIndex = useMinor ? index : MAJOR_PATTERN_FROM_MINOR[index];
    const pattern = MINOR_BOX_PATTERNS[patternIndex];
    let anchorFret = baseAnchor + offset;
    while (anchorFret < 0) anchorFret += 12;
    const positions = buildBoxPositions(pattern, anchorFret, maxFret, rootPc);
    const bounds = getBoxBounds(positions);

    return {
      number: (index + 1) as 1 | 2 | 3 | 4 | 5,
      anchorFret,
      positions,
      bounds: bounds ?? { minFret: 0, maxFret: 0, minString: 1, maxString: 6 },
    };
  }).filter((box) => box.positions.length >= 6);
}

/** All five CAGED pentatonic boxes for the detected key. */
export function getKeyCagedBoxes(key: KeyInfo | undefined, maxFret = 18): CagedBox[] {
  if (!key) return [];
  const mode: PentatonicMode = keyUsesMinorPentatonic(key) ? 'minor' : 'major';
  return getBoxesForMode(key.root, mode, maxFret);
}

export function getImprovGuideData(
  key: KeyInfo | undefined,
  maxFret = 18,
  boxFilter?: number | null,
): ImprovGuideData {
  let boxes = getKeyCagedBoxes(key, maxFret);
  if (boxFilter && boxFilter >= 1 && boxFilter <= 5) {
    boxes = boxes.filter((b) => b.number === boxFilter);
  }

  const dotMap = new Map<string, ImprovGuideDot>();

  for (const box of boxes) {
    for (const pos of box.positions) {
      const keyStr = positionKey(pos.string, pos.fret);
      const existing = dotMap.get(keyStr);
      if (existing) {
        if (!existing.boxes.includes(box.number)) {
          existing.boxes.push(box.number);
        }
      } else {
        dotMap.set(keyStr, { ...pos, boxes: [box.number] });
      }
    }
  }

  return {
    boxes,
    dots: Array.from(dotMap.values()),
  };
}

/** Pick the CAGED box that contains the chord root on the fretboard. */
export function getActiveBoxForChord(chord: string, boxes: CagedBox[]): CagedBox | null {
  const parsed = parseChord(chord);
  if (!parsed || boxes.length === 0) return null;

  const containing = boxes.filter((box) =>
    box.positions.some(
      (p) => p.noteName === parsed.root || (p.isKeyRoot && p.noteName === parsed.root),
    ),
  );
  if (containing.length === 1) return containing[0];
  if (containing.length > 1) {
    const rootFret6 = lowestRootFretOnString6(parsed.rootPc);
    return containing.reduce((best, box) => {
      const hasLowE = box.positions.some((p) => p.string === 6 && p.fret === rootFret6);
      if (hasLowE) return box;
      return best;
    }, containing[0]);
  }

  const rootFret6 = lowestRootFretOnString6(parsed.rootPc);
  let best: CagedBox | null = null;
  let bestDistance = Infinity;
  for (const box of boxes) {
    if (rootFret6 < box.bounds.minFret - 2 || rootFret6 > box.bounds.maxFret + 2) continue;
    const distance = Math.abs(box.anchorFret - rootFret6);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = box;
    }
  }
  return best ?? boxes[0] ?? null;
}

export function positionKey(string: number, fret: number): string {
  return `${string}-${fret}`;
}

/** Dev helper: every note in a box must belong to the scale. */
export function boxMatchesScale(box: CagedBox, scalePcs: number[]): boolean {
  const scale = new Set(scalePcs);
  return box.positions.every((p) => scale.has((OPEN_STRING_PC[p.string] + p.fret) % 12));
}
