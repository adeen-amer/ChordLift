import {
  getImprovGuideData,
  getKeyCagedBoxes,
  keyInfoFromSelection,
  noteNameAt,
  normalizeRoot,
  rootToPc,
  type BoxBounds,
  type ImprovGuideData,
  type PentatonicMode,
} from './pentatonic';

export { PENTATONIC_ROOTS } from './pentatonic';

export type ScaleId =
  | 'major-penta'
  | 'minor-penta'
  | 'ionian'
  | 'dorian'
  | 'phrygian'
  | 'lydian'
  | 'mixolydian'
  | 'aeolian'
  | 'locrian';

export type LabelMode = 'note' | 'degree';

export interface ScaleDefinition {
  id: ScaleId;
  label: string;
  group: 'pentatonic' | 'modes';
  intervals: readonly number[];
  degrees: readonly string[];
}

/** Open-string pitch classes: string 1 = high e … string 6 = low E. */
const OPEN_STRING_PC: Record<number, number> = {
  1: 4, 2: 11, 3: 7, 4: 2, 5: 9, 6: 4,
};

export const SCALE_REGISTRY: ScaleDefinition[] = [
  { id: 'major-penta', label: 'Major pentatonic', group: 'pentatonic', intervals: [0, 2, 4, 7, 9], degrees: ['1', '2', '3', '5', '6'] },
  { id: 'minor-penta', label: 'Minor pentatonic', group: 'pentatonic', intervals: [0, 3, 5, 7, 10], degrees: ['1', '♭3', '4', '5', '♭7'] },
  { id: 'ionian', label: 'Ionian (Major)', group: 'modes', intervals: [0, 2, 4, 5, 7, 9, 11], degrees: ['1', '2', '3', '4', '5', '6', '7'] },
  { id: 'dorian', label: 'Dorian', group: 'modes', intervals: [0, 2, 3, 5, 7, 9, 10], degrees: ['1', '2', '♭3', '4', '5', '6', '♭7'] },
  { id: 'phrygian', label: 'Phrygian', group: 'modes', intervals: [0, 1, 3, 5, 7, 8, 10], degrees: ['1', '♭2', '♭3', '4', '5', '♭6', '♭7'] },
  { id: 'lydian', label: 'Lydian', group: 'modes', intervals: [0, 2, 4, 6, 7, 9, 11], degrees: ['1', '2', '3', '♯4', '5', '6', '7'] },
  { id: 'mixolydian', label: 'Mixolydian', group: 'modes', intervals: [0, 2, 4, 5, 7, 9, 10], degrees: ['1', '2', '3', '4', '5', '6', '♭7'] },
  { id: 'aeolian', label: 'Aeolian (Minor)', group: 'modes', intervals: [0, 2, 3, 5, 7, 8, 10], degrees: ['1', '2', '♭3', '4', '5', '♭6', '♭7'] },
  { id: 'locrian', label: 'Locrian', group: 'modes', intervals: [0, 1, 3, 5, 6, 8, 10], degrees: ['1', '♭2', '♭3', '4', '♭5', '♭6', '♭7'] },
];

export function getScaleById(id: ScaleId): ScaleDefinition {
  const scale = SCALE_REGISTRY.find((s) => s.id === id);
  if (!scale) throw new Error(`Unknown scale: ${id}`);
  return scale;
}

export function isPentatonicScaleId(id: ScaleId): boolean {
  return id === 'major-penta' || id === 'minor-penta';
}

function pentatonicModeForScale(id: ScaleId): PentatonicMode {
  return id === 'major-penta' ? 'major' : 'minor';
}

export function scalePitchClasses(root: string, scale: ScaleDefinition): number[] {
  const rootPc = rootToPc(root);
  if (rootPc === null) return [];
  return scale.intervals.map((iv) => (rootPc + iv) % 12);
}

function degreeForPc(rootPc: number, pc: number, scale: ScaleDefinition): string {
  const interval = (pc - rootPc + 12) % 12;
  const idx = scale.intervals.indexOf(interval);
  return idx >= 0 ? scale.degrees[idx] : '?';
}

function positionAnchors(root: string, maxFret: number): number[] {
  const key = keyInfoFromSelection(root, 'minor');
  return getKeyCagedBoxes(key, maxFret).map((box) => box.anchorFret);
}

function fretWindow(anchor: number): { min: number; max: number } {
  return { min: Math.max(0, anchor - 1), max: anchor + 4 };
}

function inWindow(fret: number, min: number, max: number): boolean {
  return fret >= min && fret <= max;
}

export interface ScaleGuideDot {
  string: number;
  fret: number;
  isKeyRoot: boolean;
  noteName: string;
  degreeLabel: string;
  boxes: number[];
  dimmed: boolean;
}

export interface ScaleGuideBox {
  number: number;
  anchorFret: number;
  bounds: BoxBounds;
}

export interface ScaleGuideData {
  scaleLabel: string;
  noteCount: number;
  dots: ScaleGuideDot[];
  boxes: ScaleGuideBox[];
  isPentatonic: boolean;
}

function boxesFromImprov(data: ImprovGuideData): ScaleGuideBox[] {
  return data.boxes.map((b) => ({
    number: b.number,
    anchorFret: b.anchorFret,
    bounds: b.bounds,
  }));
}

function convertPentatonicGuide(
  data: ImprovGuideData,
  scale: ScaleDefinition,
  rootPc: number,
  positionFilter: number,
): ScaleGuideData {
  const dots: ScaleGuideDot[] = data.dots.map((dot) => {
    const pc = (OPEN_STRING_PC[dot.string] + dot.fret) % 12;
    const dimmed =
      positionFilter >= 1 && positionFilter <= 5
        ? !dot.boxes.includes(positionFilter)
        : false;
    return {
      string: dot.string,
      fret: dot.fret,
      isKeyRoot: dot.isKeyRoot,
      noteName: dot.noteName,
      degreeLabel: degreeForPc(rootPc, pc, scale),
      boxes: dot.boxes,
      dimmed,
    };
  });

  return {
    scaleLabel: scale.label,
    noteCount: scale.intervals.length,
    dots,
    boxes: boxesFromImprov(data),
    isPentatonic: true,
  };
}

function buildDiatonicGuide(
  root: string,
  scale: ScaleDefinition,
  maxFret: number,
  positionFilter: number,
): ScaleGuideData {
  const rootPc = rootToPc(root)!;
  const pcs = new Set(scalePitchClasses(root, scale));
  const anchors = positionAnchors(root, maxFret);

  const boxes: ScaleGuideBox[] = anchors.map((anchorFret, i) => {
    const win = fretWindow(anchorFret);
    return {
      number: i + 1,
      anchorFret,
      bounds: {
        minFret: win.min,
        maxFret: win.max,
        minString: 1,
        maxString: 6,
      },
    };
  });

  const activeWindow =
    positionFilter >= 1 && positionFilter <= 5
      ? fretWindow(anchors[positionFilter - 1] ?? 0)
      : null;

  const dotMap = new Map<string, ScaleGuideDot>();

  for (let string = 1; string <= 6; string++) {
    for (let fret = 0; fret <= maxFret; fret++) {
      const pc = (OPEN_STRING_PC[string] + fret) % 12;
      if (!pcs.has(pc)) continue;

      const key = `${string}-${fret}`;
      const dimmed = activeWindow ? !inWindow(fret, activeWindow.min, activeWindow.max) : false;
      const membership: number[] = [];
      boxes.forEach((box) => {
        if (inWindow(fret, box.bounds.minFret, box.bounds.maxFret)) {
          membership.push(box.number);
        }
      });

      dotMap.set(key, {
        string,
        fret,
        isKeyRoot: pc === rootPc,
        noteName: noteNameAt(string, fret),
        degreeLabel: degreeForPc(rootPc, pc, scale),
        boxes: membership,
        dimmed,
      });
    }
  }

  return {
    scaleLabel: scale.label,
    noteCount: scale.intervals.length,
    dots: Array.from(dotMap.values()),
    boxes,
    isPentatonic: false,
  };
}

export function buildScaleGuideData(
  root: string,
  scaleId: ScaleId,
  maxFret = 18,
  positionFilter = 0,
): ScaleGuideData {
  const scale = getScaleById(scaleId);
  const normalized = normalizeRoot(root);
  const rootPc = rootToPc(normalized);
  if (rootPc === null) {
    return { scaleLabel: scale.label, noteCount: 0, dots: [], boxes: [], isPentatonic: isPentatonicScaleId(scaleId) };
  }

  if (isPentatonicScaleId(scaleId)) {
    const mode = pentatonicModeForScale(scaleId);
    const key = keyInfoFromSelection(normalized, mode);
    const improv = getImprovGuideData(key, maxFret, null);
    return convertPentatonicGuide(improv, scale, rootPc, positionFilter);
  }

  return buildDiatonicGuide(normalized, scale, maxFret, positionFilter);
}

export function positionLabel(positionFilter: number, isPentatonic: boolean): string {
  if (positionFilter === 0) return 'full neck';
  if (isPentatonic) return `Box ${positionFilter}`;
  return `Position ${positionFilter}`;
}

/** Adapt legacy improv guide data for the shared fretboard component. */
export function improvGuideToScaleGuide(data: ImprovGuideData, label = 'Pentatonic'): ScaleGuideData {
  return {
    scaleLabel: label,
    noteCount: 5,
    dots: data.dots.map((dot) => ({
      string: dot.string,
      fret: dot.fret,
      isKeyRoot: dot.isKeyRoot,
      noteName: dot.noteName,
      degreeLabel: '',
      boxes: dot.boxes,
      dimmed: false,
    })),
    boxes: data.boxes.map((b) => ({
      number: b.number,
      anchorFret: b.anchorFret,
      bounds: b.bounds,
    })),
    isPentatonic: true,
  };
}
