/** Guitar chord diagram data. Strings: low E → high e (index 0–5). */

export interface ChordShape {
  frets: number[]; // -1 muted, 0 open, >0 absolute fret number
  fingers: number[];
  baseFret: number;
}

export interface PreparedShape extends ChordShape {
  /** Number of fret spaces drawn (4–5). */
  fretCount: number;
  /** True when no reliable shape exists. */
  unknown: boolean;
}

const ROOT_PC: Record<string, number> = {
  C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, F: 5,
  'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11,
};

const ENHARMONIC: Record<string, string> = {
  Db: 'C#', Eb: 'D#', Gb: 'F#', Ab: 'G#', Bb: 'A#',
};

/** Open-position shapes (preferred over generated barres). */
export const CHORD_SHAPES: Record<string, ChordShape> = {
  C:    { frets: [-1, 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0], baseFret: 1 },
  'C#': { frets: [-1, 4, 3, 1, 2, 1], fingers: [0, 4, 3, 1, 2, 1], baseFret: 1 },
  Db:   { frets: [-1, 4, 3, 1, 2, 1], fingers: [0, 4, 3, 1, 2, 1], baseFret: 1 },
  D:    { frets: [-1, -1, 0, 2, 3, 2], fingers: [0, 0, 0, 1, 3, 2], baseFret: 1 },
  'D#': { frets: [-1, -1, 1, 3, 4, 3], fingers: [0, 0, 1, 2, 4, 3], baseFret: 1 },
  Eb:   { frets: [-1, -1, 1, 3, 4, 3], fingers: [0, 0, 1, 2, 4, 3], baseFret: 1 },
  E:    { frets: [0, 2, 2, 1, 0, 0], fingers: [0, 2, 3, 1, 0, 0], baseFret: 1 },
  F:    { frets: [1, 3, 3, 2, 1, 1], fingers: [1, 3, 4, 2, 1, 1], baseFret: 1 },
  'F#': { frets: [2, 4, 4, 3, 2, 2], fingers: [1, 3, 4, 2, 1, 1], baseFret: 1 },
  Gb:   { frets: [2, 4, 4, 3, 2, 2], fingers: [1, 3, 4, 2, 1, 1], baseFret: 1 },
  G:    { frets: [3, 2, 0, 0, 0, 3], fingers: [2, 1, 0, 0, 0, 3], baseFret: 1 },
  'G#': { frets: [4, 3, 1, 1, 1, 4], fingers: [3, 2, 1, 1, 1, 4], baseFret: 1 },
  Ab:   { frets: [4, 3, 1, 1, 1, 4], fingers: [3, 2, 1, 1, 1, 4], baseFret: 1 },
  A:    { frets: [-1, 0, 2, 2, 2, 0], fingers: [0, 0, 1, 2, 3, 0], baseFret: 1 },
  'A#': { frets: [-1, 1, 3, 3, 3, 1], fingers: [0, 1, 2, 3, 4, 1], baseFret: 1 },
  Bb:   { frets: [-1, 1, 3, 3, 3, 1], fingers: [0, 1, 2, 3, 4, 1], baseFret: 1 },
  B:    { frets: [-1, 2, 4, 4, 4, 2], fingers: [0, 1, 2, 3, 4, 1], baseFret: 1 },

  Cm:   { frets: [-1, 3, 5, 5, 4, 3], fingers: [0, 1, 3, 4, 2, 1], baseFret: 3 },
  Dm:   { frets: [-1, -1, 0, 2, 3, 1], fingers: [0, 0, 0, 2, 3, 1], baseFret: 1 },
  Em:   { frets: [0, 2, 2, 0, 0, 0], fingers: [0, 2, 3, 0, 0, 0], baseFret: 1 },
  Fm:   { frets: [1, 3, 3, 1, 1, 1], fingers: [1, 3, 4, 1, 1, 1], baseFret: 1 },
  Gm:   { frets: [3, 5, 5, 3, 3, 3], fingers: [1, 3, 4, 1, 1, 1], baseFret: 3 },
  Am:   { frets: [-1, 0, 2, 2, 1, 0], fingers: [0, 0, 2, 3, 1, 0], baseFret: 1 },
  Bm:   { frets: [-1, 2, 4, 4, 3, 2], fingers: [0, 1, 3, 4, 2, 1], baseFret: 1 },

  C7:   { frets: [-1, 3, 2, 3, 1, 0], fingers: [0, 3, 2, 4, 1, 0], baseFret: 1 },
  D7:   { frets: [-1, -1, 0, 2, 1, 2], fingers: [0, 0, 0, 2, 1, 3], baseFret: 1 },
  E7:   { frets: [0, 2, 0, 1, 0, 0], fingers: [0, 2, 0, 1, 0, 0], baseFret: 1 },
  F7:   { frets: [1, 3, 1, 2, 1, 1], fingers: [1, 3, 1, 2, 1, 1], baseFret: 1 },
  G7:   { frets: [3, 2, 0, 0, 0, 1], fingers: [2, 1, 0, 0, 0, 3], baseFret: 1 },
  A7:   { frets: [-1, 0, 2, 0, 2, 0], fingers: [0, 0, 2, 0, 3, 0], baseFret: 1 },
  B7:   { frets: [-1, 2, 1, 2, 0, 2], fingers: [0, 2, 1, 3, 0, 4], baseFret: 1 },

  Cmaj7: { frets: [-1, 3, 2, 0, 0, 0], fingers: [0, 3, 2, 0, 0, 0], baseFret: 1 },
  Dmaj7: { frets: [-1, -1, 0, 2, 2, 2], fingers: [0, 0, 0, 1, 2, 3], baseFret: 1 },
  Emaj7: { frets: [0, 2, 1, 1, 0, 0], fingers: [0, 2, 1, 1, 0, 0], baseFret: 1 },
  Fmaj7: { frets: [1, 3, 2, 2, 1, 0], fingers: [1, 3, 2, 2, 1, 0], baseFret: 1 },
  Gmaj7: { frets: [3, 2, 0, 0, 0, 2], fingers: [2, 1, 0, 0, 0, 3], baseFret: 1 },
  Amaj7: { frets: [-1, 0, 2, 1, 2, 0], fingers: [0, 0, 2, 1, 3, 0], baseFret: 1 },

  Cm7:  { frets: [-1, 3, 1, 3, 1, 3], fingers: [0, 2, 1, 3, 1, 4], baseFret: 3 },
  Dm7:  { frets: [-1, -1, 0, 2, 1, 1], fingers: [0, 0, 0, 2, 1, 1], baseFret: 1 },
  Em7:  { frets: [0, 2, 0, 0, 0, 0], fingers: [0, 2, 0, 0, 0, 0], baseFret: 1 },
  Am7:  { frets: [-1, 0, 2, 0, 1, 0], fingers: [0, 0, 2, 0, 1, 0], baseFret: 1 },
  Bm7:  { frets: [-1, 2, 4, 2, 3, 2], fingers: [0, 1, 3, 1, 2, 1], baseFret: 1 },

  Csus2: { frets: [-1, 3, 0, 0, 1, 3], fingers: [0, 2, 0, 0, 1, 3], baseFret: 1 },
  Dsus2: { frets: [-1, -1, 0, 2, 3, 0], fingers: [0, 0, 0, 1, 3, 0], baseFret: 1 },
  Asus2: { frets: [-1, 0, 2, 2, 0, 0], fingers: [0, 0, 1, 2, 0, 0], baseFret: 1 },
  Csus4: { frets: [-1, 3, 3, 0, 1, 1], fingers: [0, 3, 4, 0, 1, 1], baseFret: 1 },
  Dsus4: { frets: [-1, -1, 0, 2, 3, 3], fingers: [0, 0, 0, 1, 3, 4], baseFret: 1 },
  Esus4: { frets: [0, 2, 2, 2, 0, 0], fingers: [0, 1, 2, 3, 0, 0], baseFret: 1 },
  Asus4: { frets: [-1, 0, 2, 2, 3, 0], fingers: [0, 0, 1, 2, 3, 0], baseFret: 1 },

  F5:  { frets: [1, 3, 3, -1, -1, -1], fingers: [1, 3, 4, 0, 0, 0], baseFret: 1 },
  C5:  { frets: [-1, 3, 5, -1, -1, -1], fingers: [0, 1, 3, 0, 0, 0], baseFret: 1 },
  Bb5: { frets: [-1, 1, 3, -1, -1, -1], fingers: [0, 1, 3, 0, 0, 0], baseFret: 1 },
  G5:  { frets: [3, 5, 5, -1, -1, -1], fingers: [1, 3, 4, 0, 0, 0], baseFret: 1 },
  D5:  { frets: [-1, -1, 0, 2, 3, -1], fingers: [0, 0, 0, 1, 3, 0], baseFret: 1 },
  E5:  { frets: [0, 2, 2, -1, -1, -1], fingers: [0, 1, 3, 0, 0, 0], baseFret: 1 },
  A5:  { frets: [-1, 0, 2, 2, -1, -1], fingers: [0, 0, 1, 3, 0, 0], baseFret: 1 },
};

/** E-root and A-root moveable barre templates (open position = fret 0 on that form). */
const E_FORM: Record<string, number[]> = {
  '': [0, 2, 2, 1, 0, 0],
  m: [0, 2, 2, 0, 0, 0],
  m7: [0, 2, 0, 0, 0, 0],
  '7': [0, 2, 0, 1, 0, 0],
  maj7: [0, 2, 1, 1, 0, 0],
  '5': [0, 2, 2, -1, -1, -1],
  sus4: [0, 2, 2, 2, 0, 0],
  sus2: [0, 2, 2, 0, 0, 2],
};

const A_FORM: Record<string, number[]> = {
  '': [-1, 0, 2, 2, 2, 0],
  m: [-1, 0, 2, 2, 1, 0],
  m7: [-1, 0, 2, 0, 1, 0],
  '7': [-1, 0, 2, 0, 2, 0],
  maj7: [-1, 0, 2, 1, 2, 0],
  '5': [-1, 0, 2, 2, -1, -1],
  sus4: [-1, 0, 2, 2, 3, 0],
  sus2: [-1, 0, 2, 2, 0, 0],
};

const E_ROOT_PC = 4;
const A_ROOT_PC = 9;

function cloneShape(shape: ChordShape): ChordShape {
  return { frets: [...shape.frets], fingers: [...shape.fingers], baseFret: shape.baseFret };
}

function parseChordName(chordName: string): { root: string; suffix: string } | null {
  const match = chordName.match(/^([A-G](?:#|b)?)(.*)$/);
  if (!match) return null;
  return { root: match[1], suffix: match[2] || '' };
}

function rootPc(root: string): number | undefined {
  return ROOT_PC[root] ?? ROOT_PC[ENHARMONIC[root] ?? ''];
}

function transposeFrets(frets: number[], semitones: number): number[] {
  return frets.map((f) => {
    if (f <= 0) return f;
    return f + semitones;
  });
}

function normalizeShape(shape: ChordShape): ChordShape {
  const positive = shape.frets.filter((f) => f > 0);
  if (positive.length === 0) {
    return { ...shape, baseFret: 1 };
  }
  const minFret = Math.min(...positive);
  const maxFret = Math.max(...positive);
  let baseFret = shape.baseFret;
  if (baseFret < 1 || minFret < baseFret) {
    baseFret = minFret;
  }
  // If shape spans more than 5 frets, slide base up so dots fit the window.
  if (maxFret - baseFret > 4) {
    baseFret = maxFret - 4;
  }
  return { ...shape, baseFret: Math.max(1, baseFret) };
}

function generateBarreShape(root: string, suffix: string): ChordShape | null {
  const targetPc = rootPc(root);
  if (targetPc === undefined) return null;
  const template = E_FORM[suffix] ?? A_FORM[suffix];
  if (!template) return null;

  const eBarre = (targetPc - E_ROOT_PC + 12) % 12;
  const aBarre = (targetPc - A_ROOT_PC + 12) % 12;

  // Prefer the lower barre; tie-break toward A-form for mid-neck chords (B, F#).
  const useE = eBarre < aBarre || (eBarre === aBarre && eBarre <= 4);
  const barre = useE ? eBarre : aBarre;
  const baseTemplate = useE ? E_FORM[suffix] : A_FORM[suffix];
  if (!baseTemplate) return null;

  const frets = transposeFrets(baseTemplate, barre);
  const positive = frets.filter((f) => f > 0);
  const baseFret = positive.length ? Math.min(...positive) : 1;
  return { frets, fingers: frets.map((f) => (f > 0 ? 1 : 0)), baseFret };
}

function lookupStatic(name: string): ChordShape | undefined {
  if (CHORD_SHAPES[name]) return cloneShape(CHORD_SHAPES[name]);
  const sharp = ENHARMONIC[name];
  if (sharp && CHORD_SHAPES[sharp]) return cloneShape(CHORD_SHAPES[sharp]);
  return undefined;
}

export function resolveChordShape(chordName: string): ChordShape {
  const cleaned = chordName.trim();
  if (!cleaned || cleaned === 'N') {
    return { frets: [-1, -1, -1, -1, -1, -1], fingers: [0, 0, 0, 0, 0, 0], baseFret: 1 };
  }

  const staticShape = lookupStatic(cleaned);
  if (staticShape) return normalizeShape(staticShape);

  const parsed = parseChordName(cleaned);
  if (parsed) {
    const { root, suffix } = parsed;
    const generated = generateBarreShape(root, suffix);
    if (generated) return normalizeShape(generated);

    // Flat/sharp root alias for generation
    const altRoot = ENHARMONIC[root];
    if (altRoot) {
      const altGen = generateBarreShape(altRoot, suffix);
      if (altGen) return normalizeShape(altGen);
    }

    // Last resort: same quality on enharmonic static entry
    const altKey = altRoot ? `${altRoot}${suffix}` : '';
    const altStatic = altKey ? lookupStatic(altKey) : undefined;
    if (altStatic) return normalizeShape(altStatic);

    // Never show a major triad for an explicit minor/7/sus chord
    if (suffix && !lookupStatic(root)) {
      const fallback = generateBarreShape(root, suffix.replace(/7$/, '').replace(/^maj/, ''))
        ?? generateBarreShape(root, 'm');
      if (fallback) return normalizeShape(fallback);
    }
  }

  const powerMatch = cleaned.match(/^([A-G][#b]?)5$/);
  if (powerMatch) {
    const gen = generateBarreShape(powerMatch[1], '5');
    if (gen) return normalizeShape(gen);
  }

  return { frets: [-1, -1, -1, -1, -1, -1], fingers: [0, 0, 0, 0, 0, 0], baseFret: 1 };
}

export function prepareShape(chordName: string): PreparedShape {
  const shape = resolveChordShape(chordName);
  const positive = shape.frets.filter((f) => f > 0);
  const unknown = positive.length === 0 && shape.frets.every((f) => f === -1);

  let maxRelative = 0;
  for (const fret of shape.frets) {
    if (fret > 0) {
      maxRelative = Math.max(maxRelative, fret - shape.baseFret + 1);
    }
  }
  const fretCount = Math.min(5, Math.max(4, maxRelative || 4));

  return { ...shape, fretCount, unknown };
}
