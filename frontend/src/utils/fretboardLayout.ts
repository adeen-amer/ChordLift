/** Shared fretboard geometry (matches FretboardViewer SVG). */

export const FRETBOARD = {
  numFrets: 18,
  numStrings: 6,
  width: 1200,
  height: 240,
  yOffset: 24,
  xOffset: 44,
  fretLabelY: 228,
} as const;

export function fretboardSpacing() {
  const { width, height, yOffset, xOffset, numFrets, numStrings } = FRETBOARD;
  return {
    fretSpacing: (width - xOffset) / numFrets,
    stringSpacing: (height - 2 * yOffset - 16) / (numStrings - 1),
    xOffset,
    yOffset,
  };
}

/** String 1 = high e (top), string 6 = low E (bottom). */
export function stringFretToXY(stringNum: number, fret: number) {
  const { fretSpacing, stringSpacing, xOffset, yOffset } = fretboardSpacing();
  const stringIdx = stringNum - 1;
  const y = yOffset + stringIdx * stringSpacing;
  const x =
    fret === 0
      ? xOffset - 18
      : xOffset + (fret - 0.5) * fretSpacing;
  return { x, y };
}

export interface SvgRect {
  x: number;
  y: number;
  width: number;
  height: number;
  cx: number;
  cy: number;
}

export function boundsToSvgRect(
  bounds: { minFret: number; maxFret: number; minString: number; maxString: number },
  pad = 10,
): SvgRect {
  const { fretSpacing, stringSpacing, xOffset, yOffset } = fretboardSpacing();

  const minX =
    bounds.minFret === 0
      ? xOffset - 24
      : xOffset + (bounds.minFret - 1) * fretSpacing;
  const maxX = xOffset + bounds.maxFret * fretSpacing;
  const minY = yOffset + (bounds.minString - 1) * stringSpacing - pad;
  const maxY = yOffset + (bounds.maxString - 1) * stringSpacing + pad;

  const x = minX;
  const y = minY;
  const width = maxX - minX;
  const height = maxY - minY + 1;

  return {
    x,
    y,
    width,
    height,
    cx: x + width / 2,
    cy: y + height / 2,
  };
}

export function fretLabelX(fret: number): number {
  const { fretSpacing, xOffset } = fretboardSpacing();
  return fret === 0 ? xOffset - 18 : xOffset + (fret - 0.5) * fretSpacing;
}
