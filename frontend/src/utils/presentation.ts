const AB_PREF_KEY = 'chordlift-presentation-ab';

export type PresentationMode = 'synced' | 'raw';

export function loadPresentationMode(): PresentationMode {
  const v = localStorage.getItem(AB_PREF_KEY);
  return v === 'raw' ? 'raw' : 'synced';
}

export function savePresentationMode(mode: PresentationMode): void {
  localStorage.setItem(AB_PREF_KEY, mode);
}
