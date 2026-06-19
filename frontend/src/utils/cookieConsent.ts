const STORAGE_KEY = 'cookie-consent';

export type CookieConsentChoice = 'accepted' | 'declined';

export function getCookieConsent(): CookieConsentChoice | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value === 'accepted' || value === 'declined') return value;
    return null;
  } catch {
    return null;
  }
}

export function setCookieConsent(choice: CookieConsentChoice): void {
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    /* quota / private mode */
  }
}

/** True when the user accepted advertising cookies (gate future AdSense load). */
export function hasAdConsent(): boolean {
  return getCookieConsent() === 'accepted';
}
