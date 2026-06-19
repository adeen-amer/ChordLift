import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { getCookieConsent, hasAdConsent, setCookieConsent } from './cookieConsent';

describe('cookieConsent', () => {
  beforeEach(() => {
    const storage: Record<string, string> = {};
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => storage[k] ?? null,
      setItem: (k: string, v: string) => { storage[k] = v; },
      clear: () => { Object.keys(storage).forEach((k) => delete storage[k]); },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns null when unset', () => {
    expect(getCookieConsent()).toBeNull();
    expect(hasAdConsent()).toBe(false);
  });

  it('persists accepted choice', () => {
    setCookieConsent('accepted');
    expect(getCookieConsent()).toBe('accepted');
    expect(hasAdConsent()).toBe(true);
  });

  it('persists declined choice', () => {
    setCookieConsent('declined');
    expect(getCookieConsent()).toBe('declined');
    expect(hasAdConsent()).toBe(false);
  });
});
