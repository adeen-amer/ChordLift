import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getCookieConsent, setCookieConsent } from '../utils/cookieConsent';

export function CookieConsentBanner() {
  const [visible, setVisible] = useState(() => getCookieConsent() === null);

  if (!visible) return null;

  const choose = (choice: 'accepted' | 'declined') => {
    setCookieConsent(choice);
    setVisible(false);
  };

  return (
    <div className="cookie-consent-banner" role="dialog" aria-label="Cookie consent">
      <p className="cookie-consent-text">
        We use essential cookies to remember your preferences and may use advertising cookies
        if you consent. See our{' '}
        <Link to="/privacy">Privacy Policy</Link> for details.
      </p>
      <div className="cookie-consent-actions">
        <button type="button" className="btn-secondary" onClick={() => choose('declined')}>
          Decline
        </button>
        <button type="button" className="btn-primary" onClick={() => choose('accepted')}>
          Accept
        </button>
      </div>
    </div>
  );
}
