import { Outlet } from 'react-router-dom';
import { CookieConsentBanner } from './CookieConsentBanner';
import { Footer } from './Footer';

export function AppLayout() {
  return (
    <>
      <Outlet />
      <Footer />
      <CookieConsentBanner />
    </>
  );
}
