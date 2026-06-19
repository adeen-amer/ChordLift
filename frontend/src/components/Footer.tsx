import { Link } from 'react-router-dom';

export function Footer() {
  return (
    <footer className="site-footer">
      <nav className="site-footer-nav" aria-label="Footer">
        <Link to="/about">About</Link>
        <span className="site-footer-sep" aria-hidden="true">·</span>
        <Link to="/privacy">Privacy Policy</Link>
      </nav>
      <p className="site-footer-copy">© ChordLift</p>
    </footer>
  );
}
