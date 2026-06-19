import { Link } from 'react-router-dom';

const LAST_UPDATED = 'June 13, 2026';
const CONTACT_EMAIL = 'adeenamer0@gmail.com';

export default function PrivacyPage() {
  return (
    <div className="app-container static-page">
      <header className="header">
        <h1>Privacy Policy</h1>
        <p>How ChordLift handles your data</p>
      </header>

      <article className="glass-panel static-page-content">
        <p className="static-page-meta">Last updated: {LAST_UPDATED}</p>

        <section>
          <h2>What ChordLift does</h2>
          <p>
            ChordLift is a guitar practice tool. You upload an audio file and we analyze it to detect
            chords, key, and related musical information so you can play along with synced chord
            diagrams. We do not offer user accounts. Audio you upload is processed to produce
            results and is not permanently stored on our servers — processing is transient only.
          </p>
        </section>

        <section>
          <h2>Third-party services</h2>
          <p>ChordLift relies on the following services to operate:</p>
          <ul>
            <li>
              <strong>Cloudflare Pages</strong> — hosts the public website and static frontend assets.
            </li>
            <li>
              <strong>Hugging Face Spaces</strong> — runs the backend that receives and processes
              uploaded audio for chord and key detection.
            </li>
            <li>
              <strong>Google AdSense</strong> — may display advertisements on the site. Google and
              its partners may set cookies and use similar technologies for ad delivery and
              measurement.
            </li>
          </ul>
        </section>

        <section>
          <h2>Cookies</h2>
          <p>We use two categories of cookies and similar storage:</p>
          <h3>Essential</h3>
          <p>
            These are required for basic site functionality. For example, we store your cookie
            consent choice in your browser&apos;s <code>localStorage</code> (key:{' '}
            <code>cookie-consent</code>) so we do not ask again on every visit. This preference is
            not used for advertising.
          </p>
          <h3>Advertising</h3>
          <p>
            If you accept advertising cookies, Google AdSense and third-party ad vendors may set
            cookies to show relevant ads, limit how often you see an ad, and measure campaign
            performance. Google&apos;s use of advertising cookies is described in their{' '}
            <a href="https://policies.google.com/technologies/ads" target="_blank" rel="noopener noreferrer">
              Advertising Technologies
            </a>{' '}
            policy. You can opt out of personalized advertising via{' '}
            <a href="https://www.google.com/settings/ads" target="_blank" rel="noopener noreferrer">
              Google Ads Settings
            </a>
            .
          </p>
          <p>
            If you decline advertising cookies, we will not load AdSense scripts based on your
            consent choice. Essential consent storage still applies.
          </p>
        </section>

        <section>
          <h2>Contact</h2>
          <p>
            Questions about this policy? Email us at{' '}
            <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
          </p>
        </section>

        <p className="static-page-back">
          <Link to="/">← Back to ChordLift</Link>
        </p>
      </article>
    </div>
  );
}
