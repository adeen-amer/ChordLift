import { Link } from 'react-router-dom';
import { Upload, Sparkles, Play } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="app-container static-page">
      <header className="header">
        <h1>About ChordLift</h1>
        <p>How it works</p>
      </header>

      <article className="glass-panel static-page-content">
        <section>
          <h2>What is ChordLift?</h2>
          <p>
            ChordLift helps guitarists learn songs faster by analyzing audio and showing chords,
            key, and playback-synced chord diagrams on an interactive timeline. Upload a track,
            wait for analysis, then play along as chords highlight in real time — no manual
            transcription required.
          </p>
        </section>

        <section>
          <h2>How it works</h2>
          <ol className="how-it-works-steps">
            <li>
              <span className="step-icon" aria-hidden="true"><Upload size={22} /></span>
              <div>
                <strong>Upload an audio file</strong>
                <p>Choose MP3, WAV, M4A, FLAC, or OGG from your device.</p>
              </div>
            </li>
            <li>
              <span className="step-icon" aria-hidden="true"><Sparkles size={22} /></span>
              <div>
                <strong>ML analysis</strong>
                <p>
                  Our machine-learning chord engine detects the song key and chord progression
                  from your audio.
                </p>
              </div>
            </li>
            <li>
              <span className="step-icon" aria-hidden="true"><Play size={22} /></span>
              <div>
                <strong>Play with synced diagrams</strong>
                <p>
                  Press play and follow along — chord names and fretboard diagrams update in sync
                  with the music.
                </p>
              </div>
            </li>
          </ol>
        </section>

        <section>
          <h2>FAQ</h2>
          <dl className="faq-list">
            <div>
              <dt>Is my audio stored?</dt>
              <dd>
                No. Your file is uploaded for transient processing only. We analyze it to produce
                chord results and do not keep a permanent copy after processing completes.
              </dd>
            </div>
            <div>
              <dt>What file formats are supported?</dt>
              <dd>MP3, WAV, M4A, FLAC, and OGG.</dd>
            </div>
            <div>
              <dt>How accurate is chord detection?</dt>
              <dd>
                Accuracy depends on recording quality, arrangement, and genre. ChordLift works best
                on clear guitar or full-band pop/rock mixes. You can correct individual chords
                locally if the model misses one.
              </dd>
            </div>
            <div>
              <dt>Is ChordLift free?</dt>
              <dd>
                Yes — the core analyze-and-play experience is free to use. We may show ads in the
                future to support hosting costs.
              </dd>
            </div>
          </dl>
        </section>

        <p className="static-page-back">
          <Link to="/">← Back to ChordLift</Link>
        </p>
      </article>
    </div>
  );
}
