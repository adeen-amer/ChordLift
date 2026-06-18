import { useEffect, useRef, type RefObject } from 'react';
import type { AnalysisData, KeyInfo } from '../types';
import { boundsToSvgRect } from '../utils/fretboardLayout';
import { lyricLineState } from '../utils/lyricsSync';
import { activeLyricIndexBinary, activeRangeIndex, activeSegmentIndex } from '../utils/timeline';
import { segmentTimeKey } from '../utils/chordCorrections';
import {
  getActiveBoxForChord,
  getKeyCagedBoxes,
  type CagedBox,
} from '../utils/pentatonic';

interface SyncEngineProps {
  audioRef: RefObject<HTMLAudioElement | null>;
  data: AnalysisData | null;
  chordContainerRef: RefObject<HTMLDivElement | null>;
  lyricsPanelRef?: RefObject<HTMLDivElement | null>;
  fretboardRef: RefObject<SVGSVGElement | null>;
  improvEnabled: boolean;
}

function updatePentatonicGuide(
  svg: SVGSVGElement,
  activeChordName: string | null,
  songKey: KeyInfo | undefined,
) {
  const boxes = getKeyCagedBoxes(songKey);
  const activeBox = activeChordName ? getActiveBoxForChord(activeChordName, boxes) : null;
  const activeBoxNum = activeBox?.number ?? null;

  const layer = svg.querySelector('.pentatonic-layer');
  layer?.classList.toggle('has-active-chord', Boolean(activeBoxNum));

  boxes.forEach((box) => {
    const group = svg.getElementById(`caged-box-${box.number}`);
    group?.classList.toggle('active-chord-box', activeBoxNum === box.number);
  });

  svg.querySelectorAll('.pent-dot-group').forEach((el) => {
    const group = el as SVGGElement;
    const boxList = group.getAttribute('data-boxes')?.split(',').map(Number) ?? [];
    const inActiveBox = activeBoxNum !== null && boxList.includes(activeBoxNum);

    group.classList.toggle('in-chord-box', inActiveBox);
  });

  const highlight = svg.getElementById('pent-box-highlight') as SVGRectElement | null;
  const chordLabel = svg.getElementById('pent-chord-label') as SVGTextElement | null;

  if (!highlight || !chordLabel) return;

  if (!activeBox || !activeChordName) {
    highlight.setAttribute('opacity', '0');
    chordLabel.setAttribute('opacity', '0');
    chordLabel.textContent = '';
    return;
  }

  applyBoxHighlight(highlight, chordLabel, activeBox, activeChordName);
}

function applyBoxHighlight(
  highlight: SVGRectElement,
  chordLabel: SVGTextElement,
  box: CagedBox,
  chordName: string,
) {
  const rect = boundsToSvgRect(box.bounds, 12);

  highlight.setAttribute('x', String(rect.x));
  highlight.setAttribute('y', String(rect.y));
  highlight.setAttribute('width', String(rect.width));
  highlight.setAttribute('height', String(rect.height));
  highlight.setAttribute('opacity', '1');

  chordLabel.textContent = chordName;
  chordLabel.setAttribute('x', String(rect.cx));
  chordLabel.setAttribute('y', String(rect.cy));
  chordLabel.setAttribute('opacity', '1');
}

function scrollLyricLineIntoView(
  scrollContainer: HTMLElement,
  activeEl: HTMLElement,
) {
  const containerRect = scrollContainer.getBoundingClientRect();
  const elRect = activeEl.getBoundingClientRect();
  const margin = 48;

  if (elRect.top < containerRect.top + margin) {
    scrollContainer.scrollTop -= containerRect.top + margin - elRect.top;
  } else if (elRect.bottom > containerRect.bottom - margin) {
    scrollContainer.scrollTop += elRect.bottom - (containerRect.bottom - margin);
  }
}

function updateLyricsPanel(
  scrollContainer: HTMLDivElement,
  lines: { time: number; text: string }[],
  currentTime: number,
  scrollToActive: boolean,
) {
  const activeIdx = activeLyricIndexBinary(lines, currentTime);
  const lineEls = scrollContainer.querySelectorAll<HTMLElement>('.lyrics-line');

  lineEls.forEach((el, i) => {
    const state = lyricLineState(i, activeIdx);
    const nextState = el.dataset.state;
    if (nextState !== state) {
      el.dataset.state = state;
      el.classList.remove('past', 'active', 'upcoming');
      el.classList.add(state);
    }
  });

  if (scrollToActive && activeIdx >= 0) {
    const activeEl = scrollContainer.querySelector<HTMLElement>(`.lyrics-line[data-index="${activeIdx}"]`);
    if (activeEl) scrollLyricLineIntoView(scrollContainer, activeEl);
  }
}

export function useSyncEngine({
  audioRef,
  data,
  chordContainerRef,
  lyricsPanelRef,
  fretboardRef,
  improvEnabled,
}: SyncEngineProps) {
  const requestRef = useRef<number>(0);
  const lastLyricScrollIdx = useRef<number>(-1);
  const lastChordIdx = useRef<number>(-1);
  const lastSoloIdx = useRef<number>(-1);
  const playingRef = useRef(false);

  useEffect(() => {
    if (!data) return;

    const syncFrame = () => {
      const audio = audioRef.current;
      if (!audio) return;

      const currentTime = audio.currentTime;
      let activeChordName: string | null = null;

      if (chordContainerRef.current) {
        const chordIdx = activeSegmentIndex(data.timeline, currentTime);
        const container = chordContainerRef.current;

        if (chordIdx >= 0) {
          activeChordName = data.timeline[chordIdx].chord;
          const segKey = segmentTimeKey(data.timeline[chordIdx].time);
          const el = container.querySelector<HTMLElement>(`[data-time="${segKey}"]`);
          if (el) {
            container.querySelectorAll('.chord-card.active').forEach((c) => {
              c.classList.remove('active');
            });
            if (!el.classList.contains('active')) {
              el.classList.add('active');
              const containerRect = container.getBoundingClientRect();
              const elRect = el.getBoundingClientRect();
              if (elRect.right > containerRect.right || elRect.left < containerRect.left) {
                container.scrollTo({
                  left: el.offsetLeft - container.clientWidth / 2 + el.clientWidth / 2,
                  behavior: 'smooth',
                });
              }
            }
          }
        } else {
          container.querySelectorAll('.chord-card.active').forEach((c) => {
            c.classList.remove('active');
          });
        }
        lastChordIdx.current = chordIdx;
      }

      const lyricLines = data.lyrics?.lines ?? [];
      if (lyricsPanelRef?.current && lyricLines.length) {
        const activeIdx = activeLyricIndexBinary(lyricLines, currentTime);
        const scrollToActive = activeIdx !== lastLyricScrollIdx.current;
        updateLyricsPanel(lyricsPanelRef.current, lyricLines, currentTime, scrollToActive);
        if (scrollToActive) lastLyricScrollIdx.current = activeIdx;
      }

      if (fretboardRef.current) {
        const wrapper = fretboardRef.current.parentElement;
        const svg = fretboardRef.current;
        const hasSolos = data.solos.length > 0;

        if (improvEnabled && data.key) {
          updatePentatonicGuide(svg, activeChordName, data.key);
          if (wrapper) wrapper.classList.remove('inactive');
        } else if (improvEnabled) {
          updatePentatonicGuide(svg, null, data.key);
        }

        if (hasSolos) {
          const soloIdx = activeRangeIndex(data.solos, currentTime);

          if (soloIdx !== lastSoloIdx.current) {
            const dots = svg.querySelectorAll('.note-dot');
            const texts = svg.querySelectorAll('.note-text');
            dots.forEach((dot) => {
              (dot as SVGCircleElement).setAttribute('opacity', '0');
              (dot as SVGCircleElement).setAttribute('r', '0');
            });
            texts.forEach((t) => (t as SVGTextElement).setAttribute('opacity', '0'));
            lastSoloIdx.current = soloIdx;
          }

          if (soloIdx >= 0) {
            const currentSolo = data.solos[soloIdx];
            if (wrapper) wrapper.classList.remove('inactive');

            const lookahead = 0.3;

            currentSolo.notes.forEach((note, noteIdx) => {
              const dot = svg.getElementById(`dot-${soloIdx}-${noteIdx}`);
              const text = svg.getElementById(`text-${soloIdx}-${noteIdx}`);

              if (currentTime >= note.time && currentTime <= note.end) {
                if (dot) {
                  dot.setAttribute('opacity', '1');
                  dot.setAttribute('r', '8');
                  dot.setAttribute('fill', 'var(--accent-primary)');
                }
                if (text) text.setAttribute('opacity', '1');
              } else if (note.time > currentTime && note.time <= currentTime + lookahead) {
                if (dot) {
                  const distance = note.time - currentTime;
                  const ratio = 1 - distance / lookahead;
                  dot.setAttribute('opacity', (0.2 + ratio * 0.4).toString());
                  dot.setAttribute('r', (4 + ratio * 2).toString());
                  dot.setAttribute('fill', 'var(--accent-secondary)');
                }
              } else if (dot) {
                dot.setAttribute('opacity', '0');
                dot.setAttribute('r', '0');
              }
            });
          } else if (wrapper && !improvEnabled) {
            wrapper.classList.add('inactive');
          }
        } else if (wrapper && !improvEnabled) {
          wrapper.classList.add('inactive');
        }
      }
    };

    const scheduleLoop = () => {
      cancelAnimationFrame(requestRef.current);
      const tick = () => {
        syncFrame();
        if (playingRef.current) {
          requestRef.current = requestAnimationFrame(tick);
        }
      };
      requestRef.current = requestAnimationFrame(tick);
    };

    const stopLoop = () => {
      playingRef.current = false;
      cancelAnimationFrame(requestRef.current);
      syncFrame();
    };

    const audio = audioRef.current;
    if (!audio) return;

    const onPlay = () => {
      playingRef.current = true;
      scheduleLoop();
    };
    const onPause = () => stopLoop();
    const onEnded = () => stopLoop();
    const onSeeked = () => syncFrame();

    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('seeked', onSeeked);

    if (!audio.paused) {
      playingRef.current = true;
      scheduleLoop();
    } else {
      syncFrame();
    }

    return () => {
      playingRef.current = false;
      cancelAnimationFrame(requestRef.current);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('seeked', onSeeked);
      lastLyricScrollIdx.current = -1;
      lastChordIdx.current = -1;
      lastSoloIdx.current = -1;
    };
  }, [data, audioRef, chordContainerRef, lyricsPanelRef, fretboardRef, improvEnabled]);
}
