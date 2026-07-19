"""Beat and downbeat grid for bar-aligned chord decoding."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import librosa
import numpy as np

logger = logging.getLogger(__name__)

BEATS_PER_BAR = 4
BEAT_ENGINE = os.getenv("CHORD_BEAT_ENGINE", "auto").lower().strip()


@dataclass(frozen=True)
class BeatGrid:
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    tempo_bpm: float
    beats_per_bar: int = BEATS_PER_BAR

    @property
    def beat_duration(self) -> float:
        if len(self.beat_times) > 1:
            return float(np.median(np.diff(self.beat_times)))
        return 60.0 / max(self.tempo_bpm, 1.0)


def track_beats(
    y_harmonic: np.ndarray,
    y_bass: np.ndarray,
    sr: int,
    hop_length: int = 512,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BeatGrid:
    """
    Estimate beat and downbeat times from harmonic + bass stems.

    Downbeats use bass onset strength on the beat grid (4/4 assumption).
    """
    y_pulse = (0.55 * y_harmonic + 0.45 * y_bass).astype(np.float32)
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_pulse, sr=sr, hop_length=hop_length, units="frames",
        )
    except TypeError:
        tempo = librosa.beat.tempo(y=y_pulse, sr=sr, hop_length=hop_length)[0]
        _, beat_frames = librosa.beat.beat_track(
            y=y_pulse, sr=sr, hop_length=hop_length, units="frames",
        )

    beat_frames = librosa.util.fix_frames(
        beat_frames, x_min=0, x_max=max(1, len(y_pulse) // hop_length) - 1,
    )
    if len(beat_frames) < 4:
        n_frames = max(1, len(y_pulse) // hop_length)
        beat_frames = np.arange(0, n_frames, max(1, n_frames // 64))

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    bass_onset = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop_length)
    fixed = librosa.util.fix_frames(beat_frames, x_min=0, x_max=len(bass_onset) - 1)
    strengths = bass_onset[fixed] if len(fixed) else np.array([0.0])
    anchor = int(np.argmax(strengths[: min(32, len(strengths))]))
    downbeat_mask = np.array(
        [(i - anchor) % beats_per_bar == 0 for i in range(len(beat_times))],
        dtype=bool,
    )
    downbeat_times = beat_times[downbeat_mask]
    if len(downbeat_times) == 0:
        downbeat_times = beat_times[::beats_per_bar]

    return BeatGrid(
        beat_times=np.asarray(beat_times, dtype=np.float64),
        downbeat_times=np.asarray(downbeat_times, dtype=np.float64),
        tempo_bpm=float(tempo),
        beats_per_bar=beats_per_bar,
    )


def _track_beats_madmom(y_full: np.ndarray, sr: int, beats_per_bar: int = BEATS_PER_BAR) -> BeatGrid:
    """Model-based beat/downbeat tracking on the full mix (no stems needed)."""
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    signal = Signal(y_full.astype(np.float32), sample_rate=sr, num_channels=1)
    activations = RNNDownBeatProcessor()(signal)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar], fps=100)
    beat_positions = proc(activations)  # rows: [time_sec, beat_number]

    if len(beat_positions) == 0:
        raise RuntimeError("madmom returned no beats")

    beat_times = np.asarray(beat_positions[:, 0], dtype=np.float64)
    downbeat_mask = beat_positions[:, 1].astype(int) == 1
    downbeat_times = beat_times[downbeat_mask]
    if len(downbeat_times) == 0:
        downbeat_times = beat_times[::beats_per_bar]

    tempo_bpm = 60.0 / float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 120.0

    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=np.asarray(downbeat_times, dtype=np.float64),
        tempo_bpm=tempo_bpm,
        beats_per_bar=beats_per_bar,
    )


def track_beats_auto(
    stems,
    sr: int,
    hop_length: int = 512,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BeatGrid:
    """
    Dispatch to the configured beat/downbeat engine (CHORD_BEAT_ENGINE).

    "auto" (default) tries madmom, falling back to the librosa heuristic on
    any failure. "madmom" forces madmom (errors surface). "librosa" keeps
    today's heuristic. Task 4 adds a "transformer" engine and upgrades
    "auto" to try it first when real Demucs stems are available.
    """
    engine = BEAT_ENGINE

    if engine == "madmom":
        return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)

    if engine == "auto":
        try:
            return _track_beats_madmom(stems.full, sr, beats_per_bar=beats_per_bar)
        except Exception:
            logger.warning("madmom beat tracking failed, falling back to librosa heuristic", exc_info=True)

    return track_beats(
        stems.chord_signal, stems.bass, sr, hop_length=hop_length, beats_per_bar=beats_per_bar,
    )
