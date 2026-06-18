"""Beat and downbeat grid for bar-aligned chord decoding."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

BEATS_PER_BAR = 4


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
