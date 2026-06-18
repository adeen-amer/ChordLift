"""Chord AI-style audio prep: stems → beat grid → shared analysis context."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beat_tracking import BeatGrid, track_beats
from stem_separation import StemBundle, separate_stems


@dataclass
class ChordPipelineContext:
    """Shared stem + beat context for ML and classic chord engines."""

    stems: StemBundle
    beats: BeatGrid
    sr: int
    hop_length: int = 512

    @property
    def y_chord(self) -> np.ndarray:
        return self.stems.chord_signal

    @property
    def y_bass(self) -> np.ndarray:
        return self.stems.bass

    @property
    def y_full(self) -> np.ndarray:
        return self.stems.full

    @property
    def beat_times(self) -> np.ndarray:
        return self.beats.beat_times

    @property
    def downbeat_times(self) -> np.ndarray:
        return self.beats.downbeat_times

    @property
    def beat_duration(self) -> float:
        return self.beats.beat_duration

    @property
    def tempo_bpm(self) -> float:
        return self.beats.tempo_bpm


def finalize_bar_timeline(segments, ctx, chroma, chroma_low, chroma_mid, frame_keys, sr, alternate_segments=None):
    """Bar-quantize and merge segments using downbeat grid + chroma scoring."""
    from analyzer import _key_at_time, _match_score_for_chord, _segment_center_chroma
    from bar_decode import apply_bar_decode

    hop = ctx.hop_length
    beat_times = ctx.beat_times

    def score_fn(start, end, candidates):
        if frame_keys is not None and len(beat_times) > 0:
            key_r, key_m = _key_at_time((start + end) / 2.0, beat_times, frame_keys)
        else:
            key_r, key_m = 0, True
        mc, mb, mm = _segment_center_chroma(
            chroma, chroma_low, chroma_mid, start, end, sr, hop,
        )
        best_chord = candidates[0] if candidates else "N"
        best_score = -1.0
        for chord in candidates:
            if not chord or chord == "N":
                continue
            score = _match_score_for_chord(mc, mb, mm, chord, key_r, key_m)
            if score > best_score:
                best_score = score
                best_chord = chord
        return best_chord, best_score, candidates

    return apply_bar_decode(
        segments, ctx.downbeat_times, ctx.beat_duration, score_fn,
        alternate_segments=alternate_segments,
    )


def build_chord_pipeline_context(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> ChordPipelineContext:
    """Separate stems and track beats/downbeats before chord decoding."""
    stems = separate_stems(y, sr)
    beats = track_beats(stems.chord_signal, stems.bass, sr, hop_length=hop_length)
    return ChordPipelineContext(stems=stems, beats=beats, sr=sr, hop_length=hop_length)

