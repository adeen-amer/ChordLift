#!/usr/bin/env python3
"""backend/chord_training/pseudo_label.py

Pseudo-labels FMA audio using the production chordia ensemble, so the
161-track hand-labeled fine-tune corpus (BASELINE_v51_README.md) can be
supplemented without hand-annotating more Harte labels. Staged like
fetch_train_data.py:

  --stage fetch     sample track ids from a flat (track_id, duration_sec)
                     pool CSV and download audio via the FMA API.
  --stage label     run recognize_chordia_probs on each fetched track,
                     drop low-confidence segments, write .lab files.
  --stage manifest  merge the pseudo (audio, lab) pairs into an existing
                     train_manifest.txt, gated by verify_training_leakage.py.

dataset.py/finetune.py are unchanged: encode_labels already treats "no
segment covers this frame" as masked (X), so dropped low-confidence
segments are handled by the existing pipeline for free.

Built up incrementally (Tasks 1-5); imports and module-level constants are
added in the task that first needs them, not all upfront, so every
commit's file stays free of unused imports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---- confidence filtering ---------------------------------------------

def frame_confidences(probs: list[np.ndarray]) -> np.ndarray:
    """Per-frame confidence: max triad posterior, same quantity
    chord_engine_chordia.chordia_confidence averages over the whole track."""
    return np.asarray(probs[0]).max(axis=1)


def segment_frame_range(seg: dict, sr: int, hop: int, n_frames: int) -> tuple[int, int]:
    """Frame index range [lo, hi) covered by a decoded segment.

    Mirrors dataset.py:encode_labels's forward mapping (frame f covers
    time f*hop/sr), inverted: t -> f = t*sr/hop.
    """
    lo = int(seg["start_time"] * sr / hop)
    hi = int(seg["end_time"] * sr / hop)
    lo = max(0, min(lo, n_frames))
    hi = max(lo, min(hi, n_frames))
    return lo, hi


def segment_mean_confidence(seg: dict, frame_conf: np.ndarray, sr: int, hop: int) -> float:
    lo, hi = segment_frame_range(seg, sr, hop, len(frame_conf))
    if hi <= lo:
        return 0.0
    return float(frame_conf[lo:hi].mean())


def filter_low_confidence_segments(
    segments: list[dict], frame_conf: np.ndarray, sr: int, hop: int, threshold: float,
) -> list[dict]:
    """Drop any segment whose mean frame confidence is below threshold.

    Dropped segments simply aren't written to the .lab file; the caller
    (label_track) relies on encode_labels's existing no-segment-covers-
    this-frame -> X masking to keep the loss ignoring these frames.
    """
    return [
        s for s in segments
        if segment_mean_confidence(s, frame_conf, sr, hop) >= threshold
    ]


# ---- .lab writing + coverage -------------------------------------------

def write_lab(path: str, segments: list[dict]) -> None:
    lines = [f"{s['start_time']}\t{s['end_time']}\t{s['chord']}\n" for s in segments]
    Path(path).write_text("".join(lines))


def retained_coverage(segments: list[dict], track_duration: float) -> float:
    if track_duration <= 0:
        return 0.0
    covered = sum(s["end_time"] - s["start_time"] for s in segments)
    return covered / track_duration
