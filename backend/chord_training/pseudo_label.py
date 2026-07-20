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

Dependencies: main backend venv (backend/requirements.txt + requirements-ml.txt),
not the GPU-only desktop bundle (requirements-train.txt). This script is a
data-acquisition tool like fetch_train_data.py and is not part of pack_bundle.py.
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/

FMA_API_BASE = "https://freemusicarchive.org/api/get/"
FMA_FILES_BASE = "https://files.freemusicarchive.org/"


# ---- FMA client and track sampling ------------------------------------

def _track_download_url(track_file: str) -> str:
    return FMA_FILES_BASE + track_file


class FreeMusicArchive:
    """Minimal client for the FMA metadata API -- only the two calls this
    pipeline needs, adapted from mdeff/fma's utils.py (MIT-licensed)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_track(self, track_id: int) -> dict:
        r = requests.get(
            f"{FMA_API_BASE}tracks.json",
            params={"track_id": track_id, "api_key": self.api_key},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            raise RuntimeError(f"FMA API error for track {track_id}: {payload['errors']}")
        return payload["dataset"][0]

    def download_track(self, track_file: str, dest_path: str) -> None:
        r = requests.get(_track_download_url(track_file), stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def select_random_track_ids(
    pool_csv_path: str, n: int, min_duration_sec: float, seed: int,
) -> list[int]:
    """Deterministic (given seed) random sample of track ids from a flat
    track_id,duration_sec CSV (Task 0 Step 4 derives this from the official
    FMA tracks.csv), filtered to tracks at least min_duration_sec long."""
    candidates = []
    with open(pool_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["duration_sec"]) >= min_duration_sec:
                candidates.append(int(row["track_id"]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def fetch_fma_sample(
    pool_csv_path: str, api_key: str, dest_dir: str, n: int, min_duration_sec: float, seed: int,
) -> list[str]:
    """Download a random FMA sample. Tracks that fail (API error, download
    error) are skipped and reported to stderr, same convention
    dataset.py:build_storages uses for unencodable pairs."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    track_ids = select_random_track_ids(pool_csv_path, n, min_duration_sec, seed)
    client = FreeMusicArchive(api_key)
    fetched = []
    for track_id in track_ids:
        dest = os.path.join(dest_dir, f"fma-{track_id}.mp3")
        if os.path.exists(dest):
            fetched.append(dest)
            continue
        try:
            meta = client.get_track(track_id)
            client.download_track(meta["track_file"], dest)
            fetched.append(dest)
        except Exception as exc:
            print(f"skipping fma track {track_id}: {exc}", file=sys.stderr)
    return fetched


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


# ---- track-level orchestration -----------------------------------------

def label_track(
    audio_path: str, lab_out_path: str, threshold: float, min_coverage: float = 0.5,
) -> float | None:
    """Pseudo-label one track with the production chordia ensemble.

    Returns retained-coverage fraction and writes lab_out_path on success;
    returns None (writes nothing) if confidence-filtering leaves less than
    min_coverage of the track's duration covered -- same skip-and-report
    convention dataset.py:build_storages uses for unencodable pairs.
    """
    import librosa

    sys.path.insert(0, str(BACKEND))
    from chord_engine_chordia import decode_chordia_probs, recognize_chordia_probs
    from lv_chordia.settings import DEFAULT_HOP_LENGTH as HOP, DEFAULT_SR as SR

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    probs, hmm, entry = recognize_chordia_probs(y, sr)
    segments = decode_chordia_probs(probs, hmm, entry)
    frame_conf = frame_confidences(probs)
    kept = filter_low_confidence_segments(segments, frame_conf, sr, HOP, threshold)
    coverage = retained_coverage(kept, duration)
    if coverage < min_coverage:
        return None
    write_lab(lab_out_path, kept)
    return coverage
