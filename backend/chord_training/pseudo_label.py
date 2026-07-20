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

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
DATA_DIR = HERE / "data"
VERIFY_SCRIPT = BACKEND / "scripts" / "verify_training_leakage.py"

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
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            # r.url (and thus the default HTTPError message) carries the API
            # key as a query param -- re-raise sanitized so it never reaches
            # stderr/CI logs via fetch_fma_sample's exception handler.
            raise RuntimeError(f"FMA API HTTP {r.status_code} for track {track_id}") from exc
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


# ---- manifest merging ----------------------------------------------------

def merge_manifests(train_manifest_path: str, pseudo_manifest_path: str) -> list[str]:
    """Append pseudo-labeled (audio, lab) pairs into an existing
    train_manifest.txt, skipping lines already present."""
    with open(train_manifest_path) as f:
        existing = {line.strip() for line in f if line.strip()}
    with open(pseudo_manifest_path) as f:
        candidates = [line.strip() for line in f if line.strip()]
    to_add = [line for line in candidates if line not in existing]
    with open(train_manifest_path, "a") as f:
        for line in to_add:
            f.write(line + "\n")
    return to_add


# ---- CLI ------------------------------------------------------------------

def main() -> int:
    from build_manifest import AUDIO_EXTS

    ap = argparse.ArgumentParser(description="Pseudo-label FMA audio for chordia fine-tuning")
    ap.add_argument("--stage", required=True, choices=["fetch", "label", "manifest"])
    ap.add_argument("--pool-csv", help="fetch stage: flat track_id,duration_sec CSV (Task 0 Step 4)")
    ap.add_argument("--api-key", default=os.environ.get("FMA_KEY", ""), help="fetch stage: FMA API key")
    ap.add_argument("--data-dir", default=str(DATA_DIR / "pseudo"),
                     help="fetch: download destination; label: audio source + .lab output dir")
    ap.add_argument("--n", type=int, default=200, help="fetch stage: sample size")
    ap.add_argument("--min-duration", type=float, default=60.0,
                     help="fetch stage: minimum track duration in seconds")
    ap.add_argument("--seed", type=int, default=0, help="fetch stage: sampling seed")
    ap.add_argument("--confidence-threshold", type=float, default=0.7,
                     help="label stage: minimum segment confidence to keep")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                     help="label stage: minimum retained-duration fraction to keep a track")
    ap.add_argument("--pseudo-manifest", default=str(DATA_DIR / "pseudo_manifest.txt"),
                     help="label stage output / manifest stage input")
    ap.add_argument("--train-manifest", help="manifest stage: train_manifest.txt to append into")
    args = ap.parse_args()

    if args.stage == "fetch":
        if not args.pool_csv:
            print("--pool-csv is required for the fetch stage", file=sys.stderr)
            return 1
        if not args.api_key:
            print("FMA_KEY env var or --api-key is required for the fetch stage", file=sys.stderr)
            return 1
        fetched = fetch_fma_sample(args.pool_csv, args.api_key, args.data_dir,
                                    args.n, args.min_duration, args.seed)
        print(f"fetched: {len(fetched)}/{args.n}")
        return 0

    if args.stage == "label":
        audio_files = sorted(
            p for ext in AUDIO_EXTS for p in Path(args.data_dir).glob(f"*{ext}")
        )
        kept = 0
        with open(args.pseudo_manifest, "w") as manifest:
            for audio_path in audio_files:
                lab_path = audio_path.with_suffix(".lab")
                try:
                    coverage = label_track(str(audio_path), str(lab_path),
                                            args.confidence_threshold, args.min_coverage)
                except Exception as exc:
                    print(f"skipping {audio_path.name}: {exc}", file=sys.stderr)
                    continue
                if coverage is None:
                    print(f"skipping {audio_path.name}: retained coverage below "
                          f"{args.min_coverage}", file=sys.stderr)
                    continue
                manifest.write(f"{audio_path}\t{lab_path}\n")
                kept += 1
        print(f"labeled: {kept}/{len(audio_files)}")
        return 0

    # manifest stage
    if not args.train_manifest:
        print("--train-manifest is required for the manifest stage", file=sys.stderr)
        return 1
    # Gate on the candidate pseudo pairs BEFORE touching train_manifest: once
    # merge_manifests appends into train_manifest there's no clean rollback
    # (a re-run just skips already-present lines), so leaked rows must never
    # land there in the first place.
    leak_rc = subprocess.run([sys.executable, str(VERIFY_SCRIPT), args.pseudo_manifest]).returncode
    if leak_rc != 0:
        return 1
    added = merge_manifests(args.train_manifest, args.pseudo_manifest)
    print(f"merged {len(added)} new pairs into {args.train_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
