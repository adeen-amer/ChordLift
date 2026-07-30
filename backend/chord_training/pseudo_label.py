#!/usr/bin/env python3
"""backend/chord_training/pseudo_label.py

Pseudo-labels Free Music Archive audio using the production chordia
ensemble, so the 161-track hand-labeled fine-tune corpus
(BASELINE_v51_README.md) can be supplemented without hand-annotating more
Harte labels. Staged like fetch_train_data.py:

  --stage fetch     sample items from the Internet Archive's
                     `freemusicarchive` collection and download full-length
                     audio. (FMA's own metadata API is retired -- every
                     documented endpoint, including /api/agreement, 404s as
                     of 2026-07-30, so there is no API key to obtain.)
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

import argparse
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

IA_BASE = "https://archive.org"
IA_COLLECTION = "freemusicarchive"
# AUDIO_EXTS in build_manifest.py is what --stage label globs for, so only
# download extensions that stage can actually pick up.
IA_AUDIO_EXTS = (".mp3", ".wav")


# ---- Internet Archive FMA collection ------------------------------------

def _ia_download_url(identifier: str, filename: str) -> str:
    return f"{IA_BASE}/download/{identifier}/{filename}"


def _parse_ia_length(raw: str | None) -> float | None:
    """IA's `length` field is seconds ("1728.35") on originals but mm:ss or
    hh:mm:ss on some derivatives. Returns None for absent/unparseable."""
    if not raw:
        return None
    parts = str(raw).split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:  # sexagesimal, most-significant first
        seconds = seconds * 60 + value
    return seconds


def select_audio_file(
    files: list[dict], min_duration_sec: float, max_duration_sec: float = float("inf"),
) -> dict | None:
    """Pick the longest usable audio file from an IA item's file list.

    Adds a `duration_sec` key to the returned dict (parsed from `length`) so
    the caller doesn't re-parse. Prefers `source == "original"`: IA items
    typically carry a low-bitrate derivative alongside the original upload,
    and the derivative is the worse training input at identical duration.

    max_duration_sec is not cosmetic: a random sample of this collection
    surfaces hour-long drones and live WFMU radio shows (talk + music), which
    the teacher happily labels with 30s-long single-chord segments -- high
    confidence, useless as song-like harmonic training data.
    """
    candidates = []
    for f in files:
        if not f.get("name", "").lower().endswith(IA_AUDIO_EXTS):
            continue
        duration = _parse_ia_length(f.get("length"))
        if duration is None or not (min_duration_sec <= duration <= max_duration_sec):
            continue
        candidates.append({**f, "duration_sec": duration})
    if not candidates:
        return None
    return max(candidates, key=lambda f: (f.get("source") == "original", f["duration_sec"]))


class InternetArchive:
    """Minimal read-only archive.org client -- only the three calls this
    pipeline needs. No API key: the scrape/metadata/download endpoints are
    public (FMA's own API is retired, see module docstring)."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def scrape_identifiers(self, collection: str = IA_COLLECTION) -> list[str]:
        """Every item id in a collection, via the cursor-paged scrape API."""
        identifiers: list[str] = []
        cursor = None
        while True:
            params = {"q": f"collection:{collection}", "count": 10000}
            if cursor:
                params["cursor"] = cursor
            r = self.session.get(
                f"{IA_BASE}/services/search/v1/scrape", params=params, timeout=120,
            )
            r.raise_for_status()
            payload = r.json()
            identifiers.extend(item["identifier"] for item in payload.get("items", []))
            cursor = payload.get("cursor")
            if not cursor:
                return identifiers

    def get_files(self, identifier: str) -> list[dict]:
        r = self.session.get(f"{IA_BASE}/metadata/{identifier}", timeout=60)
        r.raise_for_status()
        return r.json().get("files", [])

    def download(self, identifier: str, filename: str, dest_path: str) -> None:
        r = self.session.get(
            _ia_download_url(identifier, filename), stream=True, timeout=300,
        )
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def build_identifier_pool(pool_path: str, force: bool = False) -> list[str]:
    """Cache the collection's item ids to a newline-delimited file.

    Replaces the plan's 358MB fma_metadata.zip step: the pool is ~16.8k ids,
    a few hundred KB, and re-derivable in two requests.
    """
    path = Path(pool_path)
    if path.exists() and not force:
        return [line for line in path.read_text().splitlines() if line.strip()]
    identifiers = InternetArchive().scrape_identifiers()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(identifiers) + "\n")
    return identifiers


def select_random_identifiers(pool_path: str, n: int, seed: int) -> list[str]:
    """Deterministic (given seed) random sample of item ids from the pool
    file. Duration filtering happens later, per item, in fetch_fma_sample --
    the pool carries ids only."""
    candidates = [line for line in Path(pool_path).read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def fetch_fma_sample(
    pool_path: str, dest_dir: str, n: int, min_duration_sec: float, seed: int,
    max_duration_sec: float = float("inf"),
) -> list[str]:
    """Download a random full-length FMA sample from archive.org.

    Oversamples the pool by 3x because a given item may have no audio file
    long enough (min_duration_sec) or fail to download; items that fail are
    skipped and reported to stderr, same convention dataset.py:build_storages
    uses for unencodable pairs. Re-runnable: already-downloaded files count
    toward n without re-fetching.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    ia = InternetArchive()
    fetched: list[str] = []
    for identifier in select_random_identifiers(pool_path, n * 3, seed):
        if len(fetched) >= n:
            break
        try:
            picked = select_audio_file(
                ia.get_files(identifier), min_duration_sec, max_duration_sec,
            )
            if picked is None:
                print(f"skipping {identifier}: no audio in "
                      f"[{min_duration_sec}s, {max_duration_sec}s]", file=sys.stderr)
                continue
            ext = os.path.splitext(picked["name"])[1].lower()
            dest = os.path.join(dest_dir, f"fma-{identifier}{ext}")
            if not os.path.exists(dest):
                ia.download(identifier, picked["name"], dest)
            fetched.append(dest)
        except Exception as exc:
            print(f"skipping {identifier}: {exc}", file=sys.stderr)
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
    ap.add_argument("--pool", default=str(DATA_DIR / "fma_ia_pool.txt"),
                    help="fetch stage: cached archive.org id pool (built on first run)")
    ap.add_argument("--refresh-pool", action="store_true",
                    help="fetch stage: re-scrape the id pool instead of reusing the cache")
    ap.add_argument("--data-dir", default=str(DATA_DIR / "pseudo"),
                    help="fetch: download destination; label: audio source + .lab output dir")
    ap.add_argument("--n", type=int, default=200, help="fetch stage: sample size")
    ap.add_argument("--min-duration", type=float, default=60.0,
                    help="fetch stage: minimum track duration in seconds")
    ap.add_argument("--max-duration", type=float, default=600.0,
                    help="fetch stage: maximum track duration in seconds. Caps the "
                         "hour-long drones and live radio shows this collection is "
                         "full of; raise it if you want long-form material.")
    ap.add_argument("--seed", type=int, default=0, help="fetch stage: sampling seed")
    ap.add_argument("--confidence-threshold", type=float, default=0.6,
                    help="label stage: minimum segment confidence to keep. 0.6 measured, "
                         "not guessed: on real FMA tracks the teacher's mean triad posterior "
                         "sits near 0.64, so the spec's suggested 0.7 left both smoke-tested "
                         "tracks under --min-coverage and discarded them outright. Re-tune "
                         "from the coverage stats this stage prints.")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="label stage: minimum retained-duration fraction to keep a track")
    ap.add_argument("--pseudo-manifest", default=str(DATA_DIR / "pseudo_manifest.txt"),
                    help="label stage output / manifest stage input")
    ap.add_argument("--train-manifest", help="manifest stage: train_manifest.txt to append into")
    args = ap.parse_args()

    if args.stage == "fetch":
        pool = build_identifier_pool(args.pool, force=args.refresh_pool)
        print(f"pool: {len(pool)} archive.org ids ({args.pool})")
        fetched = fetch_fma_sample(args.pool, args.data_dir, args.n,
                                   args.min_duration, args.seed, args.max_duration)
        print(f"fetched: {len(fetched)}/{args.n}")
        return 0

    if args.stage == "label":
        audio_files = sorted(
            p for ext in AUDIO_EXTS for p in Path(args.data_dir).glob(f"*{ext}")
        )
        coverages = []
        with open(args.pseudo_manifest, "w") as manifest:
            for audio_path in audio_files:
                lab_path = audio_path.with_suffix(".lab")
                coverage = label_track(str(audio_path), str(lab_path),
                                       args.confidence_threshold, args.min_coverage)
                if coverage is None:
                    print(f"skipping {audio_path.name}: retained coverage below "
                          f"{args.min_coverage}", file=sys.stderr)
                    continue
                manifest.write(f"{audio_path}\t{lab_path}\n")
                coverages.append(coverage)
        mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        # Printed so the threshold can be tuned from one run: too many skips
        # means it is too tight, coverage near 1.0 means it is too loose.
        print(f"labeled: {len(coverages)}/{len(audio_files)} "
              f"(threshold={args.confidence_threshold}, mean kept coverage={mean_coverage:.3f})")
        return 0

    # manifest stage
    if not args.train_manifest:
        print("--train-manifest is required for the manifest stage", file=sys.stderr)
        return 1
    added = merge_manifests(args.train_manifest, args.pseudo_manifest)
    print(f"merged {len(added)} new pairs into {args.train_manifest}")
    leak_rc = subprocess.run([sys.executable, str(VERIFY_SCRIPT), args.train_manifest]).returncode
    return 1 if leak_rc != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
