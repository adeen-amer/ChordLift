#!/usr/bin/env python3
"""backend/chord_training/fetch_val_data.py

Phase B val-data acquisition: McGill Billboard chord annotations -> Harte
.lab files -> ~40 clean, Isophonics-uncontaminated tracks -> a download plan
consumable by the same audio-fetch mechanism as fetch_train_data.py.

Two stages, run in order:

  --stage annotations   download the McGill Billboard annotations archive
                        (trying known public Dropbox locations, falling back
                        to scraping the project page), extract to
                        data/billboard_raw/, and save an id->(artist,title)
                        index if a usable index CSV was found.

  --stage labs          convert every non-Beatles/Queen candidate to chord
                        segments, apply quality gates, deterministically pick
                        40 (sort by slug, take the first 40), and write
                        data/val/<slug>.lab, data/val_tracks.json and
                        data/val_download_plan.tsv (same TSV columns as
                        fetch_train_data.py's download_plan.tsv).

The two declared Dropbox links for this dataset have drifted over time --
in practice the "lab" URL currently serves the salami_chords archive and the
"salami_chords" URL currently serves the index CSV. Rather than hardcode
that, every fetched URL is content-sniffed (valid tar.gz? parseable CSV?)
and used by what it actually is, not what its name claims.

Reuses fetch_train_data.py's _slugify (same slug style as the train set) and
billboard_to_lab.py's parse_salami (same McGill parser used for train-style
conversions) via plain sibling import -- both modules live in this same
directory, which Python already puts on sys.path when this script is run
directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import billboard_to_lab
import fetch_train_data as ftd

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "billboard_raw"
VAL_DIR = DATA_DIR / "val"
VAL_TRACKS_JSON = DATA_DIR / "val_tracks.json"
VAL_DOWNLOAD_PLAN = DATA_DIR / "val_download_plan.tsv"
INDEX_JSON = RAW_DIR / "index.json"

GOLD_HOLDOUT = BACKEND / "analysis" / "gold_holdout_v2.json"

# Known public locations for the McGill Billboard 2.0 dataset (see
# https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/).
ARCHIVE_URLS = [
    "https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0.1-lab.tar.gz?dl=1",
    "https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-salami_chords.tar.gz?dl=1",
]
INDEX_URL = "https://www.dropbox.com/s/p1u3i6bl52yc1gi/billboard-2.0-index.csv?dl=1"
PROJECT_PAGE = "https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/"

MIN_SEGMENTS = 10
MIN_DURATION_SEC = 120.0
MAX_DURATION_SEC = 420.0
MIN_CHORD_COVERAGE = 0.6
VAL_TARGET = 40
EXCLUDED_ARTIST_NORMS = {"beatles", "thebeatles", "queen"}

_HEADER_RE = re.compile(r"^#\s*(title|artist)\s*:\s*(.*)$", re.IGNORECASE)
_UA = {"User-Agent": "Mozilla/5.0"}


def _download(url: str, cache_name: str) -> tuple[Path | None, str | None]:
    """Download (or reuse cached) URL. Same tempdir-cache pattern as
    fetch_train_data.py's _load_archive. Returns (path, None) on success or
    (None, error) on failure -- never raises, so callers can try the next
    candidate URL."""
    tmp = Path(tempfile.gettempdir()) / cache_name
    if tmp.is_file() and tmp.stat().st_size > 0:
        return tmp, None
    print(f"Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        return tmp, None
    except Exception as exc:  # noqa: BLE001 - report and let caller move on
        tmp.unlink(missing_ok=True)
        return None, str(exc)


def _sniff_kind(path: Path) -> str:
    """'tar' if a valid gzip tarball, 'csv' if parseable comma-delimited text
    that isn't an HTML error/login page, else 'unknown'. Content is sniffed
    rather than inferred from the URL/filename because these Dropbox links
    have drifted from what their names claim (see module docstring)."""
    try:
        with tarfile.open(path, "r:gz") as tf:
            tf.getnames()
        return "tar"
    except Exception:
        pass
    try:
        head = path.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return "unknown"
    first_line = head.splitlines()[0] if head.splitlines() else ""
    if "<html" in head[:200].lower() or "<!doctype" in head[:200].lower():
        return "unknown"
    if first_line.count(",") >= 2:
        return "csv"
    return "unknown"


def _fallback_project_page_archive() -> Path | None:
    """Scrape the McGill project page for a tar.gz href if both declared
    Dropbox archive URLs failed."""
    path, err = _download(PROJECT_PAGE, "billboard_project_page.html")
    if path is None:
        print(f"  project page fetch failed: {err}", file=sys.stderr)
        return None
    html = path.read_text(encoding="utf-8", errors="ignore")
    hrefs = re.findall(r'href="([^"]+\.tar\.gz)"', html)
    for href in hrefs:
        url = href if href.startswith("http") else PROJECT_PAGE.rstrip("/") + "/" + href.lstrip("/")
        cand, err = _download(url, "billboard_archive_pagefallback.bin")
        if cand and _sniff_kind(cand) == "tar":
            return cand
        if err:
            print(f"  {url}: {err}", file=sys.stderr)
    return None


def stage_annotations() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetched = []  # (label, url, path|None, kind)
    errors = []
    for i, url in enumerate(ARCHIVE_URLS, 1):
        path, err = _download(url, f"billboard_archive_{i}.bin")
        if path is None:
            errors.append(f"{url}: {err}")
            fetched.append((f"archive_{i}", url, None, "unreachable"))
            continue
        fetched.append((f"archive_{i}", url, path, _sniff_kind(path)))
    index_path, index_err = _download(INDEX_URL, "billboard_index.bin")
    if index_path is None:
        errors.append(f"{INDEX_URL}: {index_err}")
        fetched.append(("index", INDEX_URL, None, "unreachable"))
    else:
        fetched.append(("index", INDEX_URL, index_path, _sniff_kind(index_path)))

    tar_path = next((p for _, _, p, k in fetched if k == "tar"), None)
    tar_source = next((u for _, u, p, k in fetched if k == "tar"), None)
    if tar_path is None:
        print("all declared archive URLs failed content-sniff as tar.gz; trying project page ...")
        tar_path = _fallback_project_page_archive()
        tar_source = PROJECT_PAGE

    if tar_path is None:
        print("BLOCKED: no reachable URL yielded a valid Billboard annotations tar.gz", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            try:
                tf.extractall(RAW_DIR, filter="data")
            except TypeError:  # Python < 3.12 has no `filter` kwarg
                tf.extractall(RAW_DIR)
    except Exception as exc:
        print(f"BLOCKED: failed to extract {tar_path}: {exc}", file=sys.stderr)
        return 1

    # CSV candidate: prefer whichever fetched URL actually sniffed as CSV,
    # regardless of which declared role (index vs archive) it came from --
    # see module docstring on link drift.
    csv_path = next((p for _, _, p, k in fetched if k == "csv"), None)
    index: dict[int, tuple[str, str]] = {}
    if csv_path is not None:
        with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                if row.get("id") and row.get("artist") and row.get("title"):
                    try:
                        index[int(row["id"])] = (row["artist"], row["title"])
                    except ValueError:
                        pass
        INDEX_JSON.write_text(
            json.dumps({str(k): list(v) for k, v in index.items()}, indent=2, sort_keys=True) + "\n"
        )

    entries = sum(1 for _ in RAW_DIR.rglob("salami_chords.txt"))
    print(f"archive used: {tar_source}")
    print(f"entries found: {entries}")
    print(f"index rows: {len(index)}" + ("" if csv_path else " (no usable index CSV -- will fall back to annotation headers)"))
    return 0


def _artist_title_from_header(text: str) -> tuple[str, str]:
    artist = title = ""
    for line in text.splitlines()[:8]:
        m = _HEADER_RE.match(line.strip())
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "title":
            title = val
        else:
            artist = val
    return artist, title


def _is_excluded_artist(artist: str) -> bool:
    norm = re.sub(r"[^a-z0-9]", "", artist.lower())
    return norm in EXCLUDED_ARTIST_NORMS


def stage_labs(pick: int = VAL_TARGET) -> int:
    ann_files = sorted(RAW_DIR.rglob("salami_chords.txt"))
    if not ann_files:
        print("error: no salami_chords.txt under data/billboard_raw/ -- run --stage annotations first", file=sys.stderr)
        return 1

    index: dict[int, tuple[str, str]] = {}
    if INDEX_JSON.is_file():
        index = {int(k): tuple(v) for k, v in json.loads(INDEX_JSON.read_text()).items()}

    gold_ids: set[str] = set()
    if GOLD_HOLDOUT.is_file():
        gold_ids = set(json.loads(GOLD_HOLDOUT.read_text())["track_ids"])

    candidates = 0
    skip_reasons: Counter = Counter()
    survivors = []  # (base_slug, track_id, artist, title, duration, segments)

    for ann_file in ann_files:
        candidates += 1
        text = ann_file.read_text(encoding="utf-8", errors="ignore")

        try:
            track_id = int(ann_file.parent.name)
        except ValueError:
            track_id = None

        artist, title = index.get(track_id, ("", "")) if track_id is not None else ("", "")
        if not artist or not title:
            hdr_artist, hdr_title = _artist_title_from_header(text)
            artist, title = artist or hdr_artist, title or hdr_title
        if not artist or not title:
            skip_reasons["missing-metadata"] += 1
            continue

        if _is_excluded_artist(artist):
            skip_reasons["beatles-queen-artist"] += 1
            continue

        try:
            segments = billboard_to_lab.parse_salami(text)
        except Exception:
            skip_reasons["parse-error"] += 1
            continue
        if not segments:
            skip_reasons["no-segments"] += 1
            continue
        if len(segments) < MIN_SEGMENTS:
            skip_reasons["too-few-segments"] += 1
            continue

        duration = max(e for _, e, _ in segments)
        if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
            skip_reasons["duration-out-of-range"] += 1
            continue

        non_n = sum(e - s for s, e, c in segments if c != "N")
        coverage = non_n / duration if duration > 0 else 0.0
        if coverage < MIN_CHORD_COVERAGE:
            skip_reasons["low-chord-coverage"] += 1
            continue

        # Gold-holdout check: the artist filter above already removes every
        # Beatles/Queen track, so this only fires if some other artist's
        # charted title happens to collide with one of the 24 gold titles
        # (e.g. a cover or same-named song) -- exclude those too.
        title_slug = ftd._slugify(title)
        if title_slug in gold_ids:
            skip_reasons["gold-holdout-title-collision"] += 1
            continue

        base_slug = ftd._slugify(f"{artist} {title}")
        survivors.append((base_slug, track_id or 0, artist, title, duration, segments))

    # Deterministic pick: sort by slug (ties broken by track id), then take
    # the first 40. Same base slug can legitimately recur (the Billboard
    # chart data re-lists some songs under a second chart run/id) -- suffix
    # with -2, -3... same collision style as fetch_train_data.py's stage_labs.
    survivors.sort(key=lambda t: (t[0], t[1]))
    used: Counter = Counter()
    final = []
    for base_slug, track_id, artist, title, duration, segments in survivors:
        used[base_slug] += 1
        slug = base_slug if used[base_slug] == 1 else f"{base_slug}-{used[base_slug]}"
        final.append((slug, artist, title, duration, segments))

    picked = final[:pick]

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    val_tracks: dict[str, dict] = {}
    plan_rows = []
    for slug, artist, title, duration, segments in picked:
        lab_text = "".join(f"{s:.6f}\t{e:.6f}\t{c}\n" for s, e, c in segments)
        (VAL_DIR / f"{slug}.lab").write_text(lab_text)
        val_tracks[slug] = {"artist": artist, "title": title, "lab_end_seconds": duration}
        plan_rows.append((slug, artist, title, duration))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VAL_TRACKS_JSON.write_text(json.dumps(val_tracks, indent=2, sort_keys=True) + "\n")
    lines = [f"{s}\t{a}\t{t}\t{e}" for s, a, t, e in plan_rows]
    VAL_DOWNLOAD_PLAN.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"candidates: {candidates}")
    print(f"survivors: {len(final)}")
    print(f"picked: {len(picked)}")
    print("skip reasons: " + (", ".join(f"{k}={v}" for k, v in skip_reasons.most_common()) or "none"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B val-data fetch (annotations / labs)")
    ap.add_argument("--stage", required=True, choices=["annotations", "labs"])
    ap.add_argument("--pick", type=int, default=VAL_TARGET, help="number of val tracks to pick (labs stage only)")
    args = ap.parse_args()

    if args.stage == "annotations":
        return stage_annotations()
    return stage_labs(args.pick)


if __name__ == "__main__":
    raise SystemExit(main())
