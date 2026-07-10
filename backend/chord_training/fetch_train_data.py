#!/usr/bin/env python3
"""backend/chord_training/fetch_train_data.py

Phase B train-data acquisition: assemble (audio, .lab) pairs under
chord_training/data/train/ from Isophonics Beatles+Queen chord annotations,
excluding the 24 gold-holdout tracks (analysis/gold_holdout_v2.json).

Three stages, run in order:

  --stage labs         download the two Isophonics archives, extract every
                        non-gold chordlab/*.lab member to data/train/<slug>.lab,
                        write data/train_tracks.json (slug -> metadata).

  --stage audio-plan    for each slug missing audio, try to reuse audio already
                        on disk (tests/fixtures/{gold_mir_tracks_v2,chord_refs}.json)
                        gated by duration match; write data/download_plan.tsv for
                        whatever's left.

  --stage download      spotdl the remaining plan rows, duration-gate, resumable.

Reuses the archive URLs + tempfile-caching pattern from
scripts/extract_isophonics_labs.py (copied below rather than imported --
scripts/ has no __init__.py, so importing across dirs needs a sys.path hack
for two constants and a 6-line function).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import librosa

HERE = Path(__file__).resolve().parent  # backend/chord_training
BACKEND = HERE.parent  # backend/
DATA_DIR = HERE / "data"
TRAIN_DIR = DATA_DIR / "train"
TRAIN_TRACKS_JSON = DATA_DIR / "train_tracks.json"
DOWNLOAD_PLAN = DATA_DIR / "download_plan.tsv"

GOLD_HOLDOUT = BACKEND / "analysis" / "gold_holdout_v2.json"
GOLD_MIR_TRACKS = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
CHORD_REFS = BACKEND / "tests" / "fixtures" / "chord_refs.json"

DURATION_TOLERANCE_SEC = 2.0

# Same as scripts/extract_isophonics_labs.py's ARCHIVE_URLS.
ARCHIVE_URLS = {
    "beatles": "http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
    "queen": "http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
}


def _load_archive(name: str) -> tarfile.TarFile:
    """Download (or reuse cached) archive. Same tempdir cache filename as
    extract_isophonics_labs.py's _load_archive, so a warm cache from that
    script (or a prior run of this one) is reused instead of re-downloaded."""
    url = ARCHIVE_URLS[name]
    tmp = Path(tempfile.gettempdir()) / f"isophonics_{name}.tar.gz"
    if not tmp.is_file():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tmp)
    return tarfile.open(tmp, "r:gz")


_TRACK_PREFIX_RE = re.compile(r"^(?:cd\d+[\s_.-]+)?\d+[\s_.-]+", re.IGNORECASE)


def _derive_title(stem: str) -> str:
    """Filename stem minus leading track-number prefix, e.g.
    '06_-_Let_It_Be' -> 'Let It Be', '02 Another One Bites The Dust' -> same,
    'CD1_-_01_-_Back_in_the_USSR' -> 'Back in the USSR'.
    ponytail: regex covers Isophonics' two prefix conventions (NN_-_ and
    CDn_-_NN_-_); a track title that itself starts with a number (rare in
    this catalog) would get over-stripped -- upgrade to an explicit
    exception list if that ever bites."""
    stripped = _TRACK_PREFIX_RE.sub("", stem, count=1) or stem
    return stripped.replace("_", " ").strip()


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text.strip("-")


def _lab_end_seconds(lab_path: Path) -> float:
    """Max end-time across all lines. Mirrors gold_audio_verify.lab_end_time's
    logic (not imported -- that module pulls in librosa/spotify_metadata for
    gold-specific baseline comparisons this script doesn't need)."""
    last = 0.0
    for line in lab_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            last = max(last, float(parts[1]))
    return last


def stage_labs() -> int:
    gold_paths = set(json.loads(GOLD_HOLDOUT.read_text())["isophonics_paths"])
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    total_found = 0
    gold_excluded = 0

    archives = {name: _load_archive(name) for name in ARCHIVE_URLS}
    try:
        for tf in archives.values():
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                path = member.name
                if not path.startswith("chordlab/") or not path.endswith(".lab"):
                    continue
                total_found += 1
                if path in gold_paths:
                    gold_excluded += 1
                    continue
                parts = path.split("/")
                artist = parts[1] if len(parts) > 1 else ""
                album = parts[-2] if len(parts) >= 3 else ""
                title = _derive_title(Path(parts[-1]).stem)
                entries.append({
                    "member": member, "tf": tf, "archive_path": path,
                    "artist": artist, "album": album, "title": title,
                    "base_slug": _slugify(title),
                })

        # Collision resolution: same base slug -> disambiguate with album;
        # if that still collides, fall back to a numeric suffix.
        counts = Counter(e["base_slug"] for e in entries)
        used: set[str] = set()
        for e in entries:
            slug = e["base_slug"]
            if counts[slug] > 1:
                slug = f"{slug}-{_slugify(e['album'])}"
            final, n = slug, 2
            while final in used:
                final = f"{slug}-{n}"
                n += 1
            used.add(final)
            e["slug"] = final

        train_tracks: dict[str, dict] = {}
        failures = []
        for e in entries:
            extracted = e["tf"].extractfile(e["member"])
            if extracted is None:
                failures.append((e["archive_path"], "extractfile returned None"))
                continue
            dest = TRAIN_DIR / f"{e['slug']}.lab"
            dest.write_bytes(extracted.read())
            try:
                end_seconds = _lab_end_seconds(dest)
                if end_seconds <= 0:
                    raise ValueError("no parsable end time")
            except Exception as exc:
                failures.append((e["archive_path"], str(exc)))
                dest.unlink(missing_ok=True)
                continue
            train_tracks[e["slug"]] = {
                "artist": e["artist"],
                "title": e["title"],
                "album": e["album"],
                "archive_path": e["archive_path"],
                "lab_end_seconds": end_seconds,
            }
    finally:
        for tf in archives.values():
            tf.close()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_TRACKS_JSON.write_text(json.dumps(train_tracks, indent=2, sort_keys=True) + "\n")

    print(f"chord labs found: {total_found}")
    print(f"gold-excluded: {gold_excluded}")
    print(f"written: {len(train_tracks)}")
    collisions = sum(1 for c in counts.values() if c > 1)
    if collisions:
        print(f"title collisions resolved (album-suffixed): {collisions}")
    if failures:
        print(f"parse failures: {len(failures)}", file=sys.stderr)
        for path, err in failures:
            print(f"  FAIL {path}: {err}", file=sys.stderr)
    return 0


def _fixture_index() -> dict[tuple[str, str], list[Path]]:
    """(slugified title, slugified artist) -> candidate audio paths, built
    from the two fixture catalogs that already have downloaded audio."""
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)

    gold = json.loads(GOLD_MIR_TRACKS.read_text())
    for t in gold.get("tracks", []):
        title = t["title"].split(" — ")[0].strip()
        index[(_slugify(title), _slugify(t["artist"]))].append(BACKEND / t["file"])

    refs = json.loads(CHORD_REFS.read_text())
    for t in refs.get("tracks", []):
        raw = t.get("title", "")
        if " — " in raw:
            title, artist = (x.strip() for x in raw.rsplit(" — ", 1))
        elif " - " in raw:
            title, artist = (x.strip() for x in raw.rsplit(" - ", 1))
        else:
            title, artist = raw, ""
        index[(_slugify(title), _slugify(artist))].append(BACKEND / t["file"])

    return index


def _audio_duration(path: Path) -> float:
    return float(librosa.get_duration(path=str(path)))


def stage_audio_plan() -> int:
    train_tracks = json.loads(TRAIN_TRACKS_JSON.read_text())
    index = _fixture_index()

    matched = 0
    plan_rows = []
    for slug in sorted(train_tracks):
        meta = train_tracks[slug]
        if (TRAIN_DIR / f"{slug}.mp3").exists() or (TRAIN_DIR / f"{slug}.m4a").exists():
            continue

        key = (_slugify(meta["title"]), _slugify(meta["artist"]))
        found = False
        for cand in index.get(key, []):
            if not cand.is_file():
                continue
            try:
                dur = _audio_duration(cand)
            except Exception as exc:
                print(f"WARN duration probe failed for {cand}: {exc}", file=sys.stderr)
                continue
            if abs(dur - meta["lab_end_seconds"]) <= DURATION_TOLERANCE_SEC:
                shutil.copy2(cand, TRAIN_DIR / f"{slug}.mp3")
                matched += 1
                found = True
                break
        if not found:
            plan_rows.append((slug, meta["artist"], meta["title"], meta["lab_end_seconds"]))

    lines = [f"{s}\t{a}\t{t}\t{e}" for s, a, t, e in plan_rows]
    DOWNLOAD_PLAN.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"matched-local: {matched}")
    print(f"download plan: {len(plan_rows)}")
    return 0


def stage_download(plan_path: Path, limit: int | None, dest: Path = TRAIN_DIR) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        slug, artist, title, end_s = line.split("\t")
        rows.append((slug, artist, title, float(end_s)))
    if limit is not None:
        rows = rows[:limit]

    spotdl_py = BACKEND / ".venv" / "bin" / "python"
    for slug, artist, title, end_s in rows:
        if (dest / f"{slug}.mp3").exists() or (dest / f"{slug}.m4a").exists():
            continue  # resumable: already fetched

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"spotdl_{slug}_"))
        try:
            query = f"{artist} - {title}"
            subprocess.run(
                [str(spotdl_py), "-m", "spotdl", "download", query,
                 "--output", str(tmp_dir), "--format", "mp3"],
                check=True, cwd=str(BACKEND), capture_output=True, text=True,
            )
            candidates = sorted(tmp_dir.glob("*.mp3"))
            if not candidates:
                raise FileNotFoundError("spotdl produced no mp3")
            src = candidates[0]
            dur = _audio_duration(src)
            if abs(dur - end_s) <= DURATION_TOLERANCE_SEC:
                shutil.move(str(src), str(dest / f"{slug}.mp3"))
                print(f"OK {slug} (duration {dur:.1f}s vs lab {end_s:.1f}s)")
            else:
                print(f"REJECT {slug} (duration {dur:.1f} vs {end_s:.1f})")
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"FAIL {slug}: spotdl exit {exc.returncode}: {err[0][:200]}")
        except Exception as exc:
            print(f"FAIL {slug}: {exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    train_tracks = json.loads(TRAIN_TRACKS_JSON.read_text())
    ready = sum(
        1 for s in train_tracks
        if (dest / f"{s}.mp3").exists() or (dest / f"{s}.m4a").exists()
    )
    print(f"train pairs ready: {ready}/{len(train_tracks)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B training-data fetch (labs / audio-plan / download)")
    ap.add_argument("--stage", required=True, choices=["labs", "audio-plan", "download"])
    ap.add_argument("--plan", type=Path, default=DOWNLOAD_PLAN, help="download stage: plan TSV to consume")
    ap.add_argument("--limit", type=int, default=None, help="download stage: cap rows processed")
    ap.add_argument("--dest", type=Path, default=TRAIN_DIR, help="download stage: destination dir for audio files (default: data/train/)")
    args = ap.parse_args()

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "labs":
        return stage_labs()
    if args.stage == "audio-plan":
        return stage_audio_plan()
    return stage_download(args.plan, args.limit, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
