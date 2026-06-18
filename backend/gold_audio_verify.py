"""Gold audio identity gate — hardened Phase 11.5 (lab + Spotify duration + manual scrutiny)."""
from __future__ import annotations

import json
from pathlib import Path

import librosa

from spotify_metadata import verify_audio_matches_spotify

BACKEND = Path(__file__).resolve().parent
DEFAULT_CATALOG = BACKEND / "tests" / "fixtures" / "gold_mir_tracks_v2.json"
DEFAULT_MANIFEST = BACKEND / "tests" / "fixtures" / "gold_audio_identity.json"
V46_DEV = BACKEND / "analysis" / "BASELINE_mir_gold_DEV_v46.json"
V46_TEST = BACKEND / "analysis" / "BASELINE_mir_gold_TEST_v46.json"

# Lab: tightened below 2s for scrutiny; gross drift still fails when Spotify also mismatches badly
MAX_LAB_DURATION_DELTA_SEC = 1.99
MAX_LAB_GROSS_DRIFT_SEC = 5.0
MAX_SPOTIFY_DURATION_DELTA_SEC = 1.0
SCRUTINY_LAB_DRIFT_SEC = 1.0
SCRUTINY_LOW_MAJMIN = 0.50

# Back-compat alias for tests / scripts
MAX_DURATION_DELTA_SEC = MAX_LAB_DURATION_DELTA_SEC


def lab_end_time(lab_path: Path) -> float:
    last = 0.0
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            last = max(last, float(parts[1]))
    return last


def ear_check_points(lab_end: float) -> list[float]:
    if lab_end <= 0:
        return [0.0, 0.0, 0.0]
    return [
        round(min(15.0, lab_end * 0.1), 2),
        round(lab_end * 0.45, 2),
        round(max(0.0, lab_end - 10.0), 2),
    ]


def _v46_majmin_by_id() -> dict[str, float]:
    out: dict[str, float] = {}
    for path in (V46_DEV, V46_TEST):
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for t in data.get("tracks", []):
            if t.get("majmin") is not None:
                out[t["id"]] = float(t["majmin"])
    return out


def _needs_manual_scrutiny(tid: str, lab_delta: float | None, v46_majmin: dict[str, float]) -> tuple[bool, str]:
    reasons = []
    if lab_delta is not None and lab_delta > SCRUTINY_LAB_DRIFT_SEC:
        reasons.append(f"lab_drift>{SCRUTINY_LAB_DRIFT_SEC}s")
    mm = v46_majmin.get(tid)
    if mm is not None and mm < SCRUTINY_LOW_MAJMIN:
        reasons.append(f"v46_majmin<{SCRUTINY_LOW_MAJMIN}")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def _lab_duration_ok(lab_delta: float | None, spotify_status: str) -> tuple[bool, str | None]:
    if lab_delta is None:
        return True, None
    if spotify_status == "pass":
        if lab_delta > MAX_LAB_GROSS_DRIFT_SEC:
            return False, f"lab delta {lab_delta:.2f}s > {MAX_LAB_GROSS_DRIFT_SEC}s (gross drift)"
        return True, None
    if lab_delta > MAX_LAB_DURATION_DELTA_SEC:
        return False, f"lab delta {lab_delta:.2f}s > {MAX_LAB_DURATION_DELTA_SEC}s"
    return True, None


def verify_track(track: dict, backend: Path = BACKEND, *, v46_majmin: dict[str, float] | None = None) -> dict:
    tid = track["id"]
    lab_path = backend / track["lab"]
    audio_path = backend / track["file"]
    lab_end = lab_end_time(lab_path) if lab_path.is_file() else None
    v46_majmin = v46_majmin if v46_majmin is not None else _v46_majmin_by_id()

    entry: dict = {
        "id": tid,
        "title": track.get("title"),
        "label_tier": track.get("label_tier"),
        "lab_end_sec": lab_end,
        "ear_check_sec": ear_check_points(lab_end or 0.0),
        "ear_check_status": "pending",
        "needs_manual_review": False,
        "manual_review_reason": "",
    }

    if not lab_path.is_file():
        entry["status"] = "fail"
        entry["reason"] = f"missing lab {lab_path}"
        return entry

    if not audio_path.is_file():
        entry["status"] = "no_audio"
        entry["reason"] = f"missing audio {audio_path}"
        return entry

    duration = float(librosa.get_duration(path=str(audio_path)))
    lab_delta = abs(duration - lab_end) if lab_end is not None else None
    entry["audio_duration_sec"] = round(duration, 3)
    entry["duration_delta_sec"] = round(lab_delta, 3) if lab_delta is not None else None

    spotify_url = track.get("url", "")
    spotify_check = verify_audio_matches_spotify(
        spotify_url, duration, tolerance_sec=MAX_SPOTIFY_DURATION_DELTA_SEC,
    )
    entry.update(spotify_check)

    scrutiny, scrutiny_reason = _needs_manual_scrutiny(tid, lab_delta, v46_majmin)
    entry["needs_manual_review"] = scrutiny
    entry["manual_review_reason"] = scrutiny_reason
    if scrutiny:
        entry["ear_check_status"] = "pending"

    failures = []
    spotify_status = spotify_check.get("spotify_duration_status", "unknown")
    if spotify_status == "fail":
        failures.append(
            f"Spotify duration delta {spotify_check['spotify_delta_sec']:.2f}s "
            f"> {MAX_SPOTIFY_DURATION_DELTA_SEC}s"
        )
    lab_ok, lab_reason = _lab_duration_ok(lab_delta, spotify_status)
    if not lab_ok and lab_reason:
        failures.append(lab_reason)

    if failures:
        entry["status"] = "fail"
        entry["reason"] = "; ".join(failures)
        return entry

    if spotify_status == "unknown":
        entry["status"] = "pass_with_warning"
        entry["reason"] = "Spotify official duration unavailable — manual ear-check required"
        entry["needs_manual_review"] = True
        if not entry["manual_review_reason"]:
            entry["manual_review_reason"] = "spotify_duration_unknown"
        return entry

    entry["status"] = "pass"
    if lab_delta is not None and lab_delta > MAX_LAB_DURATION_DELTA_SEC:
        entry["lab_drift_note"] = (
            f"Spotify-verified remaster; lab drift {lab_delta:.2f}s > {MAX_LAB_DURATION_DELTA_SEC}s "
            "(1987 annotation vs 2009 remaster — ear-check required)"
        )
    return entry


def verify_catalog(
    catalog_path: Path = DEFAULT_CATALOG,
    backend: Path = BACKEND,
    *,
    require_ear_approved: bool = False,
) -> tuple[list[dict], list[dict]]:
    catalog = json.loads(catalog_path.read_text())
    v46 = _v46_majmin_by_id()
    results = [verify_track(t, backend, v46_majmin=v46) for t in catalog["tracks"]]
    failed = [r for r in results if r.get("status") in ("fail", "no_audio")]
    if require_ear_approved:
        manifest = load_manifest(DEFAULT_MANIFEST)
        approved = {
            k for k, v in manifest.get("tracks", {}).items()
            if v.get("ear_check_status") == "approved"
        }
        for r in results:
            if r.get("status") in ("pass", "pass_with_warning") and r["id"] not in approved:
                r["status"] = "fail"
                r["reason"] = "ear_check not approved in gold_audio_identity.json"
                failed.append(r)
    return results, failed


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"tracks": {}}


def write_manifest(results: list[dict], path: Path = DEFAULT_MANIFEST) -> None:
    existing = load_manifest(path)
    tracks = existing.get("tracks", {})
    for r in results:
        tid = r["id"]
        prev = tracks.get(tid, {})
        ear = prev.get("ear_check_status", "pending")
        if r.get("ear_check_status") == "approved":
            ear = "approved"
        elif r.get("needs_manual_review") and ear != "approved":
            ear = "pending"
        tracks[tid] = {
            **prev,
            "lab_end_sec": r.get("lab_end_sec"),
            "audio_duration_sec": r.get("audio_duration_sec"),
            "duration_delta_sec": r.get("duration_delta_sec"),
            "duration_status": r.get("status"),
            "lab_drift_note": r.get("lab_drift_note"),
            "spotify_track_id": r.get("spotify_track_id"),
            "spotify_duration_sec": r.get("spotify_duration_sec"),
            "spotify_delta_sec": r.get("spotify_delta_sec"),
            "spotify_duration_status": r.get("spotify_duration_status"),
            "needs_manual_review": r.get("needs_manual_review", False),
            "manual_review_reason": r.get("manual_review_reason", ""),
            "ear_check_sec": r.get("ear_check_sec"),
            "ear_check_status": ear,
        }
    payload = {
        "description": (
            "Gold audio identity gate (Phase 11.5). Requires Spotify official duration within 1s "
            "(independent wrong-version anchor) plus lab sanity check; manual ear-check for "
            f"lab drift >{SCRUTINY_LAB_DRIFT_SEC}s or outlier v46 majmin."
        ),
        "max_lab_duration_delta_sec": MAX_LAB_DURATION_DELTA_SEC,
        "max_lab_gross_drift_sec": MAX_LAB_GROSS_DRIFT_SEC,
        "max_spotify_duration_delta_sec": MAX_SPOTIFY_DURATION_DELTA_SEC,
        "tracks": tracks,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def approved_track_ids(manifest_path: Path = DEFAULT_MANIFEST) -> set[str]:
    manifest = load_manifest(manifest_path)
    return {
        tid
        for tid, meta in manifest.get("tracks", {}).items()
        if meta.get("duration_status") == "pass"
        and meta.get("spotify_duration_status") == "pass"
        and meta.get("ear_check_status") == "approved"
        and not meta.get("needs_manual_review")
    }
