"""Gold eval audio bundle — build locally, fetch in CI (no YouTube/spotdl in runners)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import librosa

from gold_audio_verify import (
    BACKEND,
    DEFAULT_CATALOG,
    DEFAULT_MANIFEST,
    approved_track_ids,
    load_manifest,
)

BUNDLE_NAME = "gold_audio_v2.tar.gz"
RELEASE_TAG = os.getenv("GOLD_AUDIO_RELEASE_TAG", "gold-audio-v2")
DURATION_VERIFY_TOLERANCE_SEC = 0.15


def _catalog_tracks(catalog_path: Path = DEFAULT_CATALOG) -> list[dict]:
    return json.loads(catalog_path.read_text(encoding="utf-8"))["tracks"]


def approved_gold_files(
    catalog_path: Path = DEFAULT_CATALOG,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> list[str]:
    approved = approved_track_ids(manifest_path)
    files: list[str] = []
    for track in _catalog_tracks(catalog_path):
        if track["id"] in approved and track.get("file"):
            files.append(track["file"])
    return sorted(files)


def manifest_fingerprint(
    manifest_path: Path = DEFAULT_MANIFEST,
    catalog_path: Path = DEFAULT_CATALOG,
) -> str:
    """Stable cache/release key when identity manifest or gold file set changes."""
    approved = approved_track_ids(manifest_path)
    files = approved_gold_files(catalog_path, manifest_path)
    payload = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "approved_track_count": len(approved),
        "files": files,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
    ).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_manifest_entries(
    backend: Path = BACKEND,
    manifest_path: Path = DEFAULT_MANIFEST,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict:
    identity = load_manifest(manifest_path)
    approved = approved_track_ids(manifest_path)
    tracks_meta: dict[str, dict] = {}
    files: list[dict] = []

    for track in _catalog_tracks(catalog_path):
        tid = track["id"]
        if tid not in approved:
            continue
        rel = track["file"]
        audio_path = backend / rel
        if not audio_path.is_file():
            raise FileNotFoundError(f"missing gold audio: {audio_path}")
        duration = float(librosa.get_duration(path=str(audio_path)))
        anchor = identity.get("tracks", {}).get(tid, {})
        expected = anchor.get("audio_duration_sec")
        files.append(
            {
                "id": tid,
                "path": rel,
                "sha256": _file_sha256(audio_path),
                "audio_duration_sec": round(duration, 3),
                "expected_duration_sec": expected,
            }
        )
        tracks_meta[tid] = {
            "path": rel,
            "sha256": files[-1]["sha256"],
            "audio_duration_sec": files[-1]["audio_duration_sec"],
            "expected_duration_sec": expected,
        }

    return {
        "bundle_version": 2,
        "fingerprint": manifest_fingerprint(manifest_path, catalog_path),
        "release_tag": RELEASE_TAG,
        "track_count": len(files),
        "files": files,
        "tracks": tracks_meta,
    }


def build_bundle(
    output_path: Path,
    backend: Path = BACKEND,
    manifest_path: Path = DEFAULT_MANIFEST,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict:
    """Pack identity-verified gold mp3s into a tarball for CI/release."""
    meta = bundle_manifest_entries(backend, manifest_path, catalog_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, "w:gz") as tar:
        manifest_bytes = (json.dumps(meta, indent=2) + "\n").encode("utf-8")
        manifest_info = tarfile.TarInfo(name="gold_audio_v2/BUNDLE.json")
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_bytes))

        for entry in meta["files"]:
            audio_path = backend / entry["path"]
            tar.add(audio_path, arcname=f"gold_audio_v2/{entry['path']}")

    return meta


def extract_bundle(archive_path: Path, backend: Path = BACKEND) -> dict:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=backend.parent, filter="data")
    bundle_json = backend.parent / "gold_audio_v2" / "BUNDLE.json"
    if not bundle_json.is_file():
        raise FileNotFoundError(f"missing {bundle_json} inside archive")
    meta = json.loads(bundle_json.read_text(encoding="utf-8"))
    for entry in meta.get("files", []):
        src = backend.parent / "gold_audio_v2" / entry["path"]
        dest = backend / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            raise FileNotFoundError(f"bundle missing {entry['path']}")
        dest.write_bytes(src.read_bytes())
    return meta


def verify_extracted_bundle(
    backend: Path = BACKEND,
    manifest_path: Path = DEFAULT_MANIFEST,
    catalog_path: Path = DEFAULT_CATALOG,
    *,
    expected_fingerprint: str | None = None,
) -> list[str]:
    """Fail loudly on missing files, hash mismatch, or duration drift vs identity manifest."""
    errors: list[str] = []
    current_fp = manifest_fingerprint(manifest_path, catalog_path)
    if expected_fingerprint and expected_fingerprint != current_fp:
        errors.append(
            f"bundle fingerprint {expected_fingerprint} != repo manifest {current_fp}"
        )

    identity = load_manifest(manifest_path)
    approved = approved_track_ids(manifest_path)

    for track in _catalog_tracks(catalog_path):
        tid = track["id"]
        if tid not in approved:
            continue
        rel = track["file"]
        audio_path = backend / rel
        if not audio_path.is_file():
            errors.append(f"missing audio after extract: {rel}")
            continue

        anchor = identity.get("tracks", {}).get(tid, {})
        expected = anchor.get("audio_duration_sec")
        if expected is None:
            errors.append(f"{tid}: no audio_duration_sec anchor in gold_audio_identity.json")
            continue

        actual = float(librosa.get_duration(path=str(audio_path)))
        delta = abs(actual - float(expected))
        if delta > DURATION_VERIFY_TOLERANCE_SEC:
            errors.append(
                f"{tid}: duration {actual:.3f}s != identity anchor {expected}s (Δ={delta:.3f}s)"
            )

    if errors:
        return errors
    return []
