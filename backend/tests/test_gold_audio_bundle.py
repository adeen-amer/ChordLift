"""Tests for gold audio bundle fingerprinting and verification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_audio_bundle import (
    approved_gold_files,
    manifest_fingerprint,
    verify_extracted_bundle,
)
from gold_audio_verify import DEFAULT_CATALOG, DEFAULT_MANIFEST, approved_track_ids


def test_manifest_fingerprint_stable():
    a = manifest_fingerprint()
    b = manifest_fingerprint()
    assert a == b
    assert len(a) == 16


def test_approved_gold_file_count():
    files = approved_gold_files()
    assert len(files) == len(approved_track_ids())
    assert all(path.startswith("downloads/spotify-") for path in files)


def test_verify_extracted_bundle_reports_missing(tmp_path: Path, monkeypatch):
    backend = tmp_path / "backend"
    backend.mkdir()
    catalog = backend / "tests/fixtures/gold_mir_tracks_v2.json"
    manifest = backend / "tests/fixtures/gold_audio_identity.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    real_catalog = json.loads(Path(DEFAULT_CATALOG).read_text())
    real_manifest = json.loads(Path(DEFAULT_MANIFEST).read_text())
    one_track = real_catalog["tracks"][0]
    catalog.write_text(json.dumps({"tracks": [one_track]}))
    manifest.write_text(
        json.dumps(
            {
                "tracks": {
                    one_track["id"]: real_manifest["tracks"][one_track["id"]],
                }
            }
        )
    )

    monkeypatch.setattr("gold_audio_bundle.DEFAULT_CATALOG", catalog)
    monkeypatch.setattr("gold_audio_bundle.DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr("gold_audio_verify.DEFAULT_CATALOG", catalog)
    monkeypatch.setattr("gold_audio_verify.DEFAULT_MANIFEST", manifest)

    errors = verify_extracted_bundle(backend=backend, catalog_path=catalog, manifest_path=manifest)
    assert errors
    assert any("missing audio" in err for err in errors)
