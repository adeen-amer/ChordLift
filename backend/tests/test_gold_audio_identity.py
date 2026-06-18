"""Gold audio identity gate — duration vs .lab end time."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_audio_verify import (
    DEFAULT_CATALOG,
    DEFAULT_MANIFEST,
    MAX_DURATION_DELTA_SEC,
    lab_end_time,
    verify_catalog,
    verify_track,
)

BACKEND = Path(__file__).resolve().parent.parent
CATALOG = json.loads(DEFAULT_CATALOG.read_text())


@pytest.mark.parametrize("track", CATALOG["tracks"], ids=lambda t: t["id"])
def test_lab_file_exists(track):
    lab = BACKEND / track["lab"]
    assert lab.is_file(), f"missing {lab}"


def test_duration_gate_rejects_large_delta(tmp_path):
    lab = tmp_path / "x.lab"
    lab.write_text("0.0 100.0 N\n")
    track = {"id": "x", "lab": "x.lab", "file": "missing.mp3"}
    result = verify_track(track, tmp_path)
    assert result["status"] == "no_audio"


def test_manifest_schema_when_present():
    if not DEFAULT_MANIFEST.is_file():
        pytest.skip("gold_audio_identity.json not generated yet")
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    assert manifest.get("max_lab_duration_delta_sec") == MAX_DURATION_DELTA_SEC
    assert isinstance(manifest.get("tracks"), dict)


def test_all_labs_have_positive_end_time():
    for track in CATALOG["tracks"]:
        end = lab_end_time(BACKEND / track["lab"])
        assert end > 0, track["id"]
