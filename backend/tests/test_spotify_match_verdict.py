"""The Spotify match verdict must reach the client.

_download_spotify_via_spotdl only rejects a "fail". An "unknown" verdict --
which is what you get without SPOTIFY_CLIENT_ID/SECRET, since duration then
comes from a YT Music search that spotdl itself also resolves against -- means
nothing verified the audio is the track the user asked for. Before this the
verdict was logged and dropped, so the UI could not say so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import downloader


def test_verdict_round_trips_through_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))
    verdict = {"spotify_duration_status": "unknown", "spotify_track_id": "abc123"}

    downloader.write_match_verdict("src-1", verdict)

    assert downloader.read_match_verdict("src-1") == verdict


def test_missing_verdict_reads_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))

    assert downloader.read_match_verdict("never-written") is None


def test_corrupt_verdict_reads_as_none_rather_than_raising(tmp_path, monkeypatch):
    """A truncated sidecar must not take down an otherwise fine analysis."""
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / "src-2.match.json").write_text("{not json")

    assert downloader.read_match_verdict("src-2") is None


def test_write_survives_an_unwritable_directory(tmp_path, monkeypatch):
    """Persisting the verdict is best-effort -- it must never fail a download
    that otherwise succeeded."""
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path / "does-not-exist"))

    downloader.write_match_verdict("src-3", {"spotify_duration_status": "pass"})


def test_enrich_analysis_attaches_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))
    verdict = {"spotify_duration_status": "unknown"}
    source_id = downloader.extract_source_id("https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT")
    downloader.write_match_verdict(source_id, verdict)

    import main

    data = main._enrich_analysis(
        {"song": {"title": "x"}, "timeline": [], "lyrics": {"synced": False, "lines": []}},
        "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
    )

    assert data["spotify_match"] == verdict


def test_enrich_analysis_omits_verdict_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))

    import main

    data = main._enrich_analysis(
        {"song": {"title": "x"}, "timeline": [], "lyrics": {"synced": False, "lines": []}},
        "https://open.spotify.com/track/1cOdK2wGLETKBW3PvgPWqT",
    )

    assert "spotify_match" not in data
