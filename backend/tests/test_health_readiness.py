"""/api/health readiness reporting for the two engines that degrade silently.

track_beats_auto falls back to the librosa heuristic on any failure, and the
Spotify duration gate accepts "unknown" -- both produce plausible output while
serving the weak path, so health has to say which one is live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import beat_tracking
import spotify_metadata


# ---- spotify ------------------------------------------------------------

def test_spotify_readiness_reports_api_when_credentials_set(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")

    status = spotify_metadata.spotify_readiness()

    assert status["credentials_configured"] is True
    assert status["duration_verification"] == "api"
    assert status["setup_hint"] is None


def test_spotify_readiness_flags_ytmusic_fallback_without_credentials(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    status = spotify_metadata.spotify_readiness()

    assert status["credentials_configured"] is False
    assert status["duration_verification"] == "ytmusic-fallback"
    assert status["setup_hint"]  # operator needs to know verification is weak


def test_spotify_readiness_treats_blank_credentials_as_unset(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "  ")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "")

    assert spotify_metadata.spotify_readiness()["credentials_configured"] is False


# ---- beat engine --------------------------------------------------------

@pytest.mark.parametrize(
    "requested,madmom,transformer,expected",
    [
        ("librosa", False, False, "librosa"),
        ("librosa", True, True, "librosa"),
        ("auto", False, False, "librosa"),
        ("auto", True, False, "madmom"),
        ("auto", True, True, "transformer-or-madmom"),
        ("madmom", True, True, "madmom"),
        ("madmom", False, True, "unavailable"),
        ("transformer", True, True, "transformer"),
        ("transformer", False, True, "unavailable"),
    ],
)
def test_beat_engine_readiness_effective_engine(
    monkeypatch, requested, madmom, transformer, expected,
):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", requested)
    monkeypatch.setattr(
        beat_tracking, "_module_available",
        lambda name: madmom if name == "madmom" else transformer,
    )

    assert beat_tracking.beat_engine_readiness()["effective_engine"] == expected


def test_beat_engine_readiness_transformer_requires_madmom(monkeypatch):
    """beat_transformer.infer imports madmom's DBN processors, so an importable
    transformer module without madmom is not a usable transformer path."""
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")
    monkeypatch.setattr(
        beat_tracking, "_module_available",
        lambda name: name != "madmom",  # transformer importable, madmom absent
    )

    status = beat_tracking.beat_engine_readiness()

    assert status["transformer_available"] is False
    assert status["effective_engine"] == "librosa"
    assert status["setup_hint"]


def test_beat_engine_readiness_silent_fallback_is_flagged(monkeypatch):
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "auto")
    monkeypatch.setattr(beat_tracking, "_module_available", lambda name: False)

    assert beat_tracking.beat_engine_readiness()["setup_hint"]


def test_beat_engine_readiness_explicit_librosa_is_not_degraded(monkeypatch):
    """Asking for librosa and getting librosa is a choice, not a degradation."""
    monkeypatch.setattr(beat_tracking, "BEAT_ENGINE", "librosa")
    monkeypatch.setattr(beat_tracking, "_module_available", lambda name: False)

    assert beat_tracking.beat_engine_readiness()["setup_hint"] is None


# ---- endpoint -----------------------------------------------------------

def test_health_endpoint_exposes_both_readiness_blocks():
    """Calls the handler directly: starlette 0.27's TestClient is incompatible
    with the installed httpx 0.28 (`Client.__init__` no longer takes `app`),
    and this repo has no HTTP-level API tests to borrow a pattern from. The
    handler is a plain async function, so asyncio.run is enough."""
    import asyncio

    import main

    body = asyncio.run(main.health())

    assert body["status"] == "ok"
    assert body["beat_engine"]["effective_engine"] in (
        "librosa", "madmom", "transformer", "transformer-or-madmom", "unavailable",
    )
    assert isinstance(body["spotify"]["credentials_configured"], bool)
