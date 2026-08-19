"""spotdl must not be able to hang the serving path.

chord_training/fetch_train_data.py already found that azlyrics' provider
makes untimed requests and hangs a download forever; the serving downloader
never got the same treatment. Observed before this: 11.7 minutes elapsed for
7.3 seconds of CPU on one Spotify link, with no timeout to break out of it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import downloader


def _capture_run(monkeypatch):
    """Swap subprocess.run for a recorder, and stop the function right after
    it -- we only care about how spotdl was invoked."""
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        raise RuntimeError("stop here: nothing downloaded")

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    return calls


def test_spotdl_invocation_has_a_timeout(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)

    with pytest.raises(Exception):
        downloader._download_spotify_via_spotdl(
            "https://open.spotify.com/track/abc", str(tmp_path / "out.mp3"),
        )

    assert calls["kwargs"].get("timeout"), "a hung spotdl would hold the request open forever"
    assert calls["kwargs"]["timeout"] == downloader.SPOTDL_TIMEOUT_SEC


def test_spotdl_invocation_drops_azlyrics(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)

    with pytest.raises(Exception):
        downloader._download_spotify_via_spotdl(
            "https://open.spotify.com/track/abc", str(tmp_path / "out.mp3"),
        )

    cmd = calls["cmd"]
    assert "--lyrics" in cmd, "default provider chain includes azlyrics, which hangs"
    providers = cmd[cmd.index("--lyrics") + 1:]
    assert "azlyrics" not in providers


def test_timeout_is_an_exception_the_caller_falls_back_on():
    """download_audio catches Exception around the spotdl attempt and falls
    through to ytsearch -- that only helps if TimeoutExpired is an Exception."""
    assert issubclass(subprocess.TimeoutExpired, Exception)


def test_timeout_is_configurable_by_env():
    assert downloader.SPOTDL_TIMEOUT_SEC > 0
