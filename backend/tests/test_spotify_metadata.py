"""Spotify duration metadata for identity gate."""
import pytest

from spotify_metadata import extract_spotify_track_id, verify_audio_matches_spotify


def test_extract_spotify_track_id():
    assert extract_spotify_track_id("https://open.spotify.com/track/71JkCL6msHDScuCKaJXFXP") == "71JkCL6msHDScuCKaJXFXP"


def test_verify_skip_non_spotify():
    check = verify_audio_matches_spotify("https://youtube.com/watch?v=abc", 120.0)
    assert check["spotify_duration_status"] == "skip"
