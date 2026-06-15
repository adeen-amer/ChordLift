"""Tests for lyrics parsing and alignment."""
from lyrics import (
    align_synced_lyrics,
    distribute_plain_lyrics,
    parse_lrc,
    split_artist_title,
)


def test_parse_lrc_basic():
    raw = "[00:12.50]Hello world\n[00:15.00]Second line"
    lines = parse_lrc(raw)
    assert len(lines) == 2
    assert lines[0]["text"] == "Hello world"
    assert abs(lines[0]["time"] - 12.5) < 0.01
    assert lines[1]["text"] == "Second line"


def test_align_synced_lyrics_to_chords():
    timeline = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    lyric_lines = [
        {"time": 0.5, "text": "When"},
        {"time": 1.0, "text": "I"},
        {"time": 2.5, "text": "find"},
    ]
    out = align_synced_lyrics(timeline, lyric_lines)
    assert out[0]["lyrics"] == "When I"
    assert out[1]["lyrics"] == "find"


def test_align_synced_lyrics_overlap_boundary():
    timeline = [{"time": 0.0, "end_time": 2.0, "chord": "C"}]
    lyric_lines = [{"time": 1.8, "text": "find"}]
    out = align_synced_lyrics(timeline, lyric_lines)
    assert out[0]["lyrics"] == "find"


def test_distribute_plain_lyrics():
    timeline = [
        {"time": 0.0, "end_time": 2.0, "chord": "C"},
        {"time": 2.0, "end_time": 4.0, "chord": "G"},
    ]
    plain = "Line one\nLine two\nLine three\nLine four"
    out = distribute_plain_lyrics(timeline, plain)
    assert out[0]["lyrics"]
    assert out[1]["lyrics"]


def test_split_artist_title():
    artist, title = split_artist_title({"title": "The Beatles - Let It Be (Official Video)"})
    assert artist == "The Beatles"
    assert title == "Let It Be"
