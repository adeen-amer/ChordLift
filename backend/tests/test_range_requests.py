"""Byte-range parsing for /api/audio.

Without 206 responses the browser reports the audio stream as unseekable, so
progress-bar seeking and the A/B practice loop both silently do nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from range_requests import UnsatisfiableRangeError, iter_file_range, parse_byte_range

SIZE = 1000


@pytest.mark.parametrize(
    "header,expected",
    [
        ("bytes=0-499", (0, 499)),
        ("bytes=500-999", (500, 999)),
        ("bytes=500-", (500, 999)),          # open-ended -> end of file
        ("bytes=0-", (0, 999)),
        ("bytes=-100", (900, 999)),          # suffix: last 100 bytes
        ("bytes=0-99999", (0, 999)),         # end past EOF clamps
        ("bytes=-99999", (0, 999)),          # suffix longer than file clamps
        ("  bytes=0-9  ", (0, 9)),           # surrounding whitespace tolerated
        ("bytes=999-999", (999, 999)),       # single final byte
    ],
)
def test_parse_byte_range_valid(header, expected):
    assert parse_byte_range(header, SIZE) == expected


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "bytes=",
        "bytes=-",
        "bytes=abc-def",
        "items=0-99",                 # wrong unit
        "bytes=0-99, 200-299",        # multi-range needs multipart; serve whole
        "0-99",                       # missing unit
        "bytes=0-99extra",
    ],
)
def test_parse_byte_range_falls_back_to_whole_file(header):
    """A malformed header must not fail the request -- serve the whole file."""
    assert parse_byte_range(header, SIZE) is None


def test_parse_byte_range_ignores_range_on_empty_file():
    assert parse_byte_range("bytes=0-10", 0) is None


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=1000-1005", "bytes=5000-6000", "bytes=-0"])
def test_parse_byte_range_unsatisfiable(header):
    with pytest.raises(UnsatisfiableRangeError):
        parse_byte_range(header, SIZE)


def test_parse_byte_range_rejects_inverted_range():
    with pytest.raises(UnsatisfiableRangeError):
        parse_byte_range("bytes=500-100", SIZE)


def test_iter_file_range_yields_exact_slice(tmp_path):
    data = bytes(range(256)) * 8  # 2048 bytes
    f = tmp_path / "a.bin"
    f.write_bytes(data)

    got = b"".join(iter_file_range(f, 100, 199, chunk_size=7))

    assert got == data[100:200]
    assert len(got) == 100


def test_iter_file_range_reads_final_byte(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"0123456789")

    assert b"".join(iter_file_range(f, 9, 9)) == b"9"


def test_iter_file_range_covers_whole_file(tmp_path):
    data = b"x" * 5000
    f = tmp_path / "a.bin"
    f.write_bytes(data)

    assert b"".join(iter_file_range(f, 0, 4999, chunk_size=512)) == data
