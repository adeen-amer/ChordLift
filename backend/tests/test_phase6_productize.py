"""Tests for safe_paths and cache_eviction (Phase 6)."""
from __future__ import annotations

import json
import os
import time

import pytest

from cache_eviction import evict_analysis_cache
from safe_paths import InvalidSourceIdError, resolve_cache_json, resolve_download_mp3, validate_source_id


def test_validate_source_id_accepts_known_formats():
    assert validate_source_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert validate_source_id("spotify-5V1AHQugSTASVez5ffJtFo") == "spotify-5V1AHQugSTASVez5ffJtFo"
    assert validate_source_id("upload-a1b2c3d4e5f6") == "upload-a1b2c3d4e5f6"


@pytest.mark.parametrize(
    "bad_id",
    ["../cache/x", "..\\cache\\x", "foo/bar", "", "a" * 200],
)
def test_validate_source_id_rejects_traversal(bad_id):
    with pytest.raises(InvalidSourceIdError):
        validate_source_id(bad_id)


def test_resolve_download_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    monkeypatch.setenv("DOWNLOAD_DIR", str(downloads))

    safe = resolve_download_mp3("abc123")
    assert safe.parent == downloads.resolve()

    with pytest.raises(InvalidSourceIdError):
        resolve_download_mp3("../etc/passwd")


def test_evict_analysis_cache_lru(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_MAX_ENTRIES", "2")
    monkeypatch.setenv("CACHE_MAX_BYTES", "1000000")
    cache = tmp_path / "cache"
    cache.mkdir()

    for i, name in enumerate(("a", "b", "c")):
        path = cache / f"{name}.json"
        path.write_text(json.dumps({"n": i}))
        os.utime(path, (time.time() + i, time.time() + i))

    removed = evict_analysis_cache(str(cache))
    assert removed == 1
    remaining = {p.name for p in cache.glob("*.json")}
    assert remaining == {"b.json", "c.json"}
