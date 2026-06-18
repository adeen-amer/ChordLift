"""LRU eviction for analysis cache JSON files."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_BYTES = 500 * 1024 * 1024


def _cache_limits() -> tuple[int, int]:
    max_entries = int(os.getenv("CACHE_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES)))
    max_bytes = int(os.getenv("CACHE_MAX_BYTES", str(DEFAULT_MAX_BYTES)))
    return max(1, max_entries), max(1024, max_bytes)


def evict_analysis_cache(cache_dir: str = "cache") -> int:
    """
    Delete oldest cache/*.json files until under CACHE_MAX_ENTRIES and CACHE_MAX_BYTES.
    Returns number of files removed.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return 0

    max_entries, max_bytes = _cache_limits()
    files = []
    for p in root.glob("*.json"):
        if not p.is_file():
            continue
        try:
            files.append((p.stat().st_mtime, p.stat().st_size, p))
        except OSError:
            continue
    files.sort(key=lambda x: x[0])
    total_bytes = sum(s for _, s, _ in files)
    file_paths = [p for _, _, p in files]
    removed = 0

    while file_paths and (len(file_paths) > max_entries or total_bytes > max_bytes):
        victim = file_paths.pop(0)
        try:
            size = victim.stat().st_size
            victim.unlink(missing_ok=True)
            total_bytes -= size
            removed += 1
        except OSError as exc:
            logger.debug("Cache eviction skip %s: %s", victim, exc)
            continue

    if removed:
        logger.info(
            "Evicted %d cache file(s); %d remain (~%.1f MB)",
            removed,
            len(file_paths),
            total_bytes / (1024 * 1024),
        )
    return removed
