"""Safe resolution of cache/download paths (Phase 6 — block path traversal)."""
from __future__ import annotations

import os
import re
from pathlib import Path

SAFE_SOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class InvalidSourceIdError(ValueError):
    pass


def validate_source_id(source_id: str) -> str:
    if not source_id or not SAFE_SOURCE_ID_RE.match(source_id):
        raise InvalidSourceIdError("Invalid source id")
    return source_id


def _resolve_under(base_dir: str, filename: str) -> Path:
    base = Path(base_dir).resolve()
    path = (base / filename).resolve()
    if path != base and base not in path.parents:
        raise InvalidSourceIdError("Path escapes base directory")
    return path


def resolve_download_mp3(source_id: str) -> Path:
    validate_source_id(source_id)
    return _resolve_under(os.environ.get("DOWNLOAD_DIR", "downloads"), f"{source_id}.mp3")


def resolve_cache_json(source_id: str) -> Path:
    validate_source_id(source_id)
    return _resolve_under(os.environ.get("CACHE_DIR", "cache"), f"{source_id}.json")
