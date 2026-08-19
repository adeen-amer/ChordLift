"""HTTP Range parsing for /api/audio (RFC 7233 section 3.1).

Starlette 0.27's FileResponse answers every request with 200 and the whole
body, so a browser marks the stream unseekable (`audio.seekable` is empty) and
seeking silently does nothing. Audio needs byte ranges.

Header values come straight from the client, so anything malformed is ignored
in favour of serving the whole file rather than raising -- a bad Range header
is not worth failing a playback request over.
"""
from __future__ import annotations

import re

# One range only: "bytes=X-Y", "bytes=X-", or "bytes=-N". Multi-range requests
# (comma separated) need a multipart/byteranges body, which no browser needs
# for media playback -- those fall through to the whole-file response.
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class UnsatisfiableRangeError(ValueError):
    """A syntactically valid range that lies outside the file (-> 416)."""

    def __init__(self, file_size: int):
        super().__init__(f"range not satisfiable for {file_size}-byte file")
        self.file_size = file_size


def parse_byte_range(header: str | None, file_size: int) -> tuple[int, int] | None:
    """Resolve a Range header to an inclusive (start, end) byte pair.

    Returns None when the whole file should be served: no header, a malformed
    one, a multi-range request, or a zero-length file. Raises
    UnsatisfiableRangeError when the range parses but cannot be served.
    """
    if not header or file_size <= 0:
        return None

    match = _RANGE_RE.match(header.strip())
    if not match:
        return None

    raw_start, raw_end = match.group(1), match.group(2)

    if not raw_start and not raw_end:
        return None  # "bytes=-" is malformed, not a request for everything

    if not raw_start:
        # Suffix range: the last N bytes. N larger than the file is clamped to
        # the whole file rather than rejected, per RFC 7233 section 2.1.
        suffix = int(raw_end)
        if suffix == 0:
            raise UnsatisfiableRangeError(file_size)
        return max(0, file_size - suffix), file_size - 1

    start = int(raw_start)
    if start >= file_size:
        raise UnsatisfiableRangeError(file_size)

    end = int(raw_end) if raw_end else file_size - 1
    end = min(end, file_size - 1)
    if end < start:
        raise UnsatisfiableRangeError(file_size)

    return start, end


def iter_file_range(path, start: int, end: int, chunk_size: int = 64 * 1024):
    """Yield bytes [start, end] inclusive, without reading the file into RAM."""
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
