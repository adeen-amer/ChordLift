"""Analysis concurrency limits and in-flight deduplication."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

MAX_CONCURRENT = max(1, int(os.getenv("MAX_CONCURRENT_ANALYSES", "2")))
_analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_inflight: dict[str, asyncio.Task] = {}
_inflight_downloads: dict[str, asyncio.Task] = {}


async def run_analysis_deduped(
    source_id: str,
    runner: Callable[[], Any],
) -> Any:
    """
    Run analysis under a global semaphore. Concurrent requests for the same
    source_id share one in-flight task.
    """
    existing = _inflight.get(source_id)
    if existing is not None and not existing.done():
        logger.info("Joining in-flight analysis for %s", source_id)
        return await existing

    async def _guarded() -> Any:
        async with _analysis_semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, runner)

    task = asyncio.create_task(_guarded())
    _inflight[source_id] = task
    try:
        return await task
    finally:
        if _inflight.get(source_id) is task:
            del _inflight[source_id]


async def run_download_deduped(source_id: str, runner: Callable[[], Any]) -> Any:
    """
    Share one in-flight download per source_id.

    The download runs before (and outside) run_analysis_deduped, so without
    this every reconnect to /api/progress started another full download of the
    same track -- and EventSource reconnects on its own after any dropped
    connection. Observed in the wild as three concurrent spotdl subprocesses
    for one Spotify link. MAX_CONCURRENT_ANALYSES does not help: it guards the
    analysis executor, not downloads.

    Deliberately not under the analysis semaphore -- downloads are network
    bound, and holding an analysis slot while waiting on one would throttle
    throughput for no gain.

    asyncio.shield keeps the download alive when the client that started it
    disconnects: with SSE the first client dropping is the common case, and
    everyone else waiting on that track should not lose the download with it.
    """
    existing = _inflight_downloads.get(source_id)
    if existing is not None and not existing.done():
        logger.info("Joining in-flight download for %s", source_id)
        return await asyncio.shield(existing)

    task = asyncio.create_task(runner())
    _inflight_downloads[source_id] = task
    try:
        return await asyncio.shield(task)
    finally:
        if _inflight_downloads.get(source_id) is task and task.done():
            del _inflight_downloads[source_id]
