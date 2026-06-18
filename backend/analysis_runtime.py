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
