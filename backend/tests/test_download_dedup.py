"""One download per source, however many progress streams ask for it.

/api/progress calls download_audio before (and outside) run_analysis_deduped,
so without this dedup every EventSource reconnect started another full
download of the same track -- observed as three concurrent spotdl subprocesses
for one Spotify link.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from analysis_runtime import run_download_deduped

# No pytest-asyncio in this environment and no async tests anywhere else in
# this suite, so each case is an async body driven by a sync asyncio.run
# wrapper at the bottom of the file rather than a new plugin dependency.


async def _test_concurrent_requests_share_one_download():
    calls = 0

    async def runner():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"audio_path": "a.mp3"}

    results = await asyncio.gather(*(run_download_deduped("src-1", runner) for _ in range(5)))

    assert calls == 1, "each caller started its own download"
    assert all(r == {"audio_path": "a.mp3"} for r in results)


async def _test_sequential_requests_download_again():
    """Dedup is in-flight only -- a later request re-downloads if needed
    rather than replaying a stale result forever."""
    calls = 0

    async def runner():
        nonlocal calls
        calls += 1
        return calls

    assert await run_download_deduped("src-2", runner) == 1
    assert await run_download_deduped("src-2", runner) == 2


async def _test_different_sources_do_not_share():
    calls: list[str] = []

    def make(name):
        async def runner():
            calls.append(name)
            await asyncio.sleep(0.02)
            return name
        return runner

    a, b = await asyncio.gather(
        run_download_deduped("src-a", make("a")),
        run_download_deduped("src-b", make("b")),
    )

    assert (a, b) == ("a", "b")
    assert sorted(calls) == ["a", "b"]


async def _test_failure_propagates_and_clears_the_slot():
    attempts = 0

    async def runner():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        await run_download_deduped("src-3", runner)
    # A failed download must not poison the key for later requests.
    with pytest.raises(RuntimeError):
        await run_download_deduped("src-3", runner)
    assert attempts == 2


async def _test_download_survives_the_starting_client_disconnecting():
    """With SSE the client that kicked off the download dropping is the common
    case; the joiners still waiting must not lose the download with it."""
    finished = False

    async def runner():
        nonlocal finished
        await asyncio.sleep(0.1)
        finished = True
        return "done"

    starter = asyncio.create_task(run_download_deduped("src-4", runner))
    await asyncio.sleep(0.01)
    joiner = asyncio.create_task(run_download_deduped("src-4", runner))
    await asyncio.sleep(0.01)

    starter.cancel()  # first client goes away
    with pytest.raises(asyncio.CancelledError):
        await starter

    assert await joiner == "done"
    assert finished


# ---- sync drivers ----------------------------------------------------

def test_concurrent_requests_share_one_download():
    asyncio.run(_test_concurrent_requests_share_one_download())

def test_sequential_requests_download_again():
    asyncio.run(_test_sequential_requests_download_again())

def test_different_sources_do_not_share():
    asyncio.run(_test_different_sources_do_not_share())

def test_failure_propagates_and_clears_the_slot():
    asyncio.run(_test_failure_propagates_and_clears_the_slot())

def test_download_survives_the_starting_client_disconnecting():
    asyncio.run(_test_download_survives_the_starting_client_disconnecting())
