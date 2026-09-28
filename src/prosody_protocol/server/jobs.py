"""Worker processes for the CPU-heavy endpoints.

Audio conversion and synthesis run in a pool of worker processes, not
threads: Praat holds the GIL for seconds at a time on long recordings, which
would stall the event loop (and ``/v1/health``) even from a worker thread.
The pool size bounds how many jobs run, and how many audio buffers are in
memory, at once; up to ``max_queued`` further jobs wait for a free worker,
and requests beyond that are refused with 503 ``server_busy`` rather than
queued without limit. A worker that dies (e.g. killed for memory) is
replaced on the next request, and workers exit when the server process
does, however it ends.

Workers are started with the ``spawn`` method, which re-imports the main
script in each worker: a script that serves the app itself must guard its
entry point with ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import multiprocessing
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TypeVar

from . import _worker
from .errors import APIError

T = TypeVar("T")

# Retry-After sent with a 503 server_busy response, in seconds.
BUSY_RETRY_AFTER_S = 10


class JobRunner:
    """Run functions of :mod:`._worker` in up to ``max_workers`` processes.

    Callers hold a place from :meth:`admit` while they prepare and run a
    job; at most ``max_workers + max_queued`` places are handed out, and
    ``admitted`` counts those held.
    """

    def __init__(self, max_workers: int, max_queued: int = 8) -> None:
        self.max_workers = max_workers
        self.max_queued = max_queued
        self.admitted = 0
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def admit(self) -> Iterator[None]:
        """Hold a place for one job, or raise a 503 if all places are taken.

        Taken before the job's input is prepared (an upload saved to disk),
        so a refused request costs nothing more. Not thread-safe: call it
        from the event loop only.
        """
        if self.admitted >= self.max_workers + self.max_queued:
            raise APIError(
                503,
                "server_busy",
                f"The server is busy with {self.admitted} audio conversions and syntheses; "
                f"try again in {BUSY_RETRY_AFTER_S} seconds.",
                headers={"Retry-After": str(BUSY_RETRY_AFTER_S)},
            )
        self.admitted += 1
        try:
            yield
        finally:
            self.admitted -= 1

    async def run(self, func: Callable[..., T], *args: object) -> T:
        """Return ``func(*args)`` computed in a worker process.

        *func* and *args* must be picklable (module-level functions). SDK
        exceptions are re-raised here with their attributes intact. Call it
        inside :meth:`admit`.
        """
        pool = self._get_pool()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(pool, functools.partial(_worker.call, func, *args))
        except _worker.JobError as exc:
            raise exc.rebuild() from None
        except BrokenProcessPool as exc:
            self._discard(pool)
            raise APIError(
                500,
                "internal_error",
                "The worker process handling this request exited unexpectedly "
                "(for example, it ran out of memory).",
            ) from exc

    def shutdown(self) -> None:
        """Stop the worker processes and wait for them; a later job starts new ones.

        Jobs not yet started are cancelled. Waiting matters: uvicorn ends the
        process with the signal that stopped it, so no exit handler would
        get to stop the workers.
        """
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _get_pool(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_worker.exit_with_parent,
                )
            return self._pool

    def _discard(self, pool: ProcessPoolExecutor) -> None:
        with self._lock:
            if self._pool is pool:
                self._pool = None
        pool.shutdown(wait=False, cancel_futures=True)
