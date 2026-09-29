"""Worker processes for the CPU-heavy endpoints.

Audio conversion and synthesis run in worker processes, not threads: Praat
holds the GIL for seconds at a time on long recordings, which would stall
the event loop (and ``/v1/health``) even from a worker thread. At most
``max_workers`` jobs run at once, one per process, which also bounds how
many audio buffers are in memory; up to ``max_queued`` further jobs wait
for a free worker, and requests beyond that are refused with 503
``server_busy`` rather than queued without limit.

Each worker process has its own pipe to the server, so a worker that dies
(for example, killed for using too much memory) fails only the job it was
running, with a 500. Jobs waiting for a worker are not affected, and a job
sent to a worker that died before taking it runs on another. Dead workers
are replaced when the next job needs one, and workers exit when the server
process does, however it ends.

A job that runs longer than ``job_timeout_s`` has its worker killed and
fails with a 504. When the caller passes ``is_disconnected`` and the client
goes away, a waiting job is dropped and the worker running a started one is
killed, so abandoned requests do not keep a worker busy.

Workers are started with the ``spawn`` method, which re-imports the main
script in each worker: a script that serves the app itself must guard its
entry point with ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.process
import signal
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar, cast

from . import _worker
from .errors import APIError

__all__ = ["BUSY_RETRY_AFTER_S", "JobRunner"]

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Retry-After sent with a 503 server_busy response, in seconds.
BUSY_RETRY_AFTER_S = 10

# How often a job is sent to a new worker when the one it was sent to died
# before taking it (a worker killed while idle, or one that fails to start).
_SEND_ATTEMPTS = 3

# Seconds a worker is given to exit by itself when the runner shuts down.
_STOP_TIMEOUT_S = 5.0

# How often, in seconds, a request waiting for its job checks whether the
# client is still connected.
_DISCONNECT_POLL_S = 0.5


class _RemoteTraceback(Exception):
    """The traceback of an exception raised in a worker process (its ``__cause__``)."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return f"\n\"\"\"\n{self.text}\"\"\""


class _WorkerExited(Exception):
    """The worker process running a job exited before it answered."""

    def __init__(self, exitcode: int | None) -> None:
        super().__init__(exitcode)
        self.exitcode = exitcode


class _JobTimedOut(Exception):
    """The job ran longer than the runner's ``job_timeout_s``; its worker was killed."""


class _JobAbandoned(Exception):
    """The client went away; the job's worker was killed."""


class _Job:
    """One job's link to the worker process running it, so it can be stopped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: _WorkerProcess | None = None
        self.abandoned = False

    def attach(self, worker: _WorkerProcess) -> bool:
        """Record that *worker* runs the job; ``False`` if it was abandoned already."""
        with self._lock:
            self._worker = worker
            return not self.abandoned

    def abandon(self) -> None:
        """Stop the job: kill its worker if one has started it."""
        with self._lock:
            self.abandoned = True
            worker = self._worker
        if worker is not None:
            worker.kill()


class _NoWorker(Exception):
    """No worker process would take the job."""


class _WorkerProcess:
    """One worker process running :func:`._worker.serve`, and the server's end of its pipe."""

    def __init__(self, context: Any) -> None:
        conn, child_conn = context.Pipe()
        self.process: multiprocessing.process.BaseProcess = context.Process(
            target=_worker.serve,
            args=(child_conn,),
            name="prosody-protocol-worker",
            # Stopped at exit by multiprocessing, if the runner was not shut down.
            daemon=True,
        )
        self.process.start()
        child_conn.close()  # So that recv() sees EOF when the worker dies.
        self.conn: multiprocessing.connection.Connection = conn

    @property
    def pid(self) -> int | None:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.is_alive()

    def exitcode(self) -> int | None:
        """The exit status of the worker, which has closed its pipe: it is ending."""
        self.process.join(timeout=1.0)
        return self.process.exitcode

    def kill(self) -> None:
        """Kill the worker at once; its pipe then reports EOF."""
        with contextlib.suppress(OSError, ValueError, AttributeError):
            self.process.kill()

    def stop(self) -> None:
        """Ask the worker to exit; kill it if it does not."""
        with contextlib.suppress(OSError, ValueError):
            self.conn.send(None)
        self.process.join(timeout=_STOP_TIMEOUT_S)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.close()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.conn.close()


class JobRunner:
    """Run functions of :mod:`._worker` in up to ``max_workers`` processes.

    Callers hold a place from :meth:`admit` while they prepare and run a
    job; at most ``max_workers + max_queued`` places are handed out, and
    ``admitted`` counts those held. ``running`` counts the jobs a worker
    process is working on. A worker that spends more than *job_timeout_s*
    seconds on one job (``None``: no limit) is killed.
    """

    def __init__(
        self, max_workers: int, max_queued: int = 8, *, job_timeout_s: float | None = None
    ) -> None:
        self.max_workers = max_workers
        self.max_queued = max_queued
        self.job_timeout_s = job_timeout_s
        self.admitted = 0
        self.running = 0
        self._lock = threading.Lock()
        # One thread per worker process: it hands the process a job and
        # waits for the answer. Jobs beyond max_workers wait in its queue.
        self._threads: ThreadPoolExecutor | None = None
        self._idle: list[_WorkerProcess] = []
        self._workers: set[_WorkerProcess] = set()
        self._context = multiprocessing.get_context("spawn")

    @contextlib.contextmanager
    def admit(self) -> Iterator[None]:
        """Hold a place for one job, or raise a 503 if all places are taken.

        Taken before the job's input is prepared (an upload saved to disk),
        so a refused request costs nothing more. A job whose request goes
        away while a worker runs it keeps its place until it ends.
        """
        with self._lock:
            if self.admitted >= self.max_workers + self.max_queued:
                raise APIError(
                    503,
                    "server_busy",
                    f"The server is busy with {self.admitted} audio conversions and "
                    f"syntheses; try again in {BUSY_RETRY_AFTER_S} seconds.",
                    headers={"Retry-After": str(BUSY_RETRY_AFTER_S)},
                )
            self.admitted += 1
        try:
            yield
        finally:
            self._release()

    async def run(
        self,
        func: Callable[..., T],
        *args: object,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> T:
        """Return ``func(*args)`` computed in a worker process.

        *func* and *args* must be picklable (module-level functions). SDK
        exceptions are re-raised here with their attributes intact. Call it
        inside :meth:`admit`. With *is_disconnected* (such as Starlette's
        ``Request.is_disconnected``), the job is stopped when the client
        goes away, and a 499 ``client_closed_request`` is raised.
        """
        job = _Job()
        future = self._get_threads().submit(self._execute, func, args, job)
        waiting = asyncio.wrap_future(future)
        try:
            if is_disconnected is not None:
                while not future.done():
                    await asyncio.wait({waiting}, timeout=_DISCONNECT_POLL_S)
                    if not future.done() and await is_disconnected():
                        logger.info("Client went away; stopping its job")
                        waiting.add_done_callback(_retrieve_exception)
                        self._abandon(future, job)
                        raise APIError(
                            499, "client_closed_request", "The client closed the request."
                        )
            return await waiting
        except asyncio.CancelledError:
            # The request task was cancelled: stop its job the same way.
            self._abandon(future, job)
            raise
        except _worker.JobError as exc:
            raise exc.rebuild() from None
        except _JobTimedOut:
            raise APIError(
                504,
                "job_timeout",
                f"The request took longer than {self.job_timeout_s:g} seconds "
                "(PP_JOB_TIMEOUT_S) and was stopped. Send shorter audio, or word timings "
                "instead of relying on the server's speech recognition.",
            ) from None
        except _WorkerExited as exc:
            raise APIError(
                500,
                "internal_error",
                "The worker process running this request exited unexpectedly "
                f"({_describe_exit(exc.exitcode)}; for example, it ran out of memory). "
                "Other requests were not affected.",
            ) from None
        except _NoWorker:
            raise APIError(
                500,
                "internal_error",
                "No worker process could be started for this request; the details are in "
                "the server log.",
            ) from None

    def _abandon(self, future: Future[Any], job: _Job) -> None:
        """Stop the job of a request that went away.

        A job still waiting is dropped. A running one has its worker killed;
        it holds a place until its thread notices, so the number of jobs in
        the server stays within the limit.
        """
        if future.cancel():
            return
        job.abandon()
        if not future.done():
            with self._lock:
                self.admitted += 1
            future.add_done_callback(lambda _: self._release())

    def worker_pids(self) -> list[int]:
        """The process IDs of the live worker processes."""
        with self._lock:
            workers = list(self._workers)
        return [w.pid for w in workers if w.pid is not None and w.alive()]

    def shutdown(self) -> None:
        """Stop the worker processes and wait for them; a later job starts new ones.

        Jobs not yet started are cancelled, and running ones are waited
        for. Waiting matters: uvicorn ends the process with the signal that
        stopped it, so no exit handler would get to stop the workers.
        """
        with self._lock:
            threads, self._threads = self._threads, None
        if threads is not None:
            threads.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
            self._idle.clear()
        for worker in workers:
            worker.stop()

    # -- In the runner's threads ------------------------------------------------

    def _execute(self, func: Callable[..., T], args: tuple[object, ...], job: _Job) -> T:
        """Run one job in a worker process; runs in one of the runner's threads."""
        for _ in range(_SEND_ATTEMPTS):
            worker = self._checkout()
            try:
                worker.conn.send((func, args))
            except (OSError, EOFError):
                self._discard(worker)  # It died while idle; the job never reached it.
                continue
            except BaseException:
                self._checkin(worker)  # Nothing was sent (e.g. an unpicklable argument).
                raise
            try:
                started = worker.conn.recv()
            except (OSError, EOFError):
                self._discard(worker)  # It died before taking the job: try another.
                continue
            if started != _worker.STARTED:  # pragma: no cover - protocol error
                self._discard(worker)
                raise RuntimeError(f"Unexpected message from a worker process: {started!r}")
            with self._lock:
                self.running += 1
            try:
                if not job.attach(worker):
                    worker.kill()  # Abandoned while it was being sent.
                return cast(T, self._answer(worker, job))
            finally:
                with self._lock:
                    self.running -= 1
        logger.error("Worker processes exited before taking a job %d times", _SEND_ATTEMPTS)
        raise _NoWorker

    def _answer(self, worker: _WorkerProcess, job: _Job) -> Any:
        """The result of the job *worker* has started, or its exception raised."""
        try:
            if self.job_timeout_s is not None and not worker.conn.poll(self.job_timeout_s):
                worker.kill()
                self._discard(worker)
                logger.warning("Worker process %s ran one job for over %g s; killed it",
                               worker.pid, self.job_timeout_s)
                raise _JobTimedOut
            status, payload = worker.conn.recv()
        except (OSError, EOFError):
            exitcode = self._discard(worker)
            if job.abandoned:
                raise _JobAbandoned from None
            logger.error("Worker process %s exited while running a job (%s)",
                         worker.pid, _describe_exit(exitcode))
            raise _WorkerExited(exitcode) from None
        except BaseException:
            self._checkin(worker)  # The answer arrived but could not be unpickled.
            raise
        self._checkin(worker)
        if status == _worker.OK:
            return payload
        if status == _worker.RAISED:
            exc, text = payload
            raise cast(BaseException, exc) from _RemoteTraceback(text)
        raise RuntimeError(f"A worker process could not send its answer back:\n{payload}")

    def _checkout(self) -> _WorkerProcess:
        """An idle live worker process, or a new one."""
        with self._lock:
            while self._idle:
                worker = self._idle.pop()
                if worker.alive():
                    return worker
                self._workers.discard(worker)
                worker.close()
        worker = _WorkerProcess(self._context)
        with self._lock:
            self._workers.add(worker)
        return worker

    def _checkin(self, worker: _WorkerProcess) -> None:
        with self._lock:
            if worker in self._workers:
                self._idle.append(worker)
                return
        worker.stop()  # The runner was shut down meanwhile.

    def _discard(self, worker: _WorkerProcess) -> int | None:
        """Forget a worker that has exited; return its exit status."""
        with self._lock:
            self._workers.discard(worker)
        exitcode = worker.exitcode()
        worker.close()
        return exitcode

    # -- Helpers ----------------------------------------------------------------

    def _release(self) -> None:
        with self._lock:
            self.admitted -= 1

    def _get_threads(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._threads is None:
                self._threads = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="prosody-protocol-job"
                )
            return self._threads


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
    """Mark the outcome of a future nobody awaits any more as seen."""
    if not future.cancelled():
        future.exception()


def _describe_exit(exitcode: int | None) -> str:
    if exitcode is None:
        return "exit status unknown"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        return f"killed by {name}"
    return f"exit status {exitcode}"
