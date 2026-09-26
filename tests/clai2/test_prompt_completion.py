"""Completion worker cancellation, queue bounds, and process-safe thread lifetime."""

import asyncio
from contextvars import ContextVar
from threading import Event, Thread, current_thread

import anyio
import pytest
from anyio.to_thread import run_sync
from termflow.tui.completion import Completion  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.prompt_completion import CompletionWorker


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_worker_is_daemon_preserves_context_and_can_report_errors() -> None:
    marker = ContextVar('completion-test', default='default')
    marker.set('context value')
    worker = CompletionWorker()

    def operation() -> list[Completion]:
        assert current_thread().daemon
        return [Completion(marker.get())]

    def broken() -> list[Completion]:
        raise ValueError('provider failure')

    def stopped() -> list[Completion]:
        raise StopIteration

    try:
        assert (await worker.run(operation))[0].text == 'context value'
        with pytest.raises(ValueError, match='provider failure'):
            await worker.run(broken)
        with pytest.raises(RuntimeError, match='StopIteration'):
            await worker.run(stopped)
        assert (await worker.run(operation))[0].text == 'context value'
    finally:
        worker.close()
    worker.close()
    with pytest.raises(RuntimeError, match='closed'):
        await worker.run(operation)


async def test_cancelled_result_is_discarded_and_only_latest_queued_request_runs() -> None:
    worker = CompletionWorker()
    started, queued = anyio.Event(), anyio.Event()
    release, finished = Event(), Event()
    loop = asyncio.get_running_loop()
    executed: list[str] = []

    def blocked() -> list[Completion]:
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(5)
            return [Completion('old')]
        finally:
            finished.set()

    async def submit() -> list[Completion]:
        queued.set()
        return await worker.run(lambda: executed.append('queued') or [Completion('queued')])

    try:
        first = asyncio.create_task(worker.run(blocked))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(submit())
        await queued.wait()
        latest = asyncio.create_task(worker.run(lambda: executed.append('latest') or [Completion('latest')]))
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert (await latest)[0].text == 'latest'
        assert executed == ['latest']
    finally:
        release.set()
        assert await run_sync(finished.wait, 5)
        worker.close()


async def test_thread_start_failure_does_not_poison_later_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = CompletionWorker()

    def fail(self: Thread) -> None:
        raise RuntimeError('cannot start thread')

    with monkeypatch.context() as patch:
        patch.setattr(Thread, 'start', fail)
        with pytest.raises(RuntimeError, match='cannot start thread'):
            await worker.run(lambda: [])
    try:
        assert await worker.run(lambda: []) == []
    finally:
        worker.close()


def test_unused_worker_can_close_without_a_thread() -> None:
    worker = CompletionWorker()
    worker.close()
    worker.close()
