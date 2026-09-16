"""SIGINT cancels operations without cancelling the interactive application."""

import asyncio
import signal

import pytest

from pydantic_clai2.interrupts import Interrupts


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_worker_thread_does_not_install_signals() -> None:
    completed: list[bool] = []

    async def operation() -> None:
        completed.append(True)

    def worker() -> bool:
        return asyncio.run(Interrupts().run(operation()))

    original = signal.getsignal(signal.SIGINT)
    assert await asyncio.to_thread(worker)
    assert completed == [True]
    assert signal.getsignal(signal.SIGINT) == original


@pytest.mark.parametrize('double', [False, True])
async def test_interrupt_cleans_up_and_preserves_parent(double: bool) -> None:
    interrupts = Interrupts()
    cleaned = asyncio.Event()
    original = signal.getsignal(signal.SIGINT)

    async def operation() -> None:
        try:
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        finally:
            if double:
                signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0)
            cleaned.set()

    assert not await interrupts.run(operation())
    assert cleaned.is_set()
    assert interrupts.exit_requested == double
    assert signal.getsignal(signal.SIGINT) == original
    current = asyncio.current_task()
    assert current is not None and current.cancelling() == 0

    async def next_operation() -> None:
        return

    assert await interrupts.run(next_operation())


async def test_external_cancellation_propagates() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def operation() -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task = asyncio.create_task(Interrupts().run(operation()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


def test_double_press_window() -> None:
    now = 0.0
    interrupts = Interrupts(clock=lambda: now)
    assert not interrupts.press()
    now = 3.0
    assert not interrupts.press()
    now = 4.0
    assert interrupts.press()
