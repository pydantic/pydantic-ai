"""SIGINT cancels operations without cancelling the interactive application."""

import asyncio
import signal
import sys
import threading

import anyio
import pytest

from pydantic_clai2.ui.prompt.interrupts import Interrupts


def wakeup_fd() -> int:
    """The process's signal wakeup fd, which only setting it reads."""
    fd = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(fd, warn_on_full_buffer=False)
    return fd


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


async def test_overlapping_runs_give_ctrl_c_back_whichever_ends_first() -> None:
    """Three runs take Ctrl-C in turn and the first ends first: Ctrl-C ends where it started, not with a finished run."""
    original = signal.getsignal(signal.SIGINT)
    original_wakeup_fd = wakeup_fd()
    interrupts = Interrupts()
    started = [asyncio.Event() for _ in range(3)]
    finish = [asyncio.Event() for _ in range(3)]

    async def operation(index: int) -> None:
        started[index].set()
        await finish[index].wait()

    async def run(index: int) -> bool:
        if index:
            await started[index - 1].wait()
        return await interrupts.run(operation(index))

    runs = [asyncio.create_task(run(index)) for index in range(3)]
    await started[2].wait()
    for index in (0, 2, 1):
        # A run still holding Ctrl-C keeps the loop waking for it.
        assert wakeup_fd() != -1
        finish[index].set()
        assert await runs[index]
    assert signal.getsignal(signal.SIGINT) is original
    assert wakeup_fd() == original_wakeup_fd


async def test_a_replaced_run_takes_ctrl_c_again_on_focus() -> None:
    """The terminal moves back to a session whose Ctrl-C another took meanwhile: each gives it back once."""
    original = signal.getsignal(signal.SIGINT)
    first, second = Interrupts(), Interrupts()
    started = [asyncio.Event(), asyncio.Event()]
    finish = [asyncio.Event(), asyncio.Event()]

    async def operation(index: int) -> None:
        started[index].set()
        await finish[index].wait()

    one = asyncio.create_task(first.run(operation(0)))
    await started[0].wait()
    two = asyncio.create_task(second.run(operation(1)))
    await started[1].wait()
    first.focus(signals=True)
    finish[0].set()
    assert await one
    finish[1].set()
    assert await two
    assert signal.getsignal(signal.SIGINT) is original


@pytest.mark.parametrize('double', [False, True])
async def test_interrupt_cleans_up_and_preserves_parent(double: bool) -> None:
    interrupts = Interrupts()
    cleaned = asyncio.Event()
    original = signal.getsignal(signal.SIGINT)
    original_wakeup_fd = wakeup_fd()

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
    assert wakeup_fd() == original_wakeup_fd
    current = asyncio.current_task()
    assert current is not None
    assert current.cancelling() == 0

    async def next_operation() -> None:
        return

    assert await interrupts.run(next_operation())


async def test_signal_cancels_from_the_loop() -> None:
    """Ctrl-C can land inside an asyncio callback, such as `asyncio.shield` settling its result, which must finish first."""

    async def operation() -> None:
        current = asyncio.current_task()
        assert current is not None
        signal.raise_signal(signal.SIGINT)
        assert current.cancelling() == 0
        await asyncio.Event().wait()

    assert not await Interrupts().run(operation())


@pytest.mark.skipif(sys.platform == 'win32', reason='`signal.pthread_kill` is POSIX-only')
async def test_signal_wakes_an_idle_loop() -> None:
    """Ctrl-C must wake a loop with nothing else to do, as while a quiet `!command` runs.

    Python runs a signal handler on the main thread between bytecodes, so a signal that does not
    interrupt the selector, such as one handled on another thread or just before the loop blocks,
    waits for the loop's next event.
    """
    idle = threading.Event()

    def send() -> None:
        idle.wait()
        # This thread, not the main one, handles it, as the kernel may choose: the selector goes on waiting.
        signal.pthread_kill(threading.get_ident(), signal.SIGINT)

    sender = threading.Thread(target=send)
    loop = asyncio.get_running_loop()
    next_event = asyncio.Event()

    async def operation() -> None:
        sender.start()
        loop.call_soon(idle.set)
        await asyncio.Event().wait()

    timer = loop.call_later(5, next_event.set)
    try:
        assert not await Interrupts().run(operation())
    finally:
        timer.cancel()
        sender.join()
    assert not next_event.is_set()


@pytest.mark.skipif(sys.platform == 'win32', reason='the proactor loop has no `add_signal_handler`')
@pytest.mark.parametrize('added', ['before', 'during'])
async def test_asyncio_signal_handling_keeps_working(added: str) -> None:
    """`add_signal_handler` sets a wakeup fd of its own, which a run neither replaces nor clears."""
    loop = asyncio.get_running_loop()
    received = asyncio.Event()

    async def operation() -> None:
        if added == 'during':
            loop.add_signal_handler(signal.SIGUSR1, received.set)
        signal.raise_signal(signal.SIGINT)
        await asyncio.Event().wait()

    if added == 'before':
        loop.add_signal_handler(signal.SIGUSR1, received.set)
    try:
        assert not await Interrupts().run(operation())
        signal.raise_signal(signal.SIGUSR1)
        with anyio.fail_after(5):
            await received.wait()
    finally:
        loop.remove_signal_handler(signal.SIGUSR1)


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


async def test_direct_cancellation_in_worker_thread() -> None:
    async def scenario() -> None:
        interrupts = Interrupts()
        cleaned = asyncio.Event()

        async def operation() -> None:
            try:
                assert interrupts.cancel()
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        assert not await interrupts.run(operation())
        assert cleaned.is_set()
        assert not interrupts.cancel()

    original = signal.getsignal(signal.SIGINT)
    await asyncio.to_thread(asyncio.run, scenario())
    assert signal.getsignal(signal.SIGINT) == original


async def test_escape_cancellation_does_not_arm_exit() -> None:
    interrupts = Interrupts(clock=lambda: 0.0)
    assert not interrupts.active

    async def operation() -> None:
        assert interrupts.active
        assert interrupts.cancel(exit_on_repeat=False)
        assert interrupts.cancel(exit_on_repeat=False)
        await asyncio.Event().wait()

    assert not await interrupts.run(operation())
    assert not interrupts.active
    assert not interrupts.exit_requested
    assert not interrupts.press()
