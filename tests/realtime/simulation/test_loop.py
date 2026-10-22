"""The simulator's virtual-time event loop."""

from __future__ import annotations as _annotations

import asyncio
import time
from collections.abc import Iterator

import pytest

from . import _loop
from ._loop import SimulatedLoop, SimulationStuck


@pytest.fixture
def loop() -> Iterator[SimulatedLoop]:
    loop = SimulatedLoop()
    yield loop
    loop.close()


def test_sleeping_moves_the_virtual_clock(loop: SimulatedLoop) -> None:
    loop.run(asyncio.sleep(30))
    assert loop.time() == pytest.approx(30)


def test_a_timer_due_within_the_clock_resolution_moves_the_clock_to_it(loop: SimulatedLoop) -> None:
    fired: list[float] = []
    # Asyncio treats a timer within the monotonic clock's resolution of now as due.
    when = loop.time() + time.get_clock_info('monotonic').resolution / 2
    loop.call_at(when, lambda: fired.append(loop.time()))
    loop.run_until_idle()
    assert fired == [when]


def test_run_ticks_leaves_tasks_mid_flight(loop: SimulatedLoop) -> None:
    progress: list[int] = []

    async def count() -> None:
        for i in range(10):
            progress.append(i)
            await asyncio.sleep(0)

    task = loop.create_task(count())
    loop.run_ticks(3)
    assert 0 < len(progress) < 10
    loop.run_until_idle()
    assert task.done()


def test_the_loop_waits_on_a_worker_thread(loop: SimulatedLoop) -> None:
    loop.run(asyncio.to_thread(time.sleep, 0.05))
    assert loop.time() == 0


def test_a_loop_nothing_can_wake_is_stuck(loop: SimulatedLoop, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_loop, '_MAX_THREAD_WAITS', 1)
    with pytest.raises(SimulationStuck, match='nothing that could wake it'):
        loop.run_until_complete(loop.create_future())


def test_a_loop_that_never_idles_is_stuck(loop: SimulatedLoop) -> None:
    async def spin() -> None:
        while True:
            await asyncio.sleep(0)

    task = loop.create_task(spin())
    with pytest.raises(SimulationStuck, match='still busy'):
        loop.run_until_idle()
    task.cancel()
    loop.run_ticks(1)
    assert task.cancelled()
