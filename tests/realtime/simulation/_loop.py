"""A deterministic asyncio event loop for the realtime session simulator.

The simulator runs the real `RealtimeSession`, connection, and tool tasks on this loop, with nothing
underneath but in-memory fake transports. Two things make a run reproducible from its trace alone:

- **Virtual time.** `time()` is a counter the simulation advances explicitly, so every timer (a
  reconnect backoff, GPT-Live's turn-silence clock, a handshake timeout) fires at a point the trace
  names rather than whenever the wall clock gets there.
- **Explicit progress.** The simulation drives the loop itself, a bounded number of iterations at a
  time (`run_ticks`) or until nothing is runnable (`run_until_idle`). Asyncio runs ready callbacks in
  FIFO order, so the same sequence of steps always interleaves the tasks the same way; which
  interleavings get explored is decided by how many ticks the trace lets pass between steps.
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar('T')

_MAX_IDLE_TICKS = 20_000
"""How many loop iterations `run_until_idle` allows before calling the system livelocked."""


_THREAD_WAIT_SECONDS = 0.05
_MAX_THREAD_WAITS = 200
"""How long (in real time) an idle loop waits for a worker thread before calling the simulation stuck."""


class SimulationStuck(AssertionError):
    """The loop kept finding work to do: something in the session is spinning without making progress."""


class _NonBlockingSelector:
    """Wraps the loop's selector so an idle loop jumps its virtual clock instead of sleeping for real.

    The simulation's own drivers keep a callback ready at all times, so while a trace is being stepped
    the loop never idles and virtual time only moves when a step says so. Anything else awaited on the
    loop (tearing a simulation down, a handoff run) would otherwise block on the *real* clock for as
    long as the next virtual timer is away.
    """

    def __init__(self, selector: Any, loop: SimulatedLoop) -> None:
        self._selector = selector
        self._loop = loop

    def select(self, timeout: float | None = None) -> Any:
        if timeout is None:
            # Nothing to run and no timer: only another thread (a worker running sync code) can wake the
            # loop now, so wait for it on the real clock, but not forever.
            for _ in range(_MAX_THREAD_WAITS):
                if events := self._selector.select(_THREAD_WAIT_SECONDS):
                    return events
            raise SimulationStuck('the event loop has nothing to run and nothing that could wake it')
        if timeout > 0:
            self._loop.advance_clock(timeout)
        return self._selector.select(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._selector, name)


class SimulatedLoop(asyncio.SelectorEventLoop):
    """A selector event loop whose clock only moves when the simulation says so."""

    def __init__(self) -> None:
        super().__init__()
        self._virtual_time = 0.0
        self._selector = _NonBlockingSelector(self._selector, self)  # pyright: ignore[reportAttributeAccessIssue]

    def time(self) -> float:
        return self._virtual_time

    def _run_once(self) -> None:
        # Asyncio runs a timer as soon as it is within the clock's resolution of now. On a real clock, time
        # has moved on by the time the callback reads it; on this one it hasn't, and code that re-arms a timer
        # for the sliver of time still left (GPT-Live's turn clock does) would spin forever. So a timer that
        # runs early moves the clock to its deadline, which is where a real clock would be.
        scheduled = self._scheduled  # pyright: ignore[reportAttributeAccessIssue]
        if scheduled and not (head := scheduled[0]).cancelled():
            when = head.when()
            if self._virtual_time < when <= self._virtual_time + self._clock_resolution:  # pyright: ignore[reportAttributeAccessIssue]
                self._virtual_time = when
        super()._run_once()  # pyright: ignore[reportAttributeAccessIssue]

    def advance_clock(self, seconds: float) -> None:
        """Move the virtual clock forward; timers that fall due run on the next iterations."""
        assert seconds >= 0
        self._virtual_time += seconds

    def has_runnable_work(self) -> bool:
        """Whether any callback is ready, or any timer is due at the current virtual time."""
        if self._ready:  # pyright: ignore[reportAttributeAccessIssue]
            return True
        due = self.time() + self._clock_resolution  # pyright: ignore[reportAttributeAccessIssue]
        return any(
            not handle.cancelled() and handle.when() <= due
            for handle in self._scheduled  # pyright: ignore[reportAttributeAccessIssue]
        )

    def next_timer(self) -> float | None:
        """When the next pending timer is due, if any."""
        whens = [
            handle.when()
            for handle in self._scheduled  # pyright: ignore[reportAttributeAccessIssue]
            if not handle.cancelled()
        ]
        return min(whens) if whens else None

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a driver coroutine to completion on this loop."""
        return self.run_until_complete(coro)

    def run_ticks(self, ticks: int) -> None:
        """Let exactly `ticks` loop iterations pass."""

        async def tick() -> None:
            for _ in range(ticks):
                await asyncio.sleep(0)

        self.run(tick())

    def run_until_idle(self) -> None:
        """Iterate until no callback is ready and no timer is due."""

        async def settle() -> None:
            for _ in range(_MAX_IDLE_TICKS):
                await asyncio.sleep(0)
                if not self.has_runnable_work():
                    return
            raise SimulationStuck(f'the event loop was still busy after {_MAX_IDLE_TICKS} iterations')

        self.run(settle())
