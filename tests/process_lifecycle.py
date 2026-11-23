"""Interrupt code that starts a process at each event loop tick of the start, and check that nothing it started survives.

`assert_interrupting_each_spawn_tick_stops_the_command` runs the code under test once per tick: run `n` cancels it, or
lands its deadline, after `n` ticks, so every point of the start is covered up to asyncio connecting the new process's
pipes. That last point is where cancelling asyncio's process creation part-way kills only the direct child and leaks
its pipes. Each run gets an event loop of its own, so moving that loop's clock to land a deadline touches nothing else.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
from asyncio import base_subprocess
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal

import pytest

from tests._timeouts import READINESS_WAIT_TIMEOUT

Interrupt = Literal['cancel', 'deadline']
"""`cancel` cancels the run's task. `deadline` moves the run's clock past the `timeout` it passes the code under test."""

TIMEOUT = 0.1
"""The `timeout` a `deadline` run passes. The clock moves past it, and stays well within a spawn's own grace period."""


def background_sleep(pid_file: Path) -> str:
    """A shell command that starts a background `sleep`, then writes its own PID and the sleep's to `pid_file`."""
    return f'sleep 3600 & echo $$ $! > {shlex.quote(str(pid_file))}; wait'


def assert_interrupting_each_spawn_tick_stops_the_command(
    start: Callable[[str, float | None], Awaitable[object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    interrupt: Interrupt,
    expected: type[BaseException],
    max_ticks: int = 500,
) -> None:
    """Interrupt `start(command, timeout)` after 0, 1, 2, ... ticks; fail if an interrupted run leaves a process running.

    `start` runs the shell `command` it is given, passing `timeout` (set only for `deadline`) to the code under test,
    and each run must raise `expected`. Connecting a new process's pipes waits until the interruption, and the last
    run interrupts there once the command has started its background `sleep`, so it has a grandchild to leak.
    """
    real_connect = base_subprocess.BaseSubprocessTransport._connect_pipes
    attempt: _Attempt | None = None

    async def connect_after_the_interruption(
        transport: base_subprocess.BaseSubprocessTransport, waiter: asyncio.Future[None] | None
    ) -> None:
        assert attempt is not None
        await attempt.hold(transport)
        await real_connect(transport, waiter)

    monkeypatch.setattr(base_subprocess.BaseSubprocessTransport, '_connect_pipes', connect_after_the_interruption)
    for ticks in range(max_ticks):  # pragma: no branch
        attempt = _Attempt(tmp_path / f'pids-{ticks}')
        with asyncio.Runner(loop_factory=_ShiftedClockLoop) as runner:
            last = runner.run(attempt.run(start, ticks, interrupt=interrupt, expected=expected))
        if last:
            return
    pytest.fail(f'the command never started within {max_ticks} ticks')  # pragma: no cover


class _ShiftedClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock can jump forward, which expires every deadline the jump passes."""

    shift = 0.0

    def time(self) -> float:
        return super().time() + self.shift


class _Attempt:
    """One run, interrupted after a given number of ticks."""

    def __init__(self, pid_file: Path) -> None:
        self.pid_file = pid_file
        self.groups: list[int] = []
        self.held = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def hold(self, transport: base_subprocess.BaseSubprocessTransport) -> None:
        pid = transport.get_pid()
        assert pid is not None, 'the process exists before its pipes are connected'
        # Each command starts a session of its own, so its PID is its process group.
        self.groups.append(pid)
        self.held.set()
        await self.interrupted.wait()

    async def run(
        self,
        start: Callable[[str, float | None], Awaitable[object]],
        ticks: int,
        *,
        interrupt: Interrupt,
        expected: type[BaseException],
    ) -> bool:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, _ShiftedClockLoop)
        task = asyncio.ensure_future(
            start(background_sleep(self.pid_file), TIMEOUT if interrupt == 'deadline' else None)
        )
        # real-wait: one scheduler tick per step of the sweep, which is what the sweep counts, not a wait on a condition
        for _ in range(ticks):
            if self.held.is_set():
                break
            await asyncio.sleep(0)
        last = self.held.is_set()
        if last:
            await self._wait_for_pids()
        if interrupt == 'cancel':
            task.cancel()
        else:
            loop.shift = 2 * TIMEOUT
        self.interrupted.set()
        with pytest.raises(expected):
            await task
        for group in self.groups:
            await _assert_group_gone(group)
        return last

    async def _wait_for_pids(self) -> None:
        async with asyncio.timeout(READINESS_WAIT_TIMEOUT):
            # real-wait: the command writes the file from another process, which nothing in this one observes
            while not (self.pid_file.exists() and self.pid_file.read_text(encoding='ascii').strip()):
                await asyncio.sleep(0.01)  # pragma: lax no cover


async def _assert_group_gone(group: int) -> None:
    """Fail unless every process in `group` exits; a zombie its parent has yet to reap counts as gone."""
    try:
        async with asyncio.timeout(READINESS_WAIT_TIMEOUT):
            # real-wait: nothing announces the exit of a process in another process's group
            while _live_members(group):
                await asyncio.sleep(0.01)  # pragma: lax no cover
    except TimeoutError:  # pragma: no cover - the failure this check exists for
        survivors = _live_members(group)
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
        pytest.fail(f'processes {survivors} of group {group} survived the interrupted run')


def _live_members(group: int) -> list[int]:
    listing = subprocess.run(['ps', '-A', '-o', 'pid=,pgid=,stat='], capture_output=True, text=True, check=True)
    members: list[int] = []
    for line in listing.stdout.splitlines():
        pid, pgid, stat = line.split()[:3]
        if int(pgid) == group and not stat.startswith('Z'):
            members.append(int(pid))
    return members
