"""Fails the test that leaves a resource for someone else to clean up.

Registered by `tests/conftest.py`; see "Leaks" in `tests/AGENTS.md`.

A leak the garbage collector finalizes is reported in whichever test happens to collect it, if anywhere, and a leak
that something keeps alive is never reported at all. `tests/gc_isolation.py` already makes a test collect its own
garbage. This plugin fails the test for what that cannot see:

- An event loop it created on a thread other than the main one and never closed, such as the one `Agent.run_sync()`
  sets on a worker thread. While the thread lives it keeps the loop alive, so no `ResourceWarning` reports it, and
  once the thread exits the warning lands in whichever test is running then. Loops a `module`, `class`, `package`
  or `session`-scoped fixture creates are left to that fixture, and loops third-party code creates to that code. A
  loop left on the main thread is the one `run_sync()` reuses there; once dropped, the collector finalizes it in the
  test that dropped it.
"""

from __future__ import annotations as _annotations

import asyncio
import threading
import traceback
import weakref
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

PLUGIN_NAME = 'pydantic_ai_leak_guards'

_ROOT = Path(__file__).resolve().parent.parent
_ASYNCIO = Path(asyncio.__file__).resolve().parent


_CREATOR = '_pydantic_ai_leak_guards_creator'
"""The attribute that holds, on a loop the running test created, the frame that created it."""


@dataclass
class _State:
    test_running: bool = False
    shared_fixture_depth: int = 0
    generation: int = 0
    """Counts tests, so a loop finalized in a later test is not blamed on that one."""
    loops: weakref.WeakSet[asyncio.AbstractEventLoop] = field(
        default_factory=weakref.WeakSet[asyncio.AbstractEventLoop]
    )
    """Loops first-party code created off the main thread during the running test."""
    finalized_open: list[str] = field(default_factory=list[str])
    """Where each such loop that the collector finalized while it was still open was created."""


_state = _State()


def _first_party_creator() -> str | None:
    """The frame that created the loop being initialized, when it is this repository's own code."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        path = Path(frame.filename).resolve()
        if path.is_relative_to(_ASYNCIO) or path == Path(__file__).resolve():
            continue
        if not path.is_relative_to(_ROOT) or '.venv' in path.parts or 'site-packages' in path.parts:
            return None
        return f'{path.relative_to(_ROOT)}:{frame.lineno} in {frame.name}'
    return None  # pragma: no cover


_original_init = asyncio.BaseEventLoop.__init__


def _tracking_init(self: asyncio.BaseEventLoop, *args: Any, **kwargs: Any) -> None:
    _original_init(self, *args, **kwargs)
    on_worker_thread = threading.current_thread() is not threading.main_thread()
    if (
        on_worker_thread
        and _state.test_running
        and not _state.shared_fixture_depth
        and (creator := _first_party_creator())
    ):
        setattr(self, _CREATOR, (_state.generation, creator))
        _state.loops.add(self)


_original_del = asyncio.BaseEventLoop.__del__


def _tracking_del(self: asyncio.BaseEventLoop, *args: Any) -> None:
    # A thread that exits drops the loop it holds, so the loop can be finalized before its test reports.
    created: tuple[int, str] | None = getattr(self, _CREATOR, None)
    if created is not None and created[0] == _state.generation and not self.is_closed():
        _state.finalized_open.append(created[1])
    _original_del(self, *args)


# Patched once, at import, so a nested pytest session (`pytester`) that registers this module too cannot undo it.
setattr(asyncio.BaseEventLoop, '__init__', _tracking_init)
setattr(asyncio.BaseEventLoop, '__del__', _tracking_del)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[Any]) -> Generator[None, object, object]:
    shared = fixturedef.scope != 'function'
    _state.shared_fixture_depth += shared
    try:
        return (yield)
    finally:
        _state.shared_fixture_depth -= shared


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    _state.test_running = True
    _state.generation += 1
    try:
        return (yield)
    finally:
        _state.test_running = False
        _state.loops.clear()
        _state.finalized_open.clear()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if call.when != 'teardown':
        return report
    open_loops = [loop for loop in _state.loops if not loop.is_closed()]
    leaked = [*(getattr(loop, _CREATOR)[1] for loop in open_loops), *_state.finalized_open]
    for loop in open_loops:
        if not loop.is_running():  # pragma: no branch
            loop.close()
    if leaked and report.passed:
        report.outcome = 'failed'
        sites = ''.join(f'\n  {creator}' for creator in leaked)
        report.longrepr = (
            f'{item.nodeid} left {len(leaked)} event loop(s) open, created at:{sites}\n'
            'Close each loop where it is created, as `asyncio.get_event_loop().close()` on the thread that ran '
            '`run_sync()`; see "Leaks" in tests/AGENTS.md.'
        )
    return report
