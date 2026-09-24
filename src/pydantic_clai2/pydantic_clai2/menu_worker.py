"""Keep terminal ownership until a cancelled menu worker has restored its screen."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from threading import Event
from typing import TypeVar

import anyio
from termflow.tui.keys import read_key  # pyright: ignore[reportMissingTypeStubs]

ResultT = TypeVar('ResultT')
_STOP: ContextVar[Event | None] = ContextVar('menu_stop', default=None)
_HOLD: ContextVar[Callable[[], AbstractContextManager[None]]] = ContextVar('menu_hold', default=nullcontext)


@contextmanager
def holding_output(hold: Callable[[], AbstractContextManager[None]]) -> Generator[None]:
    """Enter `hold` around every menu worker started in this context.

    A menu opened while a turn streams sets this to the editor's output hold, so the
    run's output waits in order until the menu leaves the screen. Only the widget is
    held: anything the command prints outside `run_worker`, such as a login URL,
    still appears as it happens.
    """
    token = _HOLD.set(hold)
    try:
        yield
    finally:
        _HOLD.reset(token)


def worker_stopping() -> bool:
    """Whether the owner has requested that this menu worker release its resources."""
    stop = _STOP.get()
    return stop is not None and stop.is_set()


def menu_key() -> str:
    """Poll cancellation alongside terminal input."""
    if worker_stopping():
        return 'ctrl-c'
    return read_key(timeout=0.05)


async def run_worker(operation: Callable[[], ResultT]) -> ResultT:
    """Request menu exit on cancellation, then join before releasing terminal ownership."""
    stop = Event()
    token = _STOP.set(stop)
    try:
        with _HOLD.get()():
            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                stop.set()
                with anyio.CancelScope(shield=True):
                    await asyncio.shield(task)
                raise
    finally:
        _STOP.reset(token)
