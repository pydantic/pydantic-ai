"""Keep terminal ownership until a cancelled menu worker has restored its screen."""

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from threading import Event
from typing import TypeVar

import anyio
from termflow.tui.keys import read_key  # pyright: ignore[reportMissingTypeStubs]

ResultT = TypeVar('ResultT')
_STOP: ContextVar[Event | None] = ContextVar('menu_stop', default=None)


def menu_key() -> str:
    """Poll cancellation alongside terminal input."""
    stop = _STOP.get()
    if stop is not None and stop.is_set():
        return 'ctrl-c'
    return read_key(timeout=0.05)


async def run_worker(operation: Callable[[], ResultT]) -> ResultT:
    """Request menu exit on cancellation, then join before releasing terminal ownership."""
    stop = Event()
    token = _STOP.set(stop)
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
