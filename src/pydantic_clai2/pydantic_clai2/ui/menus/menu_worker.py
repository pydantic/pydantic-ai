"""Keep terminal ownership until a cancelled menu worker has restored its screen."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from threading import Event
from typing import Protocol, TypeVar

import anyio
from termflow.tui.keys import read_key
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_clai2.ui import telemetry

ResultT = TypeVar('ResultT')


class Hold(Protocol):
    """Pause the live panel while a widget owns the terminal."""

    def __call__(self, *, leave_screen: bool) -> AbstractContextManager[None]: ...


def _no_hold(*, leave_screen: bool) -> AbstractContextManager[None]:
    return nullcontext()


_STOP: ContextVar[Event | None] = ContextVar('menu_stop', default=None)
_HOLD: ContextVar[Hold] = ContextVar('menu_hold', default=_no_hold)


@contextmanager
def holding_output(hold: Hold) -> Generator[None]:
    """Enter `hold` around every menu worker started in this context.

    The live editor sets this to its panel's hold for as long as it is open, so a menu
    gets the screen to itself and output, such as a streaming turn's, waits in the
    transcript until the menu closes. Only the widget is held: anything the command
    prints outside `run_worker`, such as a login URL, still appears as it happens.
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


async def run_worker(operation: Callable[[], ResultT], *, inline: bool = False) -> ResultT:
    """Request menu exit on cancellation, then join before releasing terminal ownership.

    Every menu opens here, so this is where UI telemetry times it: a `menu {menu}` span named after
    `operation`, noting whether the user cancelled it, but never what they picked or typed.
    `inline` keeps the live panel on screen for a widget that paints through it, like `ask_user`'s.
    """
    stop = Event()
    token = _STOP.set(stop)
    try:
        with (
            _HOLD.get()(leave_screen=not inline),
            telemetry.span('menu {menu}', menu=telemetry.operation_name(operation)) as span,
        ):
            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                stop.set()
                span.set('closed_by', 'owner')
                with anyio.CancelScope(shield=True):
                    await asyncio.shield(task)
                raise
            _describe(span, result)
            return result
    finally:
        _STOP.reset(token)


def _describe(span: telemetry.UiSpan, result: object) -> None:
    """How the menu ended, from its result's shape alone: values can be secrets or prompt text."""
    if isinstance(result, MenuResult | TextInputResult):
        span.set('cancelled', result.cancelled)
    else:
        span.set('result', type(result).__name__)
