"""Keep terminal ownership until a cancelled menu worker has restored its screen."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _NoTerminal:
    command: str
    instead: str | None


_STOP: ContextVar[Event | None] = ContextVar('menu_stop', default=None)
_HOLD: ContextVar[Hold] = ContextVar('menu_hold', default=_no_hold)
_NO_TERMINAL: ContextVar[_NoTerminal | None] = ContextVar('menu_no_terminal', default=None)


@contextmanager
def without_terminal(command: str, *, instead: str | None = None) -> Generator[None]:
    """Refuse every terminal menu and prompt `command` opens in this context: there is no terminal to show them.

    A menu would read EOF from stdin and close as if cancelled; the command fails with `require_terminal`'s
    error instead. `instead` names a command that does the same in the web UI.
    """
    token = _NO_TERMINAL.set(_NoTerminal(command=command, instead=instead))
    try:
        yield
    finally:
        _NO_TERMINAL.reset(token)


def require_terminal(*, instead: str | None = None) -> None:
    """Raise `ValueError` before a terminal menu or prompt opens under `without_terminal`.

    `instead` replaces the alternative `without_terminal` named, for a prompt that knows a better one.
    """
    refused = _NO_TERMINAL.get()
    if refused is None:
        return
    instead = instead or refused.instead
    raise ValueError(
        f'{refused.command} opens a terminal menu or prompt, and this CLAI has no terminal: '
        f'enter it in CLAI in a terminal{f", or use {instead} in this page" if instead else ""}.'
    )


@contextmanager
def holding_output(hold: Hold) -> Generator[None]:
    """Enter `hold` around every menu worker started in this context.

    The live editor sets this to its panel's hold for as long as it is open, so a menu
    gets the screen to itself and output, such as a streaming turn's, waits in the
    transcript until the menu closes. Between workers, what a command prints appears
    as it happens. A full-screen worker takes the panel off the screen, so it does not
    show what was printed before it either: a prompt that must stay under a printed
    login URL is `line_input.read_line`, which paints through the panel inline.
    """
    token = _HOLD.set(hold)
    try:
        yield
    finally:
        _HOLD.reset(token)


def full_terminal() -> AbstractContextManager[None]:
    """Take the live panel off the screen while another process draws on the whole terminal.

    The panel repaints every cell when the block exits. Outside the live editor this does nothing.
    """
    return _HOLD.get()(leave_screen=True)


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
    Under `without_terminal`, this raises `require_terminal`'s error instead of opening the menu.
    """
    require_terminal()
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


def _describe(span: telemetry.UISpan, result: object) -> None:
    """How the menu ended, from its result's shape alone: values can be secrets or prompt text."""
    if isinstance(result, MenuResult | TextInputResult):
        span.set('cancelled', result.cancelled)
    else:
        span.set('result', type(result).__name__)
