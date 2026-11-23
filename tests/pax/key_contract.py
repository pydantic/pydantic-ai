"""Press a key on a pax surface and report what its public `run` did with it.

A case presses one key on a fresh surface. The outcome is one of three: the surface closes with a result, closes
cancelled, or stays open, asking for another key. Nothing here reads which handler ran, so a declared outcome pins
behaviour someone chose, not whether it is right. `tests/pax/key_surfaces.py` holds the cases.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import pytest
from anyio import to_thread
from termflow.tui import keys as termflow_keys
from termflow.tui.keys import Key
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.pager import Pager, PagerResult
from termflow.tui.textinput import TextInput, TextInputResult

Outcome = Literal['closes-with-result', 'closes-cancelled', 'stays']

KEYS = (Key.ENTER, Key.ESCAPE, ' ', Key.TAB, Key.UP, Key.DOWN, Key.PAGE_DOWN, 'x', 'ctrl-c')
"""The keys pressed on every surface: the ones a widget most often claims, plus a printable letter."""

PICK: Mapping[str, Outcome] = {
    Key.ENTER: 'closes-with-result',
    Key.ESCAPE: 'closes-cancelled',
    'ctrl-c': 'closes-cancelled',
}
"""A list that Enter picks from, and Esc or Ctrl-C closes; every other key moves, filters, or does nothing."""

CLOSE: Mapping[str, Outcome] = {Key.ESCAPE: 'closes-cancelled', 'ctrl-c': 'closes-cancelled'}
"""A surface only Esc and Ctrl-C close; Enter acts on the highlighted row and keeps it open."""

WidgetT = TypeVar('WidgetT')
Keys = Callable[[], str]


class _Shown(BaseException):
    """The site showed its widget. A `BaseException`, so a site's `except Exception` cannot swallow it."""

    def __init__(self, widget: object) -> None:
        self.widget = widget


class _StillOpen(BaseException):
    """The surface asked for another key after the one pressed, so it stayed open."""


@dataclass(frozen=True, kw_only=True)
class Case:
    """One way a site shows its surface, and the outcome of each key there."""

    name: str
    press: Callable[[Path, str], Awaitable[Outcome]]
    outcomes: Mapping[str, Outcome]

    def expected(self) -> dict[str, Outcome]:
        return {key: self.outcomes.get(key, 'stays') for key in KEYS}


@dataclass(frozen=True, kw_only=True)
class Surface:
    """A registered construction site: what it shows, and its cases."""

    reason: str
    cases: tuple[Case, ...]


def keys_then_open(pressed: str) -> Callable[..., str]:
    """A key source that returns `pressed`, then raises `_StillOpen` when asked for another key."""
    sent = iter([pressed])

    def read(*args: object, **kwargs: object) -> str:
        if (key := next(sent, None)) is None:
            raise _StillOpen
        return key

    return read


@contextmanager
def _pressing(key: str) -> Generator[None]:
    """Answer every reader of termflow's `read_key`, wherever it was imported, with `key`, then `_StillOpen`."""
    real = termflow_keys.read_key
    with pytest.MonkeyPatch.context() as patch:
        for module in list(sys.modules.values()):
            if getattr(module, 'read_key', None) is real:
                patch.setattr(module, 'read_key', keys_then_open(key))
        yield


async def _run_site(show: Callable[[Path], object], tmp_path: Path) -> None:
    """Run `show` off the event loop, as menus run, then await what it returns when it is a coroutine."""
    result = await to_thread.run_sync(show, tmp_path)
    if inspect.isawaitable(result):
        await result


async def _shown(kind: type[WidgetT], show: Callable[[Path], object], tmp_path: Path) -> WidgetT:
    """The first `kind` that `show` runs. A termflow widget it runs before that one must come from its runners."""
    with pytest.MonkeyPatch.context() as patch:

        def capture(self: object, *args: object, **kwargs: object) -> object:
            raise _Shown(self)

        def unscripted(self: object, *args: object, **kwargs: object) -> object:
            raise AssertionError(f'{show} ran a {type(self).__name__} before its {kind.__name__}: script it')

        for widget in (Menu, TextInput, Pager):
            if widget is not kind:
                patch.setattr(widget, 'run', unscripted)
        patch.setattr(kind, 'run', capture)
        try:
            await _run_site(show, tmp_path)
        except _Shown as event:
            widget = event.widget
            assert isinstance(widget, kind)
            return widget
    raise AssertionError(f'{show} never ran a {kind.__name__}')


def _termflow_outcome(result: MenuResult | TextInputResult | PagerResult) -> Outcome:
    if isinstance(result, TextInputResult):
        return 'closes-cancelled' if result.value is None else 'closes-with-result'
    return 'closes-cancelled' if result.cancelled else 'closes-with-result'


def widget_case(
    name: str,
    show: Callable[[Path], object],
    outcomes: Mapping[str, Outcome],
    *,
    widget: type[Menu | TextInput | Pager] = Menu,
) -> Case:
    """A site that shows a termflow widget: `show` builds and runs the widget, or runs the site that does."""

    async def press(tmp_path: Path, key: str) -> Outcome:
        opened = await _shown(widget, show, tmp_path)
        with _pressing(key):
            try:
                return _termflow_outcome(await to_thread.run_sync(opened.run))
            except _StillOpen:
                return 'stays'

    return Case(name=name, press=press, outcomes=outcomes)


def reader_case(
    name: str,
    kind: type[WidgetT],
    show: Callable[[Path], object],
    drive: Callable[[WidgetT, Keys], object],
    outcomes: Mapping[str, Outcome],
    *,
    cancelled: Callable[[object], bool] = lambda result: result is None,
) -> Case:
    """A site that shows a pax widget with its own key loop: `drive` runs the widget `show` built with a key source."""

    async def press(tmp_path: Path, key: str) -> Outcome:
        opened = await _shown(kind, show, tmp_path)
        try:
            result = await to_thread.run_sync(drive, opened, keys_then_open(key))
        except _StillOpen:
            return 'stays'
        return 'closes-cancelled' if cancelled(result) else 'closes-with-result'

    return Case(name=name, press=press, outcomes=outcomes)


async def observed(case: Case, tmp_path: Path) -> dict[str, Outcome]:
    """Press each key on a fresh surface, each in its own directory so no key sees another's writes."""
    results: dict[str, Outcome] = {}
    for index, key in enumerate(KEYS):
        directory = tmp_path / f'key-{index}'
        directory.mkdir()
        results[key] = await case.press(directory, key)
    return results
