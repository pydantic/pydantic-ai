"""`/spinner`: choose the working animation from a live preview, by name, or start a `spinners.json`."""

import math
import time
from collections.abc import Callable, Iterable

from pydantic import ValidationError
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker
from .spinners import MIN_INTERVAL, Spinner, Spinners, clamp_interval

USAGE = 'Usage: /spinner [NAME [SECONDS]] | /spinner init'
_TICK = 'spinner-tick'
"""Not a key: the menu ignores it and repaints, which is what animates the preview."""


def _step(interval: float, delta: float) -> float:
    """Snap to the `MIN_INTERVAL` grid, like Code Puppy's picker, so the bounds stay reachable."""
    return clamp_interval(round((interval + delta) / MIN_INTERVAL) * MIN_INTERVAL)


def spinner_preview(spinner: Spinner, *, interval: float, now: float) -> str:
    """The spinner as the prompt's working title shows it, at the picker's speed."""
    muted, accent, reset = theme.sgr(theme.MUTED), theme.sgr(theme.ACCENT), '\x1b[0m'
    frame = spinner.frames[int(now / interval) % len(spinner.frames)]
    lines = (
        f'{accent}{spinner.name}{reset}{muted} ({spinner.source}){reset}',
        spinner.description,
        '',
        f'{muted}─ Working {reset}{accent}{frame}{reset}{muted} ─{reset}',
        '',
        f'{muted}{len(spinner.frames)} frames at {interval:.2f}s per frame{reset}',
        f'{muted}-/+ or left/right: slower/faster{reset}',
    )
    return '\n'.join(lines)


class SpinnerPicker:
    """A searchable list with an animated preview; speed keys adjust the highlighted spinner."""

    def __init__(self, spinners: Spinners, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Snapshot the catalogue so the rows stay put while the menu is open."""
        self.catalogue = spinners.catalogue()
        self.current = spinners.active().name
        self.clock = clock
        self.speeds: dict[str, float] = {}
        """Speeds changed in this picker, by spinner name; saved only if the spinner is applied."""

    def interval(self, name: str) -> float:
        """The picker's speed for `name`, starting from the catalogue's."""
        return self.speeds.get(name, self.catalogue[name].interval)

    def nudge(self, delta: float) -> Callable[[Menu, MenuItem], None]:
        """A key handler that changes the highlighted spinner's speed and keeps the menu open."""

        def handler(_menu: Menu, item: MenuItem) -> None:
            name = str(item.value)
            self.speeds[name] = _step(self.interval(name), delta)

        return handler

    def preview(self, item: MenuItem) -> str:
        """The highlighted spinner, animated on the picker's clock."""
        name = str(item.value)
        return spinner_preview(self.catalogue[name], interval=self.interval(name), now=self.clock())

    def build(self) -> Menu:
        """The menu, with the current spinner highlighted."""
        names = list(self.catalogue)
        builder = (
            MenuBuilder('Select spinner')
            .style(markdown_style())
            .items([MenuItem(f'{name}{" (current)" if name == self.current else ""}', value=name) for name in names])
            .searchable()
            .initial_index(names.index(self.current))
            .list_width(30)
            .preview(self.preview)
            .footer_hint('type to filter - -/+ speed - Enter apply - Esc close')
            .key_source(lambda: menu_key() or _TICK)
        )
        for key, delta in (('-', MIN_INTERVAL), ('left', MIN_INTERVAL), ('+', -MIN_INTERVAL), ('=', -MIN_INTERVAL)):
            builder = builder.on_key(key, self.nudge(delta))
        return builder.on_key('right', self.nudge(-MIN_INTERVAL)).build()


def _seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{text!r} is not a number of seconds. {USAGE}')
    return seconds


async def spinner_command(
    context: CommandContext, spinners: Spinners, args: list[str], *, runners: Runners = TERMINAL
) -> str:
    """Apply a spinner by name or picker, saving a changed speed to `spinners.json`; cancelling changes nothing."""
    if args == ['init']:
        if spinners.init():
            return f'Wrote {spinners.path}. Edits show on the next frame.'
        return f'{spinners.path} already exists; edit it directly. Edits show on the next frame.'
    if len(args) > 2:
        raise ValueError(USAGE)
    if args:
        found = spinners.find(args[0])
        if found is None:
            raise ValueError(f'Unknown spinner {args[0]!r}. Choose from: {", ".join(spinners.catalogue())}')
        name, interval = found.name, _seconds(args[1]) if len(args) == 2 else None
    else:
        picker = SpinnerPicker(spinners)
        result = await run_worker(lambda: runners.run_list(picker.build()))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return ''
        name = result.item.value
        interval = picker.speeds.get(name)
    if interval is not None:
        try:
            spinners.save_interval(name, interval)
        except ValidationError as exc:
            raise ValueError(f'{spinners.path} is not a JSON object; speed not saved.') from exc
    context.set_setting(['display.spinner', name])
    spinner = spinners.active()
    saved = f' Speed saved to {spinners.path}.' if interval is not None else ''
    lines = (f'Spinner set to {spinner.name} ({len(spinner.frames)} frames at {spinner.interval:.2f}s).{saved}',)
    return '\n'.join((*lines, *spinners.problems()))


def spinner_completions(spinners: Spinners, args: list[str]) -> Iterable[str]:
    """Spinner names and `init` for the first argument."""
    return ('init', *spinners.catalogue()) if len(args) <= 1 else ()
