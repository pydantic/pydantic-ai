"""CLAI's existing brand colours, with opt-in Termflow palettes."""

# ruff: noqa: PLC0415 -- the splash must not import Termflow at startup.

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.syntax import SyntaxTheme
    from termflow.themes import TerminalPalette  # pyright: ignore[reportMissingTypeStubs]

_ACTIVE: ContextVar[Callable[[], str]] = ContextVar('clai_theme', default=lambda: 'default')


def names() -> tuple[str, ...]:
    """Offer the unchanged default appearance and Termflow's bundled palettes."""
    from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]

    return ('default', *PALETTES)


def current() -> TerminalPalette | None:
    """Read the session's palette; `None` keeps the existing CLAI appearance."""
    name = _ACTIVE.get()()
    if name == 'default':
        return None
    from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]

    return PALETTES[name]


def color(role: str) -> str:
    """Resolve a brand colour against the selected palette at render time."""
    palette = current()
    shade = role.removeprefix('bold ')
    if palette is None or shade not in _SLOTS:
        return role
    return ('bold ' if role.startswith('bold ') else '') + palette.ansi[_SLOTS[shade]]


def syntax_theme() -> SyntaxTheme:
    """Use native ANSI syntax colours, resolving selected palettes for previews too."""
    from rich.style import Style
    from rich.syntax import ANSI_LIGHT, ANSISyntaxTheme

    palette = current()
    styles = ANSI_LIGHT.copy()
    if palette is not None:
        for token, style in styles.items():
            color = style.color
            if color is not None:
                assert color.number is not None
                styles[token] = style + Style(color=palette.ansi[color.number])
    return ANSISyntaxTheme(styles)


def apply(name: str, *, output: IO[str]) -> None:
    """Apply a validated choice, or restore terminal defaults after a palette."""
    from termflow.themes import (  # pyright: ignore[reportMissingTypeStubs]
        PALETTES,
        apply_palette,  # pyright: ignore[reportUnknownVariableType] -- upstream also accepts an untyped dict.
        reset_palette,
    )

    if name == 'default':
        reset_palette(output=output)
    else:
        apply_palette(PALETTES[name], output=output, register_reset=False)


@contextmanager
def use(get_name: Callable[[], str], *, output: IO[str] | None = None) -> Generator[None]:
    """Scope colours to a shell, leaving the terminal untouched when no palette is chosen."""
    active = _ACTIVE
    token = active.set(get_name)
    try:
        if output is not None and get_name() != 'default':
            apply(get_name(), output=output)
        yield
    finally:
        if output is not None and get_name() != 'default':
            apply('default', output=output)
        active.reset(token)


LITHIUM = '#E520E9'
CALCIUM = '#FF6550'
PURPLE = '#9B77FF'
AQUA = '#77FFD8'
SUGAR = '#FBFFEA'
LIGHT_PURPLE = '#F0E0FD'
DARK_PURPLE = '#36182D'
ELEMENT_PURPLE = '#49353F'
GREY = '#8F888E'
AI_CYAN = '#00FFEB'
AI_YELLOW = '#D0FF71'

ACCENT = f'bold {LITHIUM}'
INFO = AI_CYAN
SUCCESS = AQUA
WARNING = AI_YELLOW
ERROR = CALCIUM
MUTED = GREY
THINKING = PURPLE
# The logo keeps Pydantic's brand colours under every palette; do not pass these through `color()`.
LOGO = f'bold {LITHIUM}'
BANNER = (LITHIUM, PURPLE, AI_CYAN)
DIFF_ADDITION = '#465258'
DIFF_DELETION = '#682B36'

_SLOTS = {
    LITHIUM: 12,
    CALCIUM: 1,
    PURPLE: 5,
    AQUA: 14,
    SUGAR: 7,
    LIGHT_PURPLE: 15,
    DARK_PURPLE: 0,
    ELEMENT_PURPLE: 8,
    GREY: 8,
    AI_CYAN: 6,
    AI_YELLOW: 3,
}
_BASIC = {
    LITHIUM: 95,
    CALCIUM: 91,
    PURPLE: 35,
    AQUA: 96,
    SUGAR: 97,
    LIGHT_PURPLE: 97,
    GREY: 90,
    AI_CYAN: 96,
    AI_YELLOW: 93,
}


def truecolor() -> bool:
    """Whether the terminal advertises 24-bit colour."""
    return os.getenv('COLORTERM', '').lower() in ('truecolor', '24bit')


def sgr(role: str, *, bold: bool = False) -> str:
    """Raw escape for surfaces that bypass Rich, with a 16-colour fallback."""
    resolved = color(role)
    if resolved.startswith('bold '):
        resolved = resolved.removeprefix('bold ')
        bold = True
    prefix = '1;' if bold else ''
    if truecolor():
        red, green, blue = (int(resolved[index : index + 2], 16) for index in (1, 3, 5))
        return f'\x1b[{prefix}38;2;{red};{green};{blue}m'
    palette = current()
    if palette is not None and resolved in palette.ansi:
        slot = palette.ansi.index(resolved)
        code = 30 + slot if slot < 8 else 90 + slot - 8
    else:
        code = _BASIC[resolved]
    return f'\x1b[{prefix}{code}m'
