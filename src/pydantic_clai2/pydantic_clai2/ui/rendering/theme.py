"""CLAI's existing brand colours, with opt-in Termflow palettes."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, TYPE_CHECKING

from pydantic_clai2.config.theme_names import names as theme_names

if TYPE_CHECKING:
    from rich.syntax import SyntaxTheme
    from termflow.diff import DiffRenderer, DiffTheme
    from termflow.themes import TerminalPalette

_ACTIVE: ContextVar[Callable[[], str]] = ContextVar('clai_theme', default=lambda: 'default')
_BRANDED: ContextVar[bool] = ContextVar('clai_branded', default=False)


def names() -> tuple[str, ...]:
    """Use the same theme choices as settings validation."""
    return theme_names()


def name() -> str:
    """The session's theme name, read at render time so a change applies to the next frame."""
    return _ACTIVE.get()()


@contextmanager
def branded() -> Generator[None]:
    """Mark output in Pydantic's brand colours, which keeps them when a theme change repaints the transcript."""
    token = _BRANDED.set(True)
    try:
        yield
    finally:
        _BRANDED.reset(token)


def painted_in() -> str | None:
    """The theme output written now is painted in, or `None` for branding that no theme recolours."""
    return None if _BRANDED.get() else name()


def current() -> TerminalPalette | None:
    """Read the session's palette; `None` keeps the existing CLAI appearance."""
    name = _ACTIVE.get()()
    if name == 'default':
        return None
    from termflow.themes import PALETTES

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


_DIFF_TINT = 0.28
"""How far a palette's diff backgrounds move from its background towards its green and red."""


def diff_theme() -> DiffTheme:
    """Diff line colours: CLAI's own by default, else the palette's green and red over its background."""
    from termflow.diff import DiffTheme
    from termflow.themes.palette import blend_hex

    palette = current()
    if palette is None:
        return DiffTheme(addition=DIFF_ADDITION, deletion=DIFF_DELETION, marker_brighten=2.0)
    return DiffTheme(
        addition=blend_hex(palette.bg, palette.ansi[2], _DIFF_TINT),
        deletion=blend_hex(palette.bg, palette.ansi[1], _DIFF_TINT),
        # Markers stand out from their line: brighter on a dark background, darker on a light one.
        marker_brighten=-0.5 if _is_light(palette) else 2.0,
    )


def diff_renderer() -> DiffRenderer:
    """Render diffs in `diff_theme()` colours, with code that stays readable on light palettes."""
    from termflow.diff import DiffRenderer
    from termflow.syntax import Highlighter

    palette = current()
    # Monokai's near-white text vanishes on a light line; `default` leaves names in the palette's foreground.
    highlighter = Highlighter(style='default') if palette is not None and _is_light(palette) else None
    return DiffRenderer(highlighter=highlighter, theme=diff_theme())


def _is_light(palette: TerminalPalette) -> bool:
    red, green, blue = (int(palette.bg[index : index + 2], 16) for index in (1, 3, 5))
    return 0.299 * red + 0.587 * green + 0.114 * blue > 128


def roles(name: str) -> tuple[tuple[str, str], ...]:
    """The Rich colour `name` paints each role with; where roles share a colour, the earlier one claims it.

    Roles are the background and foreground, diff lines and markers, CLAI's brand colours, and the
    16 ANSI slots. The default theme leaves the background, foreground, and slots to the terminal.
    Output painted in one theme repaints in another by translating its colours role by role.
    """
    with use(lambda: name):
        palette = current()
        diff = diff_theme()
        brands = tuple((f'brand {brand}', color(brand)) for brand in _SLOTS)
    surface = (palette.bg, palette.fg) if palette is not None else ('default', 'default')
    return (
        ('background', surface[0]),
        ('foreground', surface[1]),
        ('diff addition', diff.addition),
        ('diff deletion', diff.deletion),
        ('diff addition marker', diff.addition_marker),
        ('diff deletion marker', diff.deletion_marker),
        *brands,
        *((f'ansi {slot}', palette.ansi[slot] if palette is not None else f'color({slot})') for slot in range(16)),
    )


def apply(name: str, *, output: IO[str]) -> None:
    """Apply a validated choice, or restore terminal defaults after a palette."""
    from termflow.themes import (
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
# Claude Code's dark-theme diff backgrounds: a green and a red at the same depth, so additions read as additions.
DIFF_ADDITION = '#225C2B'
DIFF_DELETION = '#7A2936'

# `roles` gives a shared slot to the first brand here: muted grey text is commoner than element purple.
_SLOTS = {
    LITHIUM: 12,
    CALCIUM: 1,
    PURPLE: 5,
    AQUA: 14,
    SUGAR: 7,
    LIGHT_PURPLE: 15,
    DARK_PURPLE: 0,
    GREY: 8,
    ELEMENT_PURPLE: 8,
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
