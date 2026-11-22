"""CLAI's brand colours, raised to read on a light terminal background, with opt-in Termflow palettes."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import cache
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.syntax import SyntaxTheme
    from termflow.diff import DiffRenderer, DiffTheme
    from termflow.syntax import Highlighter
    from termflow.themes import TerminalPalette

_ACTIVE: ContextVar[Callable[[], str]] = ContextVar('clai_theme', default=lambda: 'default')
_BRANDED: ContextVar[bool] = ContextVar('clai_branded', default=False)
_LIGHT_BACKGROUND: ContextVar[str | None] = ContextVar('clai_light_background', default=None)


def names() -> tuple[str, ...]:
    """Use the same theme choices as settings validation."""
    # Imported on use: the splash imports this module before it paints, and importing the `config`
    # package builds Pydantic models, whose first build loads every installed Pydantic plugin.
    from pydantic_ai_pax.config.theme_names import names as theme_names

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


def light_background() -> str | None:
    """The terminal's background when the default theme paints for a light one, else `None`."""
    return _LIGHT_BACKGROUND.get()


_LIGHT_SURFACE = 'github_light'
"""The bundled palette whose ANSI colours the default theme's code and diffs use on a light terminal."""


def _surface() -> TerminalPalette | None:
    """The palette code and diff colours come from: the selected one, or a light one on a light terminal."""
    palette = current()
    background = _LIGHT_BACKGROUND.get()
    if palette is not None or background is None:
        return palette
    from termflow.themes import PALETTES

    # Black text, so code is raised to 4.5:1 even where the palette's own text would read worse.
    return replace(PALETTES[_LIGHT_SURFACE], bg=background, fg='#000000')


def color(role: str) -> str:
    """Resolve a brand colour against the selected palette, or the light terminal background, at render time."""
    palette = current()
    background = _LIGHT_BACKGROUND.get()
    shade = role.removeprefix('bold ')
    if shade not in _SLOTS:
        return role
    if palette is not None:
        resolved = _readable(palette.ansi[_SLOTS[shade]], palette.bg, palette.fg)
    elif background is not None:
        resolved = _readable(shade, background, '#000000')
        if not truecolor():
            resolved = _nearest_eight_bit(resolved, background)
    else:
        return role
    return ('bold ' if role.startswith('bold ') else '') + resolved


_MIN_CONTRAST = 4.5
"""WCAG AA's contrast for text, which palette colours are raised to, short of the palette's own text."""


@cache
def _readable(colour: str, bg: str, fg: str) -> str:
    """`colour` moved towards black or white, keeping its hue, until it reads on `bg` as well as `fg` does."""
    from termflow.themes.palette import blend_hex

    floor = min(_MIN_CONTRAST, _contrast(fg, bg))
    end = max('#000000', '#FFFFFF', key=lambda extreme: _contrast(extreme, bg))
    return next(blended for step in range(21) if _contrast(blended := blend_hex(colour, end, step / 20), bg) >= floor)


@cache
def _nearest_eight_bit(colour: str, bg: str) -> str:
    """The 256-colour shade closest to `colour` that still reads on `bg`, since the nearest one may not."""
    from rich.color import Color

    shades = (Color.from_ansi(number).get_truecolor().hex.upper() for number in range(16, 256))
    return min(
        (shade for shade in shades if _contrast(shade, bg) >= _MIN_CONTRAST),
        key=lambda shade: sum((int(shade[i : i + 2], 16) - int(colour[i : i + 2], 16)) ** 2 for i in (1, 3, 5)),
    )


def _luminance(colour: str) -> float:
    """WCAG relative luminance."""
    channels = (int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5))
    red, green, blue = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(a: str, b: str) -> float:
    darker, lighter = sorted((_luminance(a), _luminance(b)))
    return (lighter + 0.05) / (darker + 0.05)


def _lightness(colour: str) -> float:
    """CIELAB L*, which red-green and blue-yellow colour blindness leave nearly unchanged."""
    luminance = _luminance(colour)
    return 116 * luminance ** (1 / 3) - 16 if luminance > 216 / 24389 else luminance * 24389 / 27


def syntax_theme() -> SyntaxTheme:
    """Use native ANSI syntax colours, resolving selected palettes for previews too."""
    from rich.style import Style
    from rich.syntax import ANSI_LIGHT, ANSISyntaxTheme

    palette = _surface()
    styles = ANSI_LIGHT.copy()
    if palette is not None:
        for token, style in styles.items():
            color = style.color
            if color is not None:
                assert color.number is not None
                shade = _readable(palette.ansi[color.number], palette.bg, palette.fg)
                if current() is None and not truecolor():  # The adaptive default on a light terminal.
                    shade = _nearest_eight_bit(shade, palette.bg)
                styles[token] = style + Style(color=shade)
    return ANSISyntaxTheme(styles)


_DIFF_TINT = 0.28
"""How far a palette's diff backgrounds move from its background towards its green and red, on average."""
_DIFF_SPLITS = (0.08, -0.08, 0.16, -0.16)
"""How much more the addition is tinted than the deletion is, tried in order until their lightness differs."""
_DIFF_LIGHTNESS = 12
"""The L* by which addition and deletion lines differ, so they stay apart without telling red from green."""


def diff_theme() -> DiffTheme:
    """Diff line colours: CLAI's own by default, else the palette's green and red over its background."""
    from termflow.diff import DiffTheme
    from termflow.themes.palette import blend_hex

    palette = _surface()
    if palette is None:
        return DiffTheme(addition=DIFF_ADDITION, deletion=DIFF_DELETION, marker_brighten=2.0)
    pairs = [
        (
            blend_hex(palette.bg, palette.ansi[2], _DIFF_TINT + split),
            blend_hex(palette.bg, palette.ansi[1], _DIFF_TINT - split),
        )
        for split in _DIFF_SPLITS
    ]
    addition, deletion = next(
        (pair for pair in pairs if abs(_lightness(pair[0]) - _lightness(pair[1])) >= _DIFF_LIGHTNESS), pairs[-1]
    )
    return DiffTheme(
        addition=addition,
        deletion=deletion,
        # Markers stand out from their line: brighter on a dark background, darker on a light one.
        marker_brighten=-0.5 if _is_light(palette) else 2.0,
    )


def diff_renderer() -> DiffRenderer:
    """Render diffs in `diff_theme()` colours, with code that stays readable on light palettes."""
    from termflow.diff import DiffRenderer
    from termflow.syntax import Highlighter

    palette = _surface()
    colours = diff_theme()
    if palette is None or not _is_light(palette):
        return DiffRenderer(highlighter=None, theme=colours)
    if current() is not None:
        # Monokai's near-white text vanishes on a light line; `default` leaves names in the palette's foreground.
        return DiffRenderer(highlighter=Highlighter(style='default'), theme=colours)
    # Monokai's near-white text vanishes on a light line, and `default`'s strings and comments fade on the red one.
    darker = min(colours.addition, colours.deletion, key=_luminance)
    return DiffRenderer(highlighter=_readable_highlighter(darker, palette.fg), theme=colours)


_SGR = re.compile(r'\x1b\[[0-9;]*m')
_TRUECOLOR_FOREGROUND = re.compile(r'(?<=[\[;])38;2;(\d+);(\d+);(\d+)(?=[;m])')
"""A 24-bit foreground inside an `_SGR` sequence, which may also carry bold or italic."""


def _readable_highlighter(bg: str, fg: str) -> Highlighter:
    """Pygments' `default` style, each colour raised to read on `bg` as `_readable` raises a palette's."""
    from termflow.syntax import Highlighter

    def raised(match: re.Match[str]) -> str:
        colour = _readable('#' + ''.join(f'{int(channel):02X}' for channel in match.groups()), bg, fg)
        return '38;2;' + ';'.join(str(int(colour[index : index + 2], 16)) for index in (1, 3, 5))

    class Readable(Highlighter):
        def highlight_line(self, line: str, language: str = 'text') -> str:
            line = super().highlight_line(line, language)
            return _SGR.sub(lambda sgr: _TRUECOLOR_FOREGROUND.sub(raised, sgr.group()), line)

    return Readable(style='default')


def _is_light(palette: TerminalPalette) -> bool:
    return _is_light_colour(palette.bg)


def _is_light_colour(colour: str) -> bool:
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return 0.299 * red + 0.587 * green + 0.114 * blue > 128


def roles(name: str) -> tuple[tuple[str, str], ...]:
    """The Rich colour `name` paints each role with; where roles share a colour, the earlier one claims it.

    Roles are the background and foreground, diff lines and markers, CLAI's brand colours, and the
    16 ANSI slots. The default theme leaves the background and foreground to the terminal, and the slots too on a
    dark or unknown background; on a light one they are `github_light`'s.
    Output painted in one theme repaints in another by translating its colours role by role.
    """
    with use(lambda: name):
        palette = current()
        slots = _surface()
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
        *((f'ansi {slot}', slots.ansi[slot] if slots is not None else f'color({slot})') for slot in range(16)),
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
def use(
    get_name: Callable[[], str], *, output: IO[str] | None = None, background: str | None = None
) -> Generator[None]:
    """Scope colours to a shell, leaving the terminal untouched when no palette is chosen.

    `background` is the terminal's background colour (`#RRGGBB`); `None` keeps the enclosing scope's. On a
    light one the default theme raises its colours to WCAG AA contrast against it and uses light code and
    diff colours; on a dark or unknown one it keeps its original colours.
    """
    active = _ACTIVE
    token = active.set(get_name)
    light = _LIGHT_BACKGROUND.set(background if _is_light_colour(background) else None) if background else None
    try:
        if output is not None and get_name() != 'default':
            apply(get_name(), output=output)
        yield
    finally:
        if output is not None and get_name() != 'default':
            apply('default', output=output)
        if light is not None:
            _LIGHT_BACKGROUND.reset(light)
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
# Dark-theme diff backgrounds: a green and a red at the same depth, so additions read as additions.
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
    shade = role.removeprefix('bold ')
    if current() is not None and shade in _SLOTS:
        slot = _SLOTS[shade]
        code = str(30 + slot if slot < 8 else 90 + slot - 8)
    elif light_background() is not None:
        from rich.color import Color, ColorSystem

        # The bright 16 colours are tuned for dark terminals; `color` picked a readable 256-colour shade.
        code = ';'.join(Color.parse(resolved).downgrade(ColorSystem.EIGHT_BIT).get_ansi_codes())
    else:
        code = str(_BASIC[shade])
    return f'\x1b[{prefix}{code}m'
