"""Candidate previews use real rendering colours without applying a terminal palette."""

import io
import math

import pytest
from pygments.token import Keyword
from rich.color import Color
from rich.console import Console
from rich.syntax import ANSI_LIGHT
from rich.text import Text
from termflow.ansi.utils import visible_length
from termflow.themes import PALETTES

from pydantic_ai import PartStartEvent, TextPart
from pydantic_ai_harness.file_system import FileEditedEvent
from pydantic_clai2 import StreamRenderer
from pydantic_clai2.ui.menus.theme_picker import theme_preview
from pydantic_clai2.ui.rendering import theme


@pytest.mark.parametrize('name', theme.names())
@pytest.mark.parametrize('width', [46, 86])
def test_conversation_preview_uses_candidate_colours(name: str, width: int) -> None:
    console = Console()
    with theme.use(lambda: 'tokyo_night'):
        rendered = theme_preview(name, width=width)
        assert theme.current() is PALETTES['tokyo_night']
    assert '\x1b]' not in rendered
    assert all(visible_length(line) == width for line in rendered.splitlines())
    text = Text.from_ansi(rendered)
    for fragment in (
        'CLAI 2.0',
        'Thinking',
        'read_file',
        'Summary',
        'return',
        'Warning:',
        'Error:',
        'Ask a follow-up',
        'ready',
    ):
        assert fragment in text.plain
    palette = PALETTES.get(name)
    header = text.get_style_at_offset(console, text.plain.index('CLAI 2.0'))
    assert header.color == Color.parse(theme.LITHIUM)
    assert header.bgcolor == (Color.parse(palette.bg) if palette is not None else None)
    with theme.use(lambda: name):
        keyword = theme.syntax_theme().get_style_for_token(Keyword).color
    assert text.get_style_at_offset(console, text.plain.index('return')).color == (
        keyword if palette is not None else Color.parse('color(4)')
    )
    assert text.get_style_at_offset(console, text.plain.index('return')).bgcolor == (
        Color.parse(palette.bg) if palette is not None else Color.default()
    )
    assert text.get_style_at_offset(console, text.plain.index('Summarize')).color == (
        Color.parse(palette.fg) if palette is not None else None
    )


@pytest.mark.parametrize('name', theme.names())
@pytest.mark.parametrize('truecolor', [False, True])
def test_roles_support_raw_terminal_surfaces(name: str, truecolor: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor' if truecolor else '')
    with theme.use(lambda: name):
        assert theme.color('italic') == 'italic'
        for role in (
            theme.ACCENT,
            theme.INFO,
            theme.WARNING,
            theme.ERROR,
            theme.MUTED,
            theme.THINKING,
            theme.SUGAR,
            theme.LIGHT_PURPLE,
        ):
            escape = theme.sgr(role)
            assert escape.startswith('\x1b[') and escape.endswith('m')
            assert ('38;2;' in escape) == truecolor


def _linear(colour: str) -> tuple[float, ...]:
    channels = (int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5))
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def _contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio."""
    darker, lighter = sorted(sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), _linear(x))) for x in (a, b))
    return (lighter + 0.05) / (darker + 0.05)


# Machado, Oliveira and Fernandes (2009) at full severity, applied to linear RGB.
_DICHROMACIES = {
    'protanopia': ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    'deuteranopia': ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.01182, 0.04294, 0.968881)),
    'tritanopia': ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.3039)),
}


def _seen(colour: str, matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float]:
    """`colour` in CIELAB as a dichromat sees it."""
    rgb = [min(1.0, max(0.0, sum(m * c for m, c in zip(row, _linear(colour))))) for row in matrix]
    xyz = (
        (0.4124 * rgb[0] + 0.3576 * rgb[1] + 0.1805 * rgb[2]) / 0.95047,
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2],
        (0.0193 * rgb[0] + 0.1192 * rgb[1] + 0.9505 * rgb[2]) / 1.08883,
    )
    fx, fy, fz = (v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116 for v in xyz)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


@pytest.mark.parametrize('name', [name for name in theme.names() if name != 'default'])
def test_role_and_code_colours_are_readable_on_their_palette(name: str) -> None:
    """Text reaches WCAG AA (4.5:1) on the palette background, or the palette's own text contrast if lower."""
    palette = PALETTES[name]
    floor = min(4.5, _contrast(palette.fg, palette.bg))
    roles = (theme.ACCENT, theme.INFO, theme.SUCCESS, theme.WARNING, theme.ERROR, theme.MUTED, theme.THINKING)
    with theme.use(lambda: name):
        colours = {role: theme.color(role).removeprefix('bold ') for role in roles}
        syntax = theme.syntax_theme()
    for token in ANSI_LIGHT:
        color = syntax.get_style_for_token(token).color
        if color is not None:
            colours[str(token)] = color.get_truecolor().hex
    assert {role: colour for role, colour in colours.items() if _contrast(colour, palette.bg) < floor} == {}


@pytest.mark.parametrize(
    'name',
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason='The default theme keeps its public `DIFF_ADDITION`/`DIFF_DELETION` colours (#13001, #13045).',
            ),
        )
        if name == 'default'
        else name
        for name in theme.names()
    ],
)
def test_diff_lines_stay_apart_for_colour_blind_readers(name: str) -> None:
    with theme.use(lambda: name):
        colors = theme.diff_theme()
    distances = {
        kind: math.dist(_seen(colors.addition, matrix), _seen(colors.deletion, matrix))
        for kind, matrix in _DICHROMACIES.items()
    }
    assert min(distances.values()) >= 10, distances


@pytest.mark.parametrize(
    ('name', 'addition', 'deletion', 'marker'),
    [
        ('default', '#225C2B', '#7A2936', '#66ff81'),
        ('tokyo_night', '#4A5B3E', '#462D3B', '#deffba'),
        ('github_light', '#E5F4E9', '#EDA8AF', '#727a74'),
    ],
)
def test_diff_colours_follow_the_palette(name: str, addition: str, deletion: str, marker: str) -> None:
    with theme.use(lambda: name):
        colors = theme.diff_theme()
    assert (colors.addition, colors.deletion, colors.addition_marker) == (addition, deletion, marker)


async def test_selected_theme_reaches_markdown_and_diffs() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor')
    with theme.use(lambda: 'github_light'):
        renderer = StreamRenderer(console, stop_loading=lambda: None, show_tool_output=True)
        await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='# Heading\n')))
        await renderer.on_stream_event(
            FileEditedEvent(
                path='demo.py',
                root_dir='/tmp',
                content_hash='hash',
                diff='--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-old\n+new\n',
                truncated=False,
            )
        )
        await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    assert text.get_style_at_offset(console, text.plain.index('Heading')).color == Color.parse(
        PALETTES['github_light'].ansi[12]
    )
    assert '\x1b[48;2;229;244;233m' in output.getvalue()
    assert '\x1b[48;2;237;168;175m' in output.getvalue()
    assert text.get_style_at_offset(console, text.plain.index('new')).color is None
