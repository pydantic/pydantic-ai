"""Candidate previews use real rendering colours without applying a terminal palette."""

import io
import math
import re
from pathlib import Path

import pytest
from pygments.token import Keyword
from rich.color import Color
from rich.console import Console
from rich.syntax import ANSI_LIGHT
from rich.text import Text
from termflow.ansi.utils import visible_length
from termflow.themes import PALETTES
from termflow.tui.keys import Key

from pydantic_ai import PartStartEvent, TextPart
from pydantic_ai_harness.file_system import FileEditedEvent
from pydantic_ai_pax import StreamRenderer
from pydantic_ai_pax.ui.menus.theme_picker import theme_preview
from pydantic_ai_pax.ui.rendering import theme
from tests.pax.key_contract import Case, Surface
from tests.pax.key_surfaces import KEY_SURFACES, Site


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
        'Pax',
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
    header = text.get_style_at_offset(console, text.plain.index('Pax'))
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


async def _render_sample() -> str:
    """A heading, inline code, a link, a code fence, and a diff, as the default theme streams them."""
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None, show_tool_output=True)
    await renderer.on_stream_event(
        PartStartEvent(
            index=0,
            part=TextPart(
                content='# Heading `add()` after\n\nSee `add()` at https://example.com\n\n```python\ndef add(a, b):\n    return a\n```\n'
            ),
        )
    )
    await renderer.on_stream_event(
        FileEditedEvent(
            path='demo.py',
            root_dir='/tmp',
            content_hash='hash',
            diff='--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-return "old"  # was\n+return "[38;2;1;2;3m"  # now\n',
            truncated=False,
        )
    )
    await renderer.finish()
    return output.getvalue()


@pytest.mark.parametrize('background', ['#000000', '#1E1E2E', '#808080'])
async def test_default_theme_is_unchanged_on_a_dark_background(background: str) -> None:
    with theme.use(lambda: 'default'):
        unknown = (await _render_sample(), theme.roles('default'), theme.sgr(theme.INFO))
    with theme.use(lambda: 'default', background=background):
        assert theme.light_background() is None
        dark = (await _render_sample(), theme.roles('default'), theme.sgr(theme.INFO))
        assert theme.color(theme.INFO) == theme.INFO
        assert theme.diff_theme().addition == theme.DIFF_ADDITION
    assert dark == unknown


def _hex_colours(rendered: str) -> set[tuple[str, str]]:
    """Each truecolor foreground in `rendered` with the truecolor background it sits on, `''` for none."""
    colours: set[tuple[str, str]] = set()
    foreground = background = ''
    for codes, text in re.findall(r'\x1b\[([0-9;]*)m([^\x1b]*)', rendered):
        parts = codes.split(';')
        index = 0
        while index < len(parts):
            if parts[index] in ('38', '48') and parts[index + 1 : index + 2] == ['2']:
                shade = '#' + ''.join(f'{int(channel):02X}' for channel in parts[index + 2 : index + 5])
                foreground, background = (shade, background) if parts[index] == '38' else (foreground, shade)
                index += 5
                continue
            if parts[index] in ('', '0'):
                foreground = background = ''
            elif parts[index] == '39':
                foreground = ''
            elif parts[index] == '49':
                background = ''
            index += 1
        if foreground and text.strip():
            colours.add((foreground, background))
    return colours


@pytest.mark.parametrize('background', ['#FFFFFF', '#FDF6E3', '#EEEEEE'])
async def test_default_theme_reads_on_a_light_background(background: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every brand role, in truecolor and as a 256-colour code, and the diff text reach WCAG AA (4.5:1)."""
    roles = (
        theme.ACCENT,
        theme.INFO,
        theme.SUCCESS,
        theme.WARNING,
        theme.ERROR,
        theme.MUTED,
        theme.THINKING,
        theme.SUGAR,
    )
    monkeypatch.setenv('COLORTERM', 'truecolor')
    with theme.use(lambda: 'default', background=background):
        colours = {role: theme.color(role).removeprefix('bold ') for role in roles}
        rendered = await _render_sample()
        diff = theme.diff_theme()
        monkeypatch.delenv('COLORTERM', raising=False)
        codes = {role: theme.sgr(role) for role in roles}
    assert len(set(colours.values())) == len(roles)  # Still one colour per role.
    for role, code in codes.items():
        colours[f'{role} 256'] = Color.from_ansi(int(code.split(';')[-1].rstrip('m'))).get_truecolor().hex
    assert {role: colour for role, colour in colours.items() if _contrast(colour, background) < 4.5} == {}
    painted = _hex_colours(rendered)
    assert (colours[theme.MUTED].upper(), '') in painted and '\x1b[2m' not in rendered  # Inline code, not dim.
    heading, console = Text.from_ansi(rendered).split()[0], Console()
    after = heading.get_style_at_offset(console, heading.plain.index('after'))
    assert after.color == heading.get_style_at_offset(console, 0).color  # Its colour comes back after the code.
    # Diff markers are tinted from their line, like a light palette's.
    markers = {diff.addition_marker.upper(), diff.deletion_marker.upper()}
    assert {(fg, bg) for fg, bg in painted if fg not in markers and _contrast(fg, bg or background) < 4.5} == set()
    assert '[38;2;1;2;3m' in re.sub(r'\x1b\[[0-9;]*m', '', rendered)  # Code that looks like an escape stays as written.
    assert {bg for _, bg in painted} >= {diff.addition.upper(), diff.deletion.upper()}


async def test_default_preview_over_a_dark_palette_paints_the_light_background() -> None:
    """Browsing `/theme` on a light terminal with a dark palette applied previews `default` on the terminal's white."""
    with theme.use(lambda: 'tokyo_night', background='#FFFFFF'):
        preview = theme_preview('default', width=60)
    painted = _hex_colours(preview)
    assert {bg for _, bg in painted} == {'#FFFFFF'}
    assert {(fg, bg) for fg, bg in painted if _contrast(fg, bg) < 4.5} <= {
        (theme.LOGO.removeprefix('bold '), '#FFFFFF')
    }


UNSTYLED_MENUS = {
    ('models.openrouter', 'choose', 'MenuBuilder'),
    ('models.openrouter', 'connection_action', 'MenuBuilder'),
    ('models.vllm', 'choose', 'MenuBuilder'),
    ('models.vllm', 'connection_action', 'MenuBuilder'),
}
"""Menus built without `markdown_style()`: termflow's default style paints their title and highlighted row `#87CEEB`
(1.74:1 on white) and their `search:` label `#DDA0DD` (2.07:1)."""

SHOWS_LOGO = {('ui.menus.theme_picker', 'build_theme_picker', 'MenuBuilder')}
"""Menus whose preview draws the logo, which keeps its brand colours on every background."""

MENUS = [
    pytest.param(
        site,
        case,
        id=f'{".".join(site[:2])}:{case.name}',
        marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason='built without `markdown_style()`')
        if site in UNSTYLED_MENUS
        else (),
    )
    for site, surface in KEY_SURFACES.items()
    if isinstance(surface, Surface)
    for case in surface.cases
]


@pytest.mark.parametrize('background', ['#FFFFFF', '#FDF6E3', '#EEEEEE'])
@pytest.mark.parametrize(('site', 'case'), MENUS)
async def test_every_registered_menu_reads_on_a_light_background(
    site: Site,
    case: Case,
    background: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under `default` on a light terminal, every colour a `KEY_SURFACES` menu paints reaches WCAG AA (4.5:1).

    A menu coloured through `theme.color()`, `theme.sgr()` or `markdown_style()` gets the light shades; one that
    paints a colour of its own, or keeps termflow's default style, does not. Only truecolor foregrounds are measured:
    the terminal draws its own ANSI colours and faint text.
    """
    monkeypatch.setenv('COLORTERM', 'truecolor')
    with theme.use(lambda: 'default', background=background):
        await case.press(tmp_path, Key.ESCAPE)
    painted = _hex_colours(capsys.readouterr().out)
    assert painted, 'the menu painted no colour: check that its case paints on `sys.stdout`'
    logo = theme.LOGO.removeprefix('bold ') if site in SHOWS_LOGO else None
    assert {(fg, bg) for fg, bg in painted if fg != logo and _contrast(fg, bg or background) < 4.5} == set()
