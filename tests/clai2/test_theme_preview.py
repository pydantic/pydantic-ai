"""Candidate previews use real rendering colours without applying a terminal palette."""

import io

import pytest
from rich.color import Color
from rich.console import Console
from rich.text import Text
from termflow.ansi.utils import visible_length
from termflow.themes import PALETTES

from pydantic_ai import PartStartEvent, TextPart
from pydantic_ai_harness.filesystem import FileEditedEvent
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
    assert text.get_style_at_offset(console, text.plain.index('return')).color == Color.parse(
        palette.ansi[4] if palette is not None else 'color(4)'
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


@pytest.mark.parametrize(
    ('name', 'addition', 'deletion', 'marker'),
    [
        ('default', '#225C2B', '#7A2936', '#66ff81'),
        ('tokyo_night', '#3F4D39', '#583443', '#bde7ab'),
        ('github_light', '#C3E6CB', '#F4C8CC', '#617365'),
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
    assert '\x1b[48;2;195;230;203m' in output.getvalue()
    assert '\x1b[48;2;244;200;204m' in output.getvalue()
    assert text.get_style_at_offset(console, text.plain.index('new')).color is None
