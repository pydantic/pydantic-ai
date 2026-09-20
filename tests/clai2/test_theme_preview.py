"""Candidate previews use real rendering colours without applying a terminal palette."""

import io

import pytest
from pydantic_ai import PartStartEvent, TextPart
from pydantic_ai_harness.filesystem import FileEditedEvent
from rich.color import Color
from rich.console import Console
from rich.text import Text
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import StreamRenderer, theme
from pydantic_clai2.theme_picker import theme_preview


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
    foreground = palette.ansi[12] if palette is not None else theme.LITHIUM
    header = text.get_style_at_offset(console, text.plain.index('CLAI 2.0'))
    assert header.color == Color.parse(foreground)
    assert header.bgcolor == (Color.parse(palette.bg) if palette is not None else None)
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


async def test_selected_theme_reaches_markdown_and_keeps_termflow_diff_defaults() -> None:
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
    assert '\x1b[48;2;14;68;41m' in output.getvalue()
    assert '\x1b[48;2;103;6;12m' in output.getvalue()
