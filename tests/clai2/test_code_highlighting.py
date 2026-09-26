"""Fenced code stays literal and is lexed with the complete block's context."""

import io

import pytest
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart
from rich.color import Color
from rich.console import Console
from rich.text import Text
from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import StreamRenderer, theme


@pytest.mark.parametrize('chunk_size', [1, 1000])
async def test_multiline_strings_keep_their_highlight(chunk_size: int) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    content = 'Before\n\n```python\nvalue = """inside\nreturn False\n"""\nreturn False\n```\n\nAfter'
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart('')))
    for offset in range(0, len(content), chunk_size):
        await renderer.on_stream_event(
            PartDeltaEvent(index=0, delta=TextPartDelta(content[offset : offset + chunk_size]))
        )
    await renderer.finish()
    rendered = Text.from_ansi(output.getvalue())
    inside = rendered.plain.index('inside')
    string_return = rendered.plain.index('return False')
    keyword_return = rendered.plain.rindex('return False')
    assert (
        rendered.get_style_at_offset(console, inside).color
        == rendered.get_style_at_offset(console, string_return).color
    )
    assert (
        rendered.get_style_at_offset(console, string_return).color
        != rendered.get_style_at_offset(console, keyword_return).color
    )
    assert rendered.plain.index('Before') < inside < rendered.plain.index('After')


@pytest.mark.parametrize('language', ['yml', 'jsonc', 'nodejs', 'JS'])
async def test_language_aliases_keep_syntax_colors(language: str) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor')
    renderer = StreamRenderer(console, stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'```{language}\n{{"key": true}}\n```\n')))
    await renderer.finish()
    rendered = Text.from_ansi(output.getvalue())
    key = rendered.get_style_at_offset(console, rendered.plain.index('key')).color
    value = rendered.get_style_at_offset(console, rendered.plain.index('true')).color
    assert key is not None and value is not None and key != value


@pytest.mark.parametrize('language', ['', 'markdown', 'md', 'unknown-language', 'text'])
@pytest.mark.parametrize('closed', [False, True])
async def test_fences_preserve_literal_text(language: str, closed: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    code = '# Heading\n\n    **literal** [link](https://example.com)\n'
    content = f'```{language}\n{code}' + ('```' if closed else '')
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content)))
    await renderer.finish()
    assert code in output.getvalue()
    before = output.getvalue()
    await renderer.finish()
    assert output.getvalue() == before


async def test_code_wraps_without_losing_content_or_interpreting_controls() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=24), stop_loading=lambda: None)
    await renderer.on_stream_event(
        PartStartEvent(index=0, part=TextPart('```text\n' + 'x' * 60 + '\x1b]52;secret\x07\n```\n'))
    )
    await renderer.finish()
    assert ''.join(output.getvalue().splitlines()[1:-2]) == 'x' * 60 + r'\x1b]52;secret\x07'
    assert all(len(line) <= 24 for line in output.getvalue().splitlines())
    assert '\x1b]' not in output.getvalue()
    assert '\x07' not in output.getvalue()


async def test_cancelled_fence_does_not_leak_into_next_part() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart('```python\nsecret = """\n')))
    await renderer.abort()
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart('```text\nfresh\n```\n')))
    await renderer.finish()
    assert 'secret' not in output.getvalue()
    assert 'fresh' in output.getvalue()


@pytest.mark.parametrize('name', ['default', 'github_light', 'tokyo_night'])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('language', ['python', 'unknown-language', ''])
async def test_code_uses_terminal_foreground_and_palette_syntax(name: str, thinking: bool, language: str) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor')
    content = f'```{language}\nif items < 2:\n    return items\n```\n'
    with theme.use(lambda: name):
        renderer = StreamRenderer(console, stop_loading=lambda: None)
        await renderer.on_stream_event(
            PartStartEvent(index=0, part=ThinkingPart(content) if thinking else TextPart(content))
        )
        await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    for fragment in ('items', '<', ':'):
        style = text.get_style_at_offset(console, text.plain.index(fragment))
        assert style.color is None or style.color.is_default
        assert style.bgcolor is None or style.bgcolor.is_default
        assert bool(style.dim) == thinking
    if language == 'python':
        keyword = text.get_style_at_offset(console, text.plain.index('return'))
        assert keyword.color == Color.parse(PALETTES[name].ansi[4] if name != 'default' else 'color(4)')
    assert '\x1b]' not in output.getvalue()
