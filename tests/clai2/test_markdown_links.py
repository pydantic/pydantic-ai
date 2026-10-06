"""Markdown labels link to their URLs without leaking into editor output."""

import io

import anyio
import pytest
from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.text import Text

from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta
from pydantic_clai2 import StreamRenderer
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import LinkOutput

URL = 'https://github.com/pydantic/pydantic-ai-harness/pull/1006'
OPEN = f'\x1b]8;;{URL}\x1b\\'
CLOSE = '\x1b]8;;\x1b\\'


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
async def test_markdown_link_labels(*, terminal: bool, thinking: bool) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=terminal, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    content = 'Opened [PR #1006]('
    part = ThinkingPart(content) if thinking else TextPart(content)
    await renderer.on_stream_event(PartStartEvent(index=0, part=part))
    delta = ThinkingPartDelta(content_delta=f'{URL}).') if thinking else TextPartDelta(content_delta=f'{URL}).')
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=delta))
    await renderer.finish()
    value = output.getvalue()
    text = Text.from_ansi(value)
    assert 'PR #1006' in text.plain and URL in text.plain
    assert text.get_style_at_offset(console, text.plain.index('PR #1006')).link == (URL if terminal else None)
    assert text.get_style_at_offset(console, text.plain.index(URL)).link is None
    assert ('\x1b]8;' in value) is terminal


@pytest.mark.parametrize(
    ('content', 'url', 'after'),
    [
        (f'See {URL}.', URL, '.'),
        (f'See <{URL}> now', URL, ' now'),
        (f'- **{URL}**', URL, ''),
        ('| link |\n|---|\n| https://ai.pydantic.dev |', 'https://ai.pydantic.dev', ' '),
        ('(https://en.wikipedia.org/wiki/Foo_(bar)).', 'https://en.wikipedia.org/wiki/Foo_(bar)', ').'),
        ('(see http://[::1]).', 'http://[::1]', ').'),
        ('[see https://ai.pydantic.dev]', 'https://ai.pydantic.dev', ']'),
    ],
)
@pytest.mark.parametrize('terminal', [False, True])
async def test_bare_urls_are_highlighted_links(*, content: str, url: str, after: str, terminal: bool) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=terminal, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content)))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    start = text.plain.index(url)
    style = text.get_style_at_offset(console, start)
    assert style.link == (url if terminal else None)
    assert style.color is not None and style.color.triplet is not None
    assert style.color.triplet.hex == theme.AQUA.lower()
    assert bool(style.underline) is not terminal
    assert text.plain[start + len(url) :].startswith(after)
    if after.strip():
        assert text.get_style_at_offset(console, start + len(url)).link is None


@pytest.mark.parametrize(
    'content',
    [f'`curl {URL}`', f'[label]({URL})', f'![image]({URL})', f'x{URL}', f'\ufdd0{URL}\ufdd1', f'{URL}/~~a~~b'],
)
async def test_urls_outside_plain_text_stay_as_they_were(content: str) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content)))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    assert text.get_style_at_offset(console, text.plain.index(URL)).link is None


@pytest.mark.parametrize('marker', ['FDD0', 'E000'])
async def test_entities_cannot_forge_a_link_to_another_scheme(marker: str) -> None:
    end = f'{int(marker, 16) + 1:X}'
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'&#x{marker};file:///etc/passwd&#x{end};')))
    await renderer.finish()
    assert '\x1b]8;;' not in output.getvalue()
    assert 'file:///etc/passwd' in Text.from_ansi(output.getvalue()).plain


async def test_heading_colour_continues_after_a_bare_url() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'### See {URL} today')))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    before = text.get_style_at_offset(console, text.plain.index('See'))
    after = text.get_style_at_offset(console, text.plain.index('today'))
    assert text.get_style_at_offset(console, text.plain.index(URL)).link == URL
    assert before.color is not None and before.color.triplet is not None
    assert before.color.triplet.hex == theme.PURPLE.lower()
    assert (after.color, after.bold) == (before.color, True)


async def test_long_lines_of_many_urls_and_brackets() -> None:
    """Sized so a quadratic rescan per URL or per trailing bracket turns this sub-second test into ~20 seconds."""
    content = ' '.join(f'`c` https://x.dev/{i}' for i in range(8000)) + f' {URL}' + ')' * 200_000
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content)))
    await renderer.finish()
    assert output.getvalue().count('\x1b]8;;https://') == 8001
    assert f'{OPEN}' in output.getvalue()


def test_each_smooth_chunk_closes_its_link() -> None:
    output = io.StringIO()
    writer = LinkOutput(output=output)
    for chunk in (OPEN + 'PR ', '#1006', CLOSE + ' normal'):
        start = len(output.getvalue())
        assert writer.write(chunk) == len(chunk)
        writer.flush()
        emitted = output.getvalue()[start:]
        decoder = AnsiDecoder()
        text = decoder.decode_line(emitted)
        assert decoder.style.link is None
        assert text.get_style_at_offset(Console(), 0).link == (None if chunk.startswith(CLOSE) else URL)
    assert Text.from_ansi(output.getvalue()).plain == 'PR #1006 normal'


async def test_abort_mid_label_does_not_leave_a_hyperlink() -> None:
    started = anyio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if '\x1b]8;;https://' in text:  # pragma: no branch
                started.set()
            return super().write(text)

    output = Output()
    renderer = StreamRenderer(Console(file=output, force_terminal=True, width=120), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'[PR #1006]({URL}).\n')))
    await started.wait()
    await renderer.abort()
    decoder = AnsiDecoder()
    list(decoder.decode(output.getvalue()))
    assert decoder.style.link is None
    assert 'PR #1006' not in Text.from_ansi(output.getvalue()).plain


async def test_streamed_link_survives_surface_resize() -> None:
    output = io.StringIO()
    size = (120, 24)
    surface = PromptSurface(output=output, size=lambda: size)
    surface.paint(('prompt',))
    console = Console(file=surface, force_terminal=True, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'Opened [PR #1006]({URL}).')))
    await renderer.finish()
    size = (100, 30)
    start = len(output.getvalue())
    surface.paint(('prompt',))
    assert 'Opened PR #1006' in Text.from_ansi(output.getvalue()[start:]).plain
    surface.restore()
    printed = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1])
    assert printed.get_style_at_offset(console, printed.plain.index('PR #1006')).link == URL
    assert printed.get_style_at_offset(console, printed.plain.index('Opened')).link is None


async def test_model_control_bytes_cannot_inject_terminal_commands() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(
        PartStartEvent(index=0, part=TextPart('[label](https://example.com/\x1b]52;c;evil\x07)'))
    )
    await renderer.finish()
    assert '\x1b]52;' not in output.getvalue()
    assert '\x07' not in output.getvalue()


@pytest.mark.parametrize('length', [2048, 2049, 10000])
async def test_oversized_urls_remain_visible_without_repeated_metadata(*, length: int) -> None:
    url = 'https://example.com/' + 'x' * (length - len('https://example.com/'))
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=12000)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'[label]({url})')))
    await renderer.finish()
    text = Text.from_ansi(output.getvalue())
    assert 'label' in text.plain and url in text.plain
    assert text.get_style_at_offset(console, 0).link == (url if length <= 2048 else None)
    if length > 2048:
        assert output.getvalue().count(url) == 1


def test_oversized_url_does_not_amplify_slow_label_chunks() -> None:
    output = io.StringIO()
    writer = LinkOutput(output=output)
    url = 'https://example.com/' + 'x' * 10000
    writer.write(f'\x1b]8;;{url}\x1b\\')
    for _ in range(10000):
        writer.write('x')
    writer.write(CLOSE)
    assert len(output.getvalue()) < 10100
    assert Text.from_ansi(output.getvalue()).plain == 'x' * 10000


@pytest.mark.parametrize('length', [2048, 2049, 24000])
@pytest.mark.parametrize('width', [40, 120])
def test_markdown_replay_caps_hyperlink_metadata(length: int, width: int) -> None:
    from pydantic_clai2.ui.rendering._rendering import render_markdown

    url = 'https://example.com/' + 'x' * (length - len('https://example.com/'))
    label = 'label ' * 40
    rendered = render_markdown(source='[' + label + '](' + url + ')', width=width, thinking=False, colors='truecolor')
    text = Text.from_ansi(rendered)
    console = Console(file=io.StringIO())
    assert text.get_style_at_offset(console, text.plain.index('label')).link == (url if length <= 2048 else None)
    if length > 2048:
        assert '\x1b]8;;' + url not in rendered
        assert len(rendered) < 4 * (len(url) + len(label)), 'wrapped labels must not amplify oversized metadata'
