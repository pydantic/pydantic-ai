"""Markdown labels link to their URLs without leaking into editor output."""

import io

import anyio
import pytest
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta
from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import StreamRenderer
from pydantic_clai2._rendering import LinkOutput
from pydantic_clai2.prompt_surface import PromptSurface

URL = 'https://github.com/pydantic/pydantic-ai-harness/pull/1006'
OPEN = f'\x1b]8;;{URL}\x1b\\'
CLOSE = '\x1b]8;;\x1b\\'


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


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
            if '\x1b]8;;https://' in text:
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
    now = 0.0
    surface = PromptSurface(output=output, size=lambda: size, clock=lambda: now)
    surface.paint(('prompt',))
    console = Console(file=surface, force_terminal=True, width=120)
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(f'Opened [PR #1006]({URL}).')))
    await renderer.finish()
    size = (100, 30)
    surface.paint(('prompt',))
    start = len(output.getvalue())
    now = 0.3
    surface.paint(('prompt',))
    replay = output.getvalue()[start:]
    text = Text.from_ansi(replay)
    assert text.get_style_at_offset(console, text.plain.index('PR #1006')).link == URL
    assert text.get_style_at_offset(console, text.plain.index('prompt')).link is None
    surface.release()


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
