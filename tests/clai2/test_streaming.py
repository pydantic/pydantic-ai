"""Exercise the real Termflow drainer without timing-based assertions."""

import asyncio
import io
from collections.abc import Callable
from typing import IO

import pytest
from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartStartEvent, TextPart, ThinkingPart
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from rich.console import Console
from rich.text import Text
from termflow.stream import SmoothWriter, StreamSmoother  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import StreamRenderer
from pydantic_clai2.config import Settings


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_intermediate_text_flushes_before_tool_arguments() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='Working on it.')))
    await renderer.on_stream_event(PartStartEvent(index=1, part=ToolCallPart(tool_name='shell', args='')))
    assert 'Working on it.' in output.getvalue()
    assert output.getvalue().endswith('\n\n')
    assert 'CLAI' not in output.getvalue()
    before = output.getvalue()
    await renderer.finish()
    assert output.getvalue() == before


@pytest.mark.parametrize('show_tool_output', [False, True])
async def test_tools_have_one_line_with_blank_separators(show_tool_output: bool) -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_tool_output=show_tool_output)
    for name in ('shell', 'write_file', 'shell'):
        await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool_name=name, args='{}')))
        await renderer.on_stream_event(
            FunctionToolResultEvent(part=ToolReturnPart(tool_name=name, content='done', tool_call_id='test'))
        )
    await renderer.finish()
    assert output.getvalue() == '● shell\n\n● write_file\n\n● shell\n\n'


async def test_long_tool_name_does_not_wrap() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=20), stop_loading=lambda: None)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool_name='a' * 100, args='{}')))
    assert len(output.getvalue().splitlines()) == 2
    assert output.getvalue().endswith('\n\n')


async def test_markdown_uses_brand_palette() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=False), stop_loading=lambda: None)
    await renderer.on_stream_event(
        PartStartEvent(index=0, part=TextPart(content='# Heading\n\n- item with [link](https://pydantic.dev)\n'))
    )
    await renderer.finish()
    assert '\x1b[38;2;229;32;233m' in output.getvalue()  # Lithium headings
    assert '\x1b[38;2;255;101;80m' in output.getvalue()  # Calcium list markers
    assert '\x1b[38;2;119;255;216m' in output.getvalue()  # Aqua links
    assert 'Heading' in output.getvalue()
    assert '\x1b]4;' not in output.getvalue()


async def test_empty_thinking_does_not_print_heading() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='', signature='signature')))
    await renderer.finish()
    assert 'Thinking' not in output.getvalue()


async def test_thinking_streams_before_newline_or_part_end() -> None:
    emitted = asyncio.Event()

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            if 'z' in text:
                emitted.set()
            return super().write(text)

    output = ObservedOutput()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='zzzzz')))
    try:
        await asyncio.wait_for(emitted.wait(), timeout=2)
    finally:
        await renderer.finish()
    assert 'z' in output.getvalue()


async def test_redirected_thinking_is_immediate_and_literal() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=False), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='[bold]literal')))
    assert '[bold]literal' in output.getvalue()
    await renderer.finish()


async def test_burst_is_queued_then_drained() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='Burst of text\n')))
    assert 'Burst of text' not in output.getvalue()
    await renderer.finish()
    assert 'Burst of text' in output.getvalue()
    before = output.getvalue()
    await renderer.finish()
    assert output.getvalue() == before


async def test_abort_discards_pending_output() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='Discard this\n')))
    await renderer.abort()
    await renderer.finish()
    assert 'Discard this' not in output.getvalue()


async def test_cancel_during_drain_stops_writer() -> None:
    writing = asyncio.Event()

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            if 'x' in text:
                writing.set()
            return super().write(text)

    output = ObservedOutput()
    renderer = StreamRenderer(Console(file=output, width=20000, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='x' * 10000 + '\n')))
    task = asyncio.create_task(renderer.finish())
    await writing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await renderer.abort()
    assert output.getvalue().count('x') < 10000


async def test_smoothing_defaults_match_code_puppy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin smooth_stream.py defaults from Code Puppy a862bf478b63."""
    observed: dict[str, tuple[float, float, int]] = {}

    def thinking_smoother(
        emit: Callable[[str], None], *, tick_interval: float, catch_up_seconds: float, min_chars_per_tick: int
    ) -> StreamSmoother:
        observed['thinking'] = (tick_interval, catch_up_seconds, min_chars_per_tick)
        return StreamSmoother(
            emit, tick_interval=tick_interval, catch_up_seconds=catch_up_seconds, min_chars_per_tick=min_chars_per_tick
        )

    def response_writer(
        target: IO[str], *, tick_interval: float, catch_up_seconds: float, min_chars_per_tick: int
    ) -> SmoothWriter:
        observed['response'] = (tick_interval, catch_up_seconds, min_chars_per_tick)
        return SmoothWriter(
            target,
            tick_interval=tick_interval,
            catch_up_seconds=catch_up_seconds,
            min_chars_per_tick=min_chars_per_tick,
        )

    monkeypatch.setattr('pydantic_clai2._rendering.StreamSmoother', thinking_smoother)
    monkeypatch.setattr('pydantic_clai2._rendering.SmoothWriter', response_writer)
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, force_terminal=True), stop_loading=lambda: None, smooth_seconds=Settings().smooth_seconds
    )
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='thinking text')))
    await renderer.on_stream_event(PartStartEvent(index=1, part=TextPart(content='response text\n')))
    await renderer.finish()
    assert observed == {'thinking': (0.02, 0.4, 2), 'response': (0.012, 0.5, 1)}
    text = Text.from_ansi(output.getvalue()).plain
    assert 'thinking text' in text
    assert 'response text' in text
