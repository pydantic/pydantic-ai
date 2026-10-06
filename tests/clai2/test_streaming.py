"""Exercise the real Termflow drainer without timing-based assertions."""

import asyncio
import io
from typing import IO, Literal

import pytest
from rich.console import Console
from rich.text import Text
from termflow.stream import SmoothWriter

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent, TextPart
from pydantic_ai.messages import ThinkingPart, ThinkingPartDelta, ToolCallPart, ToolReturnPart
from pydantic_clai2 import StreamRenderer
from pydantic_clai2.config import Settings
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import color_system, render_markdown, thinking_heading


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


async def test_thinking_streams_complete_lines_before_part_end() -> None:
    emitted = asyncio.Event()

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            if 'z' in text:
                emitted.set()
            return super().write(text)

    output = ObservedOutput()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='zzzzz\n')))
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='zzz')))
    try:
        await asyncio.wait_for(emitted.wait(), timeout=2)
    finally:
        await renderer.finish()
    assert 'z' in output.getvalue()


async def test_redirected_thinking_is_dim_markdown() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=False), stop_loading=lambda: None)
    await renderer.on_stream_event(
        PartStartEvent(index=0, part=ThinkingPart(content='## Plan first\n[bold]literal[/bold]\n'))
    )
    await renderer.finish()
    assert '## Plan' not in output.getvalue()
    assert '\x1b[2m' in output.getvalue()  # Termflow's dim renderer paints the reasoning
    plain = Text.from_ansi(output.getvalue()).plain
    assert plain.startswith('Thinking Plan first')
    assert '[bold]literal[/bold]' in plain  # Rich markup in reasoning stays literal


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
            if 'x' in text:  # pragma: no branch
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
    observed: list[tuple[float, float, int]] = []

    def writer(
        target: IO[str], *, tick_interval: float, catch_up_seconds: float, min_chars_per_tick: int
    ) -> SmoothWriter:
        observed.append((tick_interval, catch_up_seconds, min_chars_per_tick))
        return SmoothWriter(
            target,
            tick_interval=tick_interval,
            catch_up_seconds=catch_up_seconds,
            min_chars_per_tick=min_chars_per_tick,
        )

    monkeypatch.setattr('pydantic_clai2.ui.rendering._rendering.SmoothWriter', writer)
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, force_terminal=True), stop_loading=lambda: None, smooth_seconds=Settings().smooth_seconds
    )
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='thinking text\n')))
    await renderer.on_stream_event(PartStartEvent(index=1, part=TextPart(content='response text\n')))
    await renderer.finish()
    assert observed == [(0.02, 0.4, 2), (0.012, 0.5, 1)]
    text = Text.from_ansi(output.getvalue()).plain
    assert 'Thinking thinking text' in text
    assert 'response text' in text


def _rows(surface: PromptSurface, *, width: int) -> list[str]:
    return [Text.from_ansi(row).plain for row in surface.transcript.frame(width=width, height=50).rows]


async def test_live_panel_renders_streamed_markdown_again_for_a_new_width_and_theme() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (120, 24))
    console = Console(file=surface, force_terminal=True, width=120, color_system='truecolor')
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth_seconds=0)
    words = ' '.join(f'word{index}' for index in range(30))
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content=words)))
    await renderer.finish()
    wide = _rows(surface, width=120)
    assert wide[0].startswith('Thinking word0')
    narrow = _rows(surface, width=40)
    assert len(narrow) > len(wide)
    assert all(len(row) <= 40 for row in narrow)
    assert ''.join(''.join(narrow).split()) == 'Thinking' + ''.join(words.split())

    def heading() -> str:
        """The heading's SGR as this console emits it, whatever Rich cached for the style."""
        return thinking_heading(console).split('Thinking', 1)[0]

    default = heading()
    assert surface.transcript.frame(width=120, height=50).rows[0].startswith(default)
    with theme.use(lambda: 'github_light'):
        assert heading() != default
        assert surface.transcript.frame(width=120, height=50).rows[0].startswith(heading())


@pytest.mark.parametrize('system', ['standard', '256', 'truecolor'])
def test_replay_renders_with_the_stream_consoles_colour_system(system: Literal['standard', '256', 'truecolor']) -> None:
    """Rich caches a style's ANSI codes on the shared style, so a replay must not pick its own system."""
    console = Console(file=io.StringIO(), force_terminal=True, color_system=system)
    assert color_system(console) == system
    replay = render_markdown(source='thought', width=40, thinking=True, colors=color_system(console))
    assert replay.startswith(thinking_heading(console).split('Thinking', 1)[0])


def test_replay_without_colour_emits_no_styling() -> None:
    assert color_system(Console(file=io.StringIO(), color_system=None)) is None
    replay = render_markdown(source='**thought**', width=40, thinking=True, colors=None)
    # Rich's heading is uncoloured; Termflow's bold and dim, like the stream's, are not colours.
    assert replay.startswith('Thinking ')
    assert all(span.style.color is None for span in Text.from_ansi(replay).spans if not isinstance(span.style, str))
    assert Text.from_ansi(replay).plain.split() == ['Thinking', 'thought']


async def test_aborted_live_part_never_shows_unstreamed_source() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (120, 24))
    renderer = StreamRenderer(Console(file=surface, force_terminal=True, width=120), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='never shown')))
    await renderer.abort()
    assert 'never shown' not in ' '.join(_rows(surface, width=40))


def test_whole_markdown_repaint_processes_complete_source_lines() -> None:

    assert Text.from_ansi(
        render_markdown(source='one\ntwo\n', width=40, thinking=False, colors='truecolor')
    ).plain.split() == [
        'one',
        'two',
    ]
