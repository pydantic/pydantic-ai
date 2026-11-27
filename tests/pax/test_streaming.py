"""Exercise the real Termflow drainer without timing-based assertions."""

import asyncio
import io
import re
from typing import IO, Literal

import anyio
import pytest
from rich.console import Console
from rich.text import Text

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.messages import ThinkingPart, ThinkingPartDelta, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.file_system import FileEditedEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandStartedEvent
from pydantic_ai_pax import StreamRenderer
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.rendering import (
    LinkOutput,
    PartWriter,
    color_system,
    render_markdown,
    thinking_heading,
)
from tests._timeouts import READINESS_WAIT_TIMEOUT


async def test_intermediate_text_flushes_before_tool_arguments() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='Working on it.')))
    await renderer.on_stream_event(PartStartEvent(index=1, part=ToolCallPart(tool_name='shell', args='')))
    assert 'Working on it.' in output.getvalue()
    assert output.getvalue().endswith('\n\n')
    assert 'Pax' not in output.getvalue()
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


MIXED_TURN = """\
I will look first.

● shell ls
  ⎿ exit 0

● edit_file a.py

-x = 1
+x = 2

deploy: staging ok

● lint path="a.py"

● read_file a.py

The new code:

python ─────────────────────────────────────────
x = 2
────────────────────────────────────────────────

Done.

"""


@pytest.mark.parametrize('terminal', [False, True])
async def test_one_blank_row_between_every_block_of_a_mixed_turn(terminal: bool) -> None:
    """Diffs, plugin output and code fences each get one blank row before and after."""
    output = io.StringIO()
    console = Console(file=output, width=48, force_terminal=terminal, color_system=None)

    def deploy(event: AgentStreamEvent) -> Text | None:
        deploying = isinstance(event, FunctionToolCallEvent) and event.part.tool_name == 'deploy'
        return Text('deploy: staging ok') if deploying else None

    renderer = StreamRenderer(console, stop_loading=lambda: None, renderers=[deploy], smooth=False)

    async def text(index: int, content: str) -> None:
        await renderer.on_stream_event(PartStartEvent(index=index, part=TextPart(content)))
        await renderer.on_stream_event(PartEndEvent(index=index, part=TextPart(content)))

    def call(name: str, call_id: str, **args: str) -> FunctionToolCallEvent:
        return FunctionToolCallEvent(part=ToolCallPart(name, args, tool_call_id=call_id))

    await text(0, 'I will look first.\n')
    await renderer.on_stream_event(call('shell', 'a', command='ls'))
    await renderer.on_stream_event(CommandStartedEvent(tool_call_id='a', command='ls', pid=1))
    await renderer.on_stream_event(
        CommandFinishedEvent(tool_call_id='a', pid=1, output_path='/o', status_path='/s', exit_code=0, truncated=False)
    )
    await renderer.on_stream_event(call('edit_file', 'b', path='a.py'))
    await renderer.on_stream_event(
        FileEditedEvent(
            tool_call_id='b',
            root_dir='/r',
            path='a.py',
            diff='--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n',
            truncated=False,
            content_hash='h',
        )
    )
    await renderer.on_stream_event(call('deploy', 'c'))
    await renderer.on_stream_event(call('lint', 'd', path='a.py'))
    await renderer.on_stream_event(call('read_file', 'e', path='a.py'))
    await text(1, 'The new code:\n```python\nx = 2\n```\nDone.\n')
    await renderer.finish()
    if terminal:
        # The terminal diff is Termflow's own rendering, which drops the file and hunk headers.
        rows = re.sub(r'\x1b\[[0-9;]*m', '', output.getvalue())
        assert rows == MIXED_TURN.replace('-x = 1\n+x = 2', '- x = 1\n+ x = 2')
    else:
        assert output.getvalue() == MIXED_TURN.replace(
            '-x = 1\n+x = 2', '--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2'
        )


async def test_long_tool_name_does_not_wrap() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=20), stop_loading=lambda: None)
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool_name='a' * 100, args='{}')))
    await renderer.finish()
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


@pytest.mark.parametrize('part', [TextPart(content='x' * 60 + '\n'), ThinkingPart(content='x' * 60 + '\n')])
async def test_part_end_writes_queued_text_at_once(
    part: TextPart | ThinkingPart, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool line must not wait for the smoother to type out the rest of the part."""
    writes: list[str] = []

    # A tick an hour apart: only the first can run before the part ends, so the count does not depend on the clock.
    def make_writer(self: StreamRenderer, output: IO[str]) -> PartWriter:
        return PartWriter(LinkOutput(output=output), tick_interval=3600, catch_up_seconds=0.5, min_chars_per_tick=1)

    monkeypatch.setattr(StreamRenderer, '_make_writer', make_writer)

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            writes.append(text)
            return super().write(text)

    output = ObservedOutput()
    renderer = StreamRenderer(Console(file=output, width=200, force_terminal=True), stop_loading=lambda: None)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await renderer.on_stream_event(PartStartEvent(index=0, part=part))
        await renderer.on_stream_event(PartStartEvent(index=1, part=ToolCallPart(tool_name='shell', args='')))
    counts = [text.count('x') for text in writes if 'x' in text]
    # At most the writer's first tick, then the rest in one write.
    assert sum(counts) == 60
    assert len(counts) <= 2


async def test_abort_discards_pending_output() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='Discard this\n')))
    await renderer.abort()
    await renderer.finish()
    assert 'Discard this' not in output.getvalue()


async def test_abort_while_typing_stops_writer() -> None:
    """Esc while a part is still being typed drops the rest; a part that ends writes it at once instead."""
    writing = asyncio.Event()

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            if 'x' in text:  # pragma: no branch
                writing.set()
            return super().write(text)

    output = ObservedOutput()
    renderer = StreamRenderer(Console(file=output, width=20000, force_terminal=True), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='x' * 10000 + '\n')))
    await writing.wait()
    await renderer.abort()
    shown = output.getvalue().count('x')
    await renderer.finish()
    assert 0 < shown == output.getvalue().count('x') < 10000


async def test_smoothing_defaults_match_code_puppy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin smooth_stream.py defaults from Code Puppy a862bf478b63."""
    observed: list[tuple[float, float, int]] = []

    def writer(
        target: IO[str], *, tick_interval: float, catch_up_seconds: float, min_chars_per_tick: int
    ) -> PartWriter:
        observed.append((tick_interval, catch_up_seconds, min_chars_per_tick))
        return PartWriter(
            target,
            tick_interval=tick_interval,
            catch_up_seconds=catch_up_seconds,
            min_chars_per_tick=min_chars_per_tick,
        )

    monkeypatch.setattr('pydantic_ai_pax.ui.rendering.rendering.PartWriter', writer)
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
    renderer = StreamRenderer(console, stop_loading=lambda: None, smooth=False)
    words = ' '.join(f'word{index}' for index in range(30))
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content=f'{words}\n')))
    streaming = _rows(surface, width=40)
    await renderer.finish()
    wide = _rows(surface, width=120)
    assert wide[0].startswith('Thinking word0')
    narrow = _rows(surface, width=40)
    assert len(narrow) > len(wide)
    assert all(len(row) <= 40 for row in narrow)
    assert ''.join(''.join(narrow).split()) == 'Thinking' + ''.join(words.split())
    replay = TranscriptBuffer()
    replay.write(render_markdown(source=words, width=40, thinking=True, colors='truecolor'))
    rendered = [Text.from_ansi(row).plain for row in replay.frame(width=40, height=50).rows]
    assert narrow[: len(rendered) - 1] == rendered[:-1], 'the ended part renders again'
    assert [len(row) for row in streaming[:3]] == [40] * 3, 'a live part cuts its streamed rows at 40 instead'

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


def test_smoothing_types_a_preview_but_writes_its_replacement_at_once() -> None:
    """Retyping a finished code block from its first character would blank the preview the user was reading."""
    writes: list[str] = []

    class ObservedOutput(io.StringIO):
        def write(self, text: str) -> int:
            writes.append(text)
            return super().write(text)

    writer = PartWriter(ObservedOutput(), tick_interval=1, catch_up_seconds=1, min_chars_per_tick=1)
    writer.write('preview')
    writer.commit('the rendered block\n')
    assert writes == ['preview', 'the rendered block\n'], 'what was still being typed goes first, then the block'
