"""Incremental Markdown rendering for native Pydantic AI events."""

import asyncio
from collections.abc import Callable

from pydantic_ai import (
    AgentStreamEvent,
    CapabilityEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from rich.console import Console
from termflow import Parser, Renderer  # pyright: ignore[reportMissingTypeStubs]
from termflow.render.style import RenderFeatures, RenderStyle  # pyright: ignore[reportMissingTypeStubs]
from termflow.stream import SmoothWriter, StreamSmoother  # pyright: ignore[reportMissingTypeStubs]

from .grep_output import GrepOutput
from .tool_output import ToolOutput, terminal_text


class StreamRenderer:
    """Render text and thinking separately, flushing Markdown at part boundaries."""

    def __init__(
        self,
        console: Console,
        *,
        stop_loading: Callable[[], None],
        show_thinking: bool = True,
        smooth_seconds: float = 0.5,
        shell_lines: int = 20,
        grep_lines: int = 20,
    ) -> None:
        self.console = console
        self._tool_output = ToolOutput(console, shell_lines=shell_lines)
        self._grep_output = GrepOutput(console, lines=grep_lines)
        self.smooth_seconds = smooth_seconds
        self._thinking = False
        self._heading_printed = False
        self.show_thinking = show_thinking
        self.stop_loading = stop_loading
        self._writer: SmoothWriter | None = None
        self._thinking_writer: StreamSmoother | None = None
        self._parser: Parser | None = None
        self._renderer: Renderer | None = None
        self._buffer = ''
        self._index: int | None = None
        self.rendered_text = False

    async def on_stream_event(self, event: AgentStreamEvent) -> None:
        """Bind this callback to `Session.on_stream_event`."""
        if isinstance(event, CapabilityEvent):
            await self.finish()
            if self._tool_output.render(event):
                return
        if isinstance(event, PartStartEvent) and isinstance(event.part, (TextPart, ThinkingPart)):
            await self.finish()
            self.stop_loading()
            thinking = isinstance(event.part, ThinkingPart)
            if thinking and not self.show_thinking:
                return
            self._thinking = thinking
            self._index = event.index
            self._start_part()
            self._feed(event.part.content)
            if not thinking:
                self.rendered_text = True
        elif isinstance(event, PartDeltaEvent) and event.index == self._index:
            if isinstance(event.delta, TextPartDelta):
                self._feed(event.delta.content_delta)
            elif isinstance(event.delta, ThinkingPartDelta):
                self._feed(event.delta.content_delta or '')
        elif isinstance(event, PartStartEvent) or isinstance(event, PartEndEvent) and event.index == self._index:
            await self.finish()
        elif isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent)):
            await self.finish()
            self.stop_loading()
            if isinstance(event, FunctionToolResultEvent):
                self._tool_output.discard_call(event.part.tool_call_id)
            if self._grep_output.render(event):
                return
            if isinstance(event, FunctionToolCallEvent):
                if self._tool_output.render_call(event):
                    return
                name = ''.join(char if char.isprintable() else ' ' for char in event.part.tool_name)
                self.console.print(
                    f'● {name}', style='dim', markup=False, highlight=False, overflow='ellipsis', no_wrap=True
                )
                self.console.print()

    def _start_part(self) -> None:
        if self._thinking:
            if self.console.is_terminal:
                self._thinking_writer = StreamSmoother(
                    self._emit_thinking, tick_interval=0.02, catch_up_seconds=0.4, min_chars_per_tick=2
                )
                self._thinking_writer.start()
            return
        self._parser = Parser()
        if self.console.is_terminal:
            self._writer = SmoothWriter(
                self.console.file, tick_interval=0.012, catch_up_seconds=self.smooth_seconds, min_chars_per_tick=1
            )
            self._writer.start()
        self._renderer = Renderer(
            output=self._writer or self.console.file,  # pyright: ignore[reportArgumentType]
            width=self.console.width,
            style=RenderStyle.dracula(),
            features=RenderFeatures(clipboard=False, hyperlinks=False, images=False),
        )

    def _emit_thinking(self, content: str) -> None:
        self.console.print(content, style='dim', end='', markup=False, highlight=False)

    def _feed(self, content: str) -> None:
        content = terminal_text(content)
        if content and not self._heading_printed:
            if self._thinking:
                self.console.print('Thinking', style='dim cyan')
            self._heading_printed = True
        if self._thinking:
            if self._thinking_writer is not None:
                self._thinking_writer.feed(content)
            else:
                self._emit_thinking(content)
            return
        self._buffer += content
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._line(line)

    def _line(self, line: str) -> None:
        assert self._parser is not None and self._renderer is not None
        self._renderer.render_all(self._parser.parse_line(line))

    async def finish(self) -> None:
        """Drain rendered Markdown before the next part, tool, or prompt appears."""
        if self._buffer:
            self._line(self._buffer)
        if self._parser is not None and self._renderer is not None:
            self._renderer.render_all(self._parser.finalize())
        writer, self._writer = self._writer, None
        thinking_writer, self._thinking_writer = self._thinking_writer, None
        visible = self._heading_printed
        thinking_visible = self._thinking and visible
        self._reset()
        if writer is not None:
            await writer.close()
        if thinking_writer is not None:
            await thinking_writer.close()
        if thinking_visible:
            self.console.print()
        if visible:
            self.console.print()
        self.console.file.flush()

    async def abort(self) -> None:
        """Discard pending output on cancellation and let the drainer terminate."""
        self._tool_output.abort()
        writer, self._writer = self._writer, None
        thinking_writer, self._thinking_writer = self._thinking_writer, None
        self._reset()
        if writer is not None:
            writer.abort()
        if thinking_writer is not None:
            thinking_writer.abort()
        await asyncio.sleep(0)

    def _reset(self) -> None:
        self._heading_printed = False
        self._buffer = ''
        self._parser = None
        self._renderer = None
        self._index = None
