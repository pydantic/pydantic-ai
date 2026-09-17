"""A terminal-only status row, separate from conversation output."""

import asyncio
import contextlib
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Self

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.messages import (
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from rich.console import Console

from . import theme
from .usage_report import format_cost


@dataclass(kw_only=True)
class Status:
    """Reported context and explicitly approximate live output counts."""

    model: str = 'agent default'
    context_tokens: int | None = None
    context_alert: bool = False
    """Paint the context figure `WARNING`; set by whoever knows the window, such as the `compaction` plugin."""
    output_tokens: int | None = None
    cost: Decimal | None = None
    """Retained-history cost; `None` (hidden) until a priced response exists."""
    streamed_chars: int = 0
    activity: str = 'ready'

    def observe(self, event: AgentStreamEvent) -> None:
        """Include text, thinking, and streamed tool arguments in the estimate."""
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, (TextPart, ThinkingPart)):
                self.streamed_chars += len(part.content)
                self.activity = 'thinking' if isinstance(part, ThinkingPart) else 'responding'
            elif isinstance(part, ToolCallPart):
                self.streamed_chars += len(part.args_as_json_str())
                self.activity = f'tool: {part.tool_name}'
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, (TextPartDelta, ThinkingPartDelta)):
                self.streamed_chars += len(delta.content_delta or '')
            elif isinstance(delta, ToolCallPartDelta) and isinstance(delta.args_delta, str):
                self.streamed_chars += len(delta.args_delta)
        elif isinstance(event, FunctionToolCallEvent):
            self.activity = f'running: {event.part.tool_name}'
        elif isinstance(event, FunctionToolResultEvent):
            self.activity = 'working'

    def segments(self, frame: str = '') -> tuple[str, str, str]:
        """The row as (head, context figure, tail), so the figure can be painted on its own."""
        context = '?' if self.context_tokens is None else f'{self.context_tokens:,}'
        output = f'~{math.ceil(self.streamed_chars / 4):,} streamed tokens'
        if self.output_tokens is not None:
            output = f'{self.output_tokens:,} output tokens'
        cost = '' if self.cost is None else f' | {format_cost(self.cost)}'
        return f'{frame} {self.model} | context: '.lstrip(), context, f' tokens | {output}{cost} | {self.activity}'

    def text(self, frame: str = '') -> str:
        """Use no percentage when the model's context capacity is unknown."""
        return ''.join(self.segments(frame))

    def toolbar(self) -> list[tuple[str, str]]:
        """prompt-toolkit fragments for the input prompt; the figure is `WARNING` while `context_alert` is set."""
        head, figure, tail = self.segments()
        return [('', head), (theme.WARNING if self.context_alert else '', figure), ('', tail)]


class StatusLine:
    """Reserve the last row while a run owns the terminal; restore it on exit."""

    def __init__(self, console: Console, status: Status) -> None:
        """Bind the footer to the same output stream as the renderer."""
        self.console = console
        self.status = status
        self._task: asyncio.Task[None] | None = None
        self._height = 0

    async def __aenter__(self) -> Self:
        """Reserve a row only on an interactive terminal."""
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            self.console.show_cursor(False)
            self._draw(0)
            self._task = asyncio.create_task(self._animate())
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Restore scrolling on success, failure, and cancellation."""
        if self._task is not None:
            self._task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            finally:
                self.console.file.write(f'\x1b7\x1b[r\x1b[{self._height};1H\x1b[2K\x1b8')
                self.console.show_cursor(True)
                self.console.file.flush()

    def _draw(self, frame: int) -> None:
        width, height = self.console.size
        if height < 3:
            return
        # Leave one column unused so the footer cannot trigger autowrap.
        head, figure, tail = (_printable(segment) for segment in self.status.segments())
        text = head + figure + tail
        alerted = range(len(head), len(head) + len(figure)) if self.status.context_alert else range(0)
        prefix = '\x1b7'
        if height != self._height:
            # After a prompt the cursor is usually on the last row. Index down and back up first,
            # so the cursor is inside the region before the margins exclude that row; a linefeed
            # from outside the region makes terminals either overwrite the footer or scroll it away.
            prefix = f'\x1bD\x1b[1A\x1b7\x1b[1;{height - 1}r'
            self._height = height
        text = text[: max(0, width - 1)]
        highlight = frame % (len(text) + 12) - 6
        shades = tuple(theme.sgr(color) for color in (theme.SUGAR, theme.LIGHT_PURPLE, theme.LITHIUM, theme.PURPLE))
        warning = theme.sgr(theme.WARNING)
        painted = ''.join(
            (warning if index in alerted else shades[min(abs(index - highlight) // 2, 3)]) + char
            for index, char in enumerate(text)
        )
        self.console.file.write(f'{prefix}\x1b[{height};1H\x1b[2K{painted}\x1b[0m\x1b8')
        self.console.file.flush()

    async def _animate(self) -> None:
        frame = 0
        while True:
            self._draw(frame)
            frame += 1
            await asyncio.sleep(0.1)


def _printable(text: str) -> str:
    return ''.join(char if char.isascii() and char.isprintable() else '?' for char in text)
