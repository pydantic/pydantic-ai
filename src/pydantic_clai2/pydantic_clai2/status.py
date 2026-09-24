"""A terminal-only prompt frame and status row, separate from conversation output."""

import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Self, cast

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
from .tool_output import terminal_text
from .usage_report import format_cost

StatusSegment = Callable[[], str]
"""One short status-row fragment supplied by a plugin; see `PluginHost.status_segment`."""


@dataclass(kw_only=True)
class Status:
    """Reported context and explicitly approximate live output counts."""

    model: str = 'agent default'
    workspace: str = ''
    """The session's working directory; hidden while empty."""
    context_tokens: int | None = None
    context_alert: bool = False
    """Paint the context figure `WARNING`; set by whoever knows the window, such as the `compaction` plugin."""
    output_tokens: int | None = None
    cost: Decimal | None = None
    """Retained-history cost; `None` (hidden) until a priced response exists."""
    streamed_chars: int = 0
    activity: str = 'ready'
    status_segments: tuple[StatusSegment, ...] = ()
    """Plugin fragments appended after the built-in figures; the shell fills this in each turn."""

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

    def segments(self, frame: str = '') -> tuple[str, str, str, str]:
        """The row as (head, context figure, tail, plugins), so the figure and fragments paint on their own."""
        context = '?' if self.context_tokens is None else f'{self.context_tokens:,}'
        output = f'~{math.ceil(self.streamed_chars / 4):,} streamed tokens'
        if self.output_tokens is not None:
            output = f'{self.output_tokens:,} output tokens'
        cost = '' if self.cost is None else f' | {format_cost(self.cost)}'
        # A POSIX directory name may hold a newline or an escape sequence; keep it inert on every painter.
        workspace = f' | {_short_path(terminal_text(self.workspace, keep=""))}' if self.workspace else ''
        head = f'{frame} {self.model}{workspace} | context: '.lstrip()
        return head, context, f' tokens | {output}{cost} | {self.activity}', self._plugin_text()

    def _plugin_text(self) -> str:
        """Plugin fragments, separated and prefixed; a fragment that raises or returns a non-string is reported."""
        shown: list[str] = []
        for segment in self.status_segments:
            try:
                # Plugins may be untyped, so validate the runtime result despite the callable's contract.
                text = cast(object, segment())
                if not isinstance(text, str):
                    # A non-string cannot be joined; report its type instead of raising out of the painter.
                    text = f'!{type(text).__name__}' if text else ''
                if text:
                    # Sanitized here rather than only in the row painter: the toolbar draws fragments too.
                    shown.append(_printable(text))
            except Exception as exc:  # noqa: BLE001 -- a plugin fragment must not take down the footer.
                shown.append(f'!{type(exc).__name__}')
        return '' if not shown else ' | ' + ' | '.join(shown)

    def text(self, frame: str = '') -> str:
        """Use no percentage when the model's context capacity is unknown."""
        return ''.join(self.segments(frame))

    def toolbar(self) -> list[tuple[str, str]]:
        """prompt-toolkit fragments for the input prompt; the figure is `WARNING` while `context_alert` is set."""
        head, figure, tail, plugins = self.segments()
        painted = [('', head), (theme.WARNING if self.context_alert else '', figure), ('', tail)]
        return [*painted, (theme.MUTED, plugins)] if plugins else painted


def _interrupted() -> bool:
    """Whether the current task was itself cancelled while it waited on the animation task."""
    current = asyncio.current_task()
    return current is not None and current.cancelling() > 0


class StatusLine:
    """Keep the prompt frame and status visible below streamed output during a run."""

    def __init__(self, console: Console, status: Status, *, enabled: bool = True) -> None:
        """Bind the footer to the same output stream as the renderer."""
        self.console = console
        self.status = status
        self.enabled = enabled
        self._task: asyncio.Task[None] | None = None
        self._height = 0
        self._rows = 0

    async def __aenter__(self) -> Self:
        """Reserve the prompt area only on an interactive terminal."""
        self._reserve()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Restore scrolling on success, failure, and cancellation."""
        await self._release()

    @contextlib.asynccontextmanager
    async def paused(self) -> AsyncGenerator[None]:
        """Give the whole screen to something else, then restore the prompt area."""
        await self._release()
        try:
            yield
        finally:
            self._reserve()

    def _reserve(self) -> None:
        if self.enabled and self.console.is_terminal and not self.console.is_dumb_terminal:
            self.console.show_cursor(False)
            self._draw(0)
            self._task = asyncio.create_task(self._animate())

    async def _release(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                if _interrupted():
                    # The CancelledError was ours, not the animation's: Ctrl-C must still abort.
                    raise asyncio.CancelledError
            finally:
                self._clear()
                self.console.show_cursor(True)
                self.console.file.flush()

    def _clear(self) -> None:
        if self._height:
            rows = range(max(1, self._height - self._rows + 1), min(self._height, self.console.height) + 1)
            cleared = ''.join(f'\x1b[{row};1H\x1b[2K' for row in rows)
            self.console.file.write(f'\x1b7\x1b[r{cleared}\x1b8')
            self._height = self._rows = 0

    def _draw(self, frame: int) -> None:
        width, height = self.console.size
        if height < 3:
            self._clear()
            self.console.file.flush()
            return
        rows = 4 if height >= 6 and width >= 4 else 1
        # Leave one column unused so the footer cannot trigger autowrap.
        head, figure, tail, plugins = (_printable(segment) for segment in self.status.segments())
        text = head + figure + tail + plugins
        # From the untruncated row, so a fragment wider than the terminal cannot mute the built-in part.
        plugin_start = max(0, len(text) - len(plugins))
        alerted = range(len(head), len(head) + len(figure)) if self.status.context_alert else range(0)
        prefix = '\x1b7'
        if (height, rows) != (self._height, self._rows):
            self._clear()
            # Move inside the new scroll region before excluding the footer rows.
            prefix = '\x1bD' * rows + f'\x1b[{rows}A\x1b7\x1b[1;{height - rows}r'
            self._height, self._rows = height, rows
        text = text[: max(0, width - 1)]
        highlight = frame % (len(text) + 12) - 6
        shades = tuple(theme.sgr(color) for color in (theme.SUGAR, theme.LIGHT_PURPLE, theme.LITHIUM, theme.PURPLE))
        warning = theme.sgr(theme.WARNING)
        muted = theme.sgr(theme.MUTED)

        def paint(index: int) -> str:
            if index in alerted:
                return warning
            return muted if index >= plugin_start else shades[min(abs(index - highlight) // 2, 3)]

        painted = ''.join(paint(index) + char for index, char in enumerate(text))
        if rows == 4:
            inner_width = width - 3
            hint = '> Working... Ctrl-C to interrupt'[:inner_width].ljust(inner_width)
            border = '─' * inner_width
            for row, line in enumerate((f'┌{border}┐', f'│{hint}│', f'└{border}┘'), start=height - 3):
                prefix += f'\x1b[{row};1H\x1b[2K{theme.sgr(theme.MUTED)}{line}\x1b[0m'
        self.console.file.write(f'{prefix}\x1b[{height};1H\x1b[2K{painted}\x1b[0m\x1b8')
        self.console.file.flush()

    async def _animate(self) -> None:
        frame = 0
        while True:
            self._draw(frame)
            frame += 1
            await asyncio.sleep(0.1)


def _short_path(path: str, limit: int = 40) -> str:
    """Abbreviate the home directory, then keep the path's tail, the part that tells worktrees apart."""
    # `Path.home()` raises `RuntimeError` when the account has no resolvable home directory.
    with contextlib.suppress(ValueError, RuntimeError):
        path = str(Path('~', Path(path).relative_to(Path.home())))
    return path if len(path) <= limit else '…' + path[-(limit - 1) :]


def _printable(text: str) -> str:
    return ''.join(char if char.isascii() and char.isprintable() else '?' for char in text)
