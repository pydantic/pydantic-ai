"""A terminal-only prompt frame and status row, separate from conversation output."""

import asyncio
import contextlib
import math
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Self, cast

import anyio
from rich.cells import set_cell_size
from rich.console import Console

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.messages import (
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.spinners import BUILTIN_SPINNERS, DEFAULT_SPINNER, Spinner
from pydantic_clai2.ui.rendering.tool_output import terminal_text
from pydantic_clai2.ui.rendering.usage_report import format_cost

StatusSegment = Callable[[], str]
"""One short status-row fragment supplied by a plugin; see `PluginHost.status_segment`."""


@dataclass(kw_only=True)
class Status:
    """Reported context and explicitly approximate live output counts."""

    model: str = 'agent default'
    workspace: str = ''
    """The session's working directory; hidden while empty."""
    context_tokens: int | None = None
    context_window: int | None = None
    """Known context capacity for the latest request; `None` when unresolved."""
    context_alert: bool = False
    """Paint the context figure `WARNING`; set by whoever knows the window, such as the `compaction` plugin."""
    output_tokens: int | None = None
    cost: Decimal | None = None
    """Retained-history cost; `None` (hidden) until a priced response exists."""
    streamed_chars: int = 0
    activity: str = 'ready'
    status_segments: tuple[StatusSegment, ...] = ()
    """Plugin fragments appended after the built-in figures; the shell fills this in each turn."""
    agent: str = ''
    """The subagent these figures belong to, painted before the model; empty for the main conversation."""
    subagent: 'Callable[[], Status | None]' = lambda: None
    """The running subagent whose figures replace these on the row until it settles; see `Tasks.focused`."""

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

    def segments(self, frame: str = '') -> tuple[str, str, str, str, str, str, str]:
        """Separate figures from labels so plain text and both painters share the same boundaries.

        While a subagent runs, its figures replace these; the plugin fragments stay.
        """
        return (self.subagent() or self)._segments(frame, self._plugin_text())

    def _segments(self, frame: str, plugins: str) -> tuple[str, str, str, str, str, str, str]:
        context = f'{_compact_tokens(self.context_tokens)}/{_compact_tokens(self.context_window)}'
        output = f'~{math.ceil(self.streamed_chars / 4):,}'
        output_label = 'streamed tokens'
        if self.output_tokens is not None:
            output = f'{self.output_tokens:,}'
            output_label = 'output tokens'
        cost = '' if self.cost is None else f' | {format_cost(self.cost)}'
        # A POSIX directory name may hold a newline or an escape sequence; keep it inert on every painter.
        workspace = f' | {_short_path(terminal_text(self.workspace, keep=""))}' if self.workspace else ''
        label = ': '.join(part for part in (self.agent, self.model) if part)
        head = f'{frame} {label}{workspace} | context: '.lstrip()
        return head, context, ' tokens | ', output, f' {output_label}{cost} | ', self.activity, plugins

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
        """Show compact used/max context, with `?` for unknown counts."""
        return ''.join(self.segments(frame))

    def toolbar(self) -> list[tuple[str, str]]:
        """Accent output counts, tool names, and a running subagent, keeping chrome muted and context alerts visible."""
        shown = self.subagent() or self
        head, figure, prefix, count, details, activity, plugins = shown._segments('', self._plugin_text())
        named = [(theme.INFO, shown.agent)] if shown.agent else []
        painted = [
            *named,
            (theme.MUTED, head.removeprefix(shown.agent)),
            (theme.WARNING if shown.context_alert else theme.MUTED, figure),
            (theme.MUTED, prefix),
            (theme.LITHIUM, count),
            (theme.MUTED, details),
        ]
        if activity.startswith(('tool: ', 'running: ')):
            label, _, tool = activity.partition(': ')
            painted.extend([(theme.MUTED, label + ': '), (theme.THINKING, tool)])
        else:
            painted.append((theme.MUTED, activity))
        return [*painted, (theme.MUTED, plugins)] if plugins else painted


class StatusLine:
    """Keep the prompt frame and status visible below streamed output during a run."""

    def __init__(
        self,
        console: Console,
        status: Status,
        *,
        enabled: bool = True,
        spinner: Callable[[], Spinner] = lambda: BUILTIN_SPINNERS[DEFAULT_SPINNER],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the footer to the same output stream as the renderer; `spinner` is read on every frame."""
        self.console = console
        self.status = status
        self.enabled = enabled
        self.spinner = spinner
        self.clock = clock
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
            self._draw()
            self._task = asyncio.create_task(self._animate())

    async def _release(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                # `asyncio.wait` never forwards our cancellation to the animation, so a
                # `CancelledError` here is ours and propagates: Ctrl-C must still abort.
                await asyncio.wait({task})
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()  # Surface an animation failure; its cancellation was ours.
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

    def _draw(self) -> None:
        width, height = self.console.size
        if height < 3:
            self._clear()
            self.console.file.flush()
            return
        rows = 4 if height >= 6 and width >= 4 else 1
        # Leave one column unused so the footer cannot trigger autowrap.
        fragments = self.status.toolbar()
        prefix = '\x1b7'
        if (height, rows) != (self._height, self._rows):
            self._clear()
            # Move inside the new scroll region before excluding the footer rows.
            prefix = '\x1bD' * rows + f'\x1b[{rows}A\x1b7\x1b[1;{height - rows}r'
            self._height, self._rows = height, rows
        remaining = max(0, width - 1)
        painted = ''
        for role, fragment in fragments:
            text = _printable(fragment)[:remaining]
            painted += ''.join(theme.sgr(role) + char for char in text)
            remaining -= len(text)
        if rows == 4:
            inner_width = width - 3
            glyph = self.spinner().frame(self.clock())
            hint = set_cell_size(f'> Working {glyph} Ctrl-C to interrupt', inner_width)
            border = '─' * inner_width
            for row, line in enumerate((f'┌{border}┐', f'│{hint}│', f'└{border}┘'), start=height - 3):
                prefix += f'\x1b[{row};1H\x1b[2K{theme.sgr(theme.MUTED)}{line}\x1b[0m'
        self.console.file.write(f'{prefix}\x1b[{height};1H\x1b[2K{painted}\x1b[0m\x1b8')
        self.console.file.flush()

    async def _animate(self) -> None:
        while True:
            # Refresh live counts at least ten times a second, whatever the spinner's speed.
            self._draw()
            await anyio.sleep(max(0, min(0.1, self.spinner().interval)))


def _compact_tokens(count: int | None) -> str:
    """Round to whole thousands or tenths of a million without trailing `.0`."""
    if count is None:
        return '?'
    if count >= 999_500:
        return f'{count / 1_000_000:.1f}'.removesuffix('.0') + 'm'
    if count >= 1_000:
        return f'{count / 1_000:.0f}k'
    return str(count)


def _short_path(path: str, limit: int = 40) -> str:
    """Abbreviate the home directory, then keep the path's tail, the part that tells worktrees apart."""
    # `Path.home()` raises `RuntimeError` when the account has no resolvable home directory.
    with contextlib.suppress(ValueError, RuntimeError):
        path = str(Path('~', Path(path).relative_to(Path.home())))
    return path if len(path) <= limit else '…' + path[-(limit - 1) :]


def _printable(text: str) -> str:
    return ''.join(char if char.isascii() and char.isprintable() else '?' for char in text)
