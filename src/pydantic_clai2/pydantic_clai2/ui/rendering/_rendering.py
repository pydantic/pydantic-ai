"""Incremental Markdown rendering for native Pydantic AI events."""

import io
import re
from collections.abc import Callable, Sequence
from typing import IO

import anyio
from rich.console import Console, RenderableType
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from termflow import Parser, Renderer
from termflow.ansi import UNDERLINE_OFF, UNDERLINE_ON, fg_color, make_link
from termflow.parser.events import (
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    ParseEvent,
)
from termflow.parser.inline import CODE_SPAN_RE, IMAGE_RE, LINK_RE
from termflow.render.style import RenderFeatures, RenderStyle
from termflow.stream import SmoothWriter
from termflow.syntax import LANGUAGE_ALIASES

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
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent, SandboxCallOrder
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.grep_output import GrepOutput
from pydantic_clai2.ui.rendering.tool_output import ToolOutput, print_tool_header, terminal_text, tool_arguments_text


def markdown_style() -> RenderStyle:
    """Keep the existing Markdown colours unless a Termflow palette is selected."""
    palette = theme.current()
    if palette is not None:
        return palette.to_render_style()
    return RenderStyle(
        bright=theme.LITHIUM,
        head=theme.PURPLE,
        symbol=theme.CALCIUM,
        grey=theme.GREY,
        dark=theme.DARK_PURPLE,
        mid=theme.ELEMENT_PURPLE,
        light=theme.GREY,
        link=theme.AQUA,
        error=theme.CALCIUM,
    )


_URL_RE = re.compile(r'<(https?://[^\s<>]+)>|(?<![\w/])https?://[^\s<>`]+')
_URL_START, _URL_END = '\ue000', '\ue001'
"""Private-use marks around a URL while Termflow formats the line; `terminal_text` escapes any from the model."""
_MARKED_URL_RE = re.compile(f'{_URL_START}([^{_URL_END}]*){_URL_END}')


def _trim_url(url: str) -> str:
    """Leave trailing punctuation and unbalanced closing brackets out of a bare URL, as GFM does."""
    while url[-1] in '.,;:!?\'"*_~)]':
        if url[-1] == ')' and url.count('(') >= url.count(')'):
            break
        url = url[:-1]
    return url


class MarkdownRenderer(Renderer):
    """Termflow's renderer, also highlighting bare `https://` and `<https://...>` URLs as links."""

    def _format_inline(self, text: str) -> str:
        taken = [match.span() for pattern in (CODE_SPAN_RE, IMAGE_RE, LINK_RE) for match in pattern.finditer(text)]

        def mark(match: re.Match[str]) -> str:
            if any(start <= match.start() < end for start, end in taken):
                return match[0]
            url = match[1] or _trim_url(match[0])
            rest = '' if match[1] else match[0][len(url) :]
            return f'{_URL_START}{url}{_URL_END}{rest}'

        return _MARKED_URL_RE.sub(self._link, super()._format_inline(_URL_RE.sub(mark, text)))

    def _link(self, match: re.Match[str]) -> str:
        url = match[1]
        if '\x1b' in url:  # Other formatting split the URL; leave it as Termflow styled it.
            return url
        # Reset only the foreground so surrounding bold and thinking dim continue after the link.
        label = f'{fg_color(self.style.link)}{url}\x1b[39m'
        return make_link(url, label) if self.features.hyperlinks else f'{UNDERLINE_ON}{label}{UNDERLINE_OFF}'


class LinkOutput(io.StringIO):
    """Scope streamed hyperlinks to each write so editor paints and aborts stay unlinked."""

    def __init__(self, *, output: IO[str]) -> None:
        """Wrap one part's output without closing the underlying destination."""
        super().__init__()
        self.output = output
        self._link = ''

    def write(self, text: str) -> int:
        """SmoothWriter supplies whole ANSI tokens, but can split a link's label."""
        length = len(text)
        prefix = self._link
        text = re.sub(r'\x1b\]8;;([^\x1b]*)\x1b\\', self._track_link, text)
        self.output.write(prefix + text + ('\x1b]8;;\x1b\\' if self._link else ''))
        return length

    def _track_link(self, match: re.Match[str]) -> str:
        # Smoothing repeats metadata per chunk. Cap the destination so large
        # model-generated URLs cannot amplify terminal output without bound.
        self._link = match[0] if 0 < len(match[1]) <= 2048 else ''
        return self._link or '\x1b]8;;\x1b\\'

    def flush(self) -> None:
        """Forward flushes without taking ownership of the terminal."""
        self.output.flush()


class StreamRenderer:
    """Stream text and dimmed reasoning through the same Markdown pipeline."""

    def __init__(
        self,
        console: Console,
        *,
        stop_loading: Callable[[], None],
        show_thinking: bool = True,
        smooth_seconds: float = 0.5,
        show_tool_output: bool = False,
        shell_lines: int = 20,
        grep_lines: int = 20,
        tool_arg_chars: int = 40,
        renderers: Sequence[Callable[[AgentStreamEvent], RenderableType | None]] = (),
    ) -> None:
        self.console = console
        self._renderers = tuple(renderers)
        self._sandbox_calls = SandboxCallOrder()
        self.show_tool_output = show_tool_output
        self.tool_arg_chars = tool_arg_chars
        self._tool_output = ToolOutput(console, shell_lines=shell_lines, show_output=show_tool_output)
        self._grep_output = GrepOutput(console, lines=grep_lines, show_output=show_tool_output)
        self.smooth_seconds = smooth_seconds
        self._thinking = False
        self._heading_printed = False
        self.show_thinking = show_thinking
        self.stop_loading = stop_loading
        self._writer: SmoothWriter | None = None
        self._parser: Parser | None = None
        self._renderer: Renderer | None = None
        self._buffer = ''
        self._code_lines: list[str] = []
        self._code_language = 'text'
        self._index: int | None = None
        self.rendered_text = False

    async def on_stream_event(self, event: AgentStreamEvent) -> None:
        """Bind this callback to `Session.on_stream_event`."""
        if await self._render_sandbox_call(event):
            return
        if await self._render_with_plugins(event):
            if isinstance(event, PartStartEvent) and isinstance(event.part, (TextPart, ThinkingPart)):
                self._thinking = isinstance(event.part, ThinkingPart)
                self._index = event.index
                self._start_part()
                if isinstance(event.part, TextPart):
                    self.rendered_text = True
            return
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
            await self._render_tool(event)

    async def _render_sandbox_call(self, event: AgentStreamEvent) -> bool:
        """Render a call from inside `run_code` like a direct one, under its `run_code` header."""
        tool_events = self._sandbox_calls.tool_events(event)
        if tool_events is None:
            return False
        for tool_event in tool_events:
            if not await self._render_with_plugins(tool_event):
                await self._render_tool(tool_event)
        return True

    async def _render_tool(self, event: FunctionToolCallEvent | FunctionToolResultEvent) -> None:
        await self.finish()
        self.stop_loading()
        self._render_tool_event(event)

    def _render_tool_event(self, event: FunctionToolCallEvent | FunctionToolResultEvent) -> None:
        if isinstance(event, DelegationToolCallEvent):
            return
        if isinstance(event, FunctionToolResultEvent):
            self._tool_output.discard_call(event.part.tool_call_id)
        if self._grep_output.render(event):
            return
        if isinstance(event, FunctionToolCallEvent) and not self._tool_output.render_call(event):
            name = ''.join(char if char.isprintable() else ' ' for char in event.part.tool_name)
            arguments = tool_arguments_text(event.part.args_as_dict(), max_chars=self.tool_arg_chars)
            print_tool_header(self.console, name=name, argument=arguments)

    async def _render_with_plugins(self, event: AgentStreamEvent) -> bool:
        for renderer in self._renderers:
            renderable = renderer(event)
            if renderable is not None:
                await self.finish()
                self.stop_loading()
                self.console.print(renderable)
                self.console.print()
                return True
        return False

    def _start_part(self) -> None:
        self._parser = Parser()
        if self.console.is_terminal:
            self._writer = self._make_writer()
            self._writer.start()
        self._renderer = MarkdownRenderer(
            output=self._writer or self.console.file,  # pyright: ignore[reportArgumentType]
            width=self.console.width,
            style=markdown_style(),
            features=RenderFeatures(clipboard=False, hyperlinks=self.console.is_terminal, images=False),
            dim=self._thinking,
        )

    def _make_writer(self) -> SmoothWriter:
        """Reasoning keeps Code Puppy's slower thinking pace; responses use the configured catch-up."""
        output = LinkOutput(output=self.console.file)
        if self._thinking:
            return SmoothWriter(output, tick_interval=0.02, catch_up_seconds=0.4, min_chars_per_tick=2)
        return SmoothWriter(output, tick_interval=0.012, catch_up_seconds=self.smooth_seconds, min_chars_per_tick=1)

    def _feed(self, content: str) -> None:
        content = terminal_text(content)
        if content and not self._heading_printed:
            if self._thinking:
                # No newline: the rendered reasoning continues on the heading's line.
                self.console.print('Thinking ', style=theme.color(theme.THINKING), end='')
            self._heading_printed = True
        self._buffer += content
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._line(line)

    def _line(self, line: str) -> None:
        assert self._parser is not None and self._renderer is not None
        self._render_events(self._parser.parse_line(line))

    def _render_events(self, events: list[ParseEvent]) -> None:
        assert self._renderer is not None
        for event in events:
            if isinstance(event, CodeBlockStartEvent):
                self._code_language = (event.language or 'text').split()[0]
                self._code_lines = []
            elif isinstance(event, CodeBlockLineEvent):
                self._code_lines.append(event.line)
            elif isinstance(event, CodeBlockEndEvent):
                # Lex the whole fence so multiline strings and comments keep their state.
                with self.console.capture() as capture:
                    self.console.rule(Text(self._code_language), align='left', style=theme.color(theme.MUTED))
                    self.console.print(
                        Syntax(
                            '\n'.join(self._code_lines),
                            LANGUAGE_ALIASES.get(self._code_language.lower(), self._code_language.lower()),
                            theme=theme.syntax_theme(),
                            background_color='default',
                            word_wrap=True,
                        ),
                        style=Style(dim=self._thinking),
                    )
                    self.console.rule(style=theme.color(theme.MUTED))
                (self._writer or self.console.file).write(capture.get())
                self._code_lines = []
            else:
                self._renderer.render(event)

    async def finish(self) -> None:
        """Drain rendered Markdown before the next part, tool, or prompt appears."""
        if self._buffer:
            self._line(self._buffer)
        if self._parser is not None and self._renderer is not None:
            self._render_events(self._parser.finalize())
        writer, self._writer = self._writer, None
        visible = self._heading_printed
        self._reset()
        if writer is not None:
            await writer.close()
        if visible:
            self.console.print()
        self.console.file.flush()

    async def abort(self) -> None:
        """Discard pending output on cancellation and let the drainer terminate."""
        self._tool_output.abort()
        writer, self._writer = self._writer, None
        self._reset()
        if writer is not None:
            writer.abort()
        await anyio.sleep(0)

    def _reset(self) -> None:
        self._code_lines = []
        self._code_language = 'text'
        self._heading_printed = False
        self._buffer = ''
        self._parser = None
        self._renderer = None
        self._index = None
