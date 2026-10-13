"""Incremental Markdown rendering for native Pydantic AI events."""

import io
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from typing import IO, Literal

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
    HeadingEvent,
    ParseEvent,
)
from termflow.parser.inline import CODE_SPAN_RE, IMAGE_RE, LINK_RE
from termflow.render.heading import _heading_codes  # pyright: ignore[reportPrivateUsage]
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
from pydantic_clai2.config import ToolCallDisplay
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent, SandboxCallOrder
from pydantic_clai2.ui.prompt.prompt_selection import trim_url
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import MarkdownBlock
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.grep_output import GrepOutput
from pydantic_clai2.ui.rendering.tool_group import ToolCallGroup
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
_URL_START, _URL_END = '\ufdd0', '\ufdd1'
"""Marks around a URL while Termflow formats the line.

Model text cannot forge them: `terminal_text` escapes these noncharacters, and the
HTML entity decoding Termflow applies turns `&#xFDD0;` into nothing.
"""
_MARKED_URL_RE = re.compile(f'{_URL_START}([^{_URL_END}]*){_URL_END}')


class MarkdownRenderer(Renderer):
    """Termflow's renderer, also highlighting bare `https://` and `<https://...>` URLs as links."""

    _enclosing = ''
    """Codes that restore the style a link interrupts, for headings Termflow styles around the inline text."""

    def render(self, event: ParseEvent) -> None:
        """Render one event, remembering a heading's style for links inside it."""
        self._enclosing = _heading_codes(event.level, self.style)[0] if isinstance(event, HeadingEvent) else ''
        super().render(event)

    def _format_inline(self, text: str) -> str:
        # Each pattern's matches are disjoint, so marking them is linear in the line length.
        taken = bytearray(len(text))
        for pattern in (CODE_SPAN_RE, IMAGE_RE, LINK_RE):
            for match in pattern.finditer(text):
                taken[match.start() : match.end()] = b'\x01' * (match.end() - match.start())

        def mark(match: re.Match[str]) -> str:
            if taken[match.start()]:
                return match[0]
            url = match[1] or trim_url(match[0])
            rest = '' if match[1] else match[0][len(url) :]
            return f'{_URL_START}{url}{_URL_END}{rest}'

        return _MARKED_URL_RE.sub(self._link, super()._format_inline(_URL_RE.sub(mark, text)))

    def _link(self, match: re.Match[str]) -> str:
        url = match[1]
        if '\x1b' in url:  # Other formatting split the URL; leave it as Termflow styled it.
            return url
        # Reset only the foreground so surrounding bold and thinking dim continue after the link.
        label = f'{fg_color(self.style.link)}{url}\x1b[39m'
        link = make_link(url, label) if self.features.hyperlinks else f'{UNDERLINE_ON}{label}{UNDERLINE_OFF}'
        return link + self._enclosing


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


def thinking_heading(console: Console) -> str:
    """The label reasoning starts with, styled for `console`."""
    with console.capture() as capture:
        console.print('Thinking ', style=theme.color(theme.THINKING), end='')
    return capture.get()


class MarkdownPipeline:
    """Termflow's line parser and renderer, with whole code fences highlighted by Rich."""

    def __init__(
        self,
        *,
        output: IO[str],
        console: Console,
        thinking: bool,
        hyperlinks: bool,
        continuation: tuple[Parser, str] | None = None,
    ) -> None:
        """Render to `output` at `console`'s width."""
        self.output = output
        self.console = console
        self.thinking = thinking
        self._parser = Parser()
        self._renderer = MarkdownRenderer(
            output=output,  # pyright: ignore[reportArgumentType] -- Termflow annotates TextIO but only writes and flushes.
            width=console.width,
            style=markdown_style(),
            features=RenderFeatures(clipboard=False, hyperlinks=hyperlinks, images=False),
            dim=thinking,
        )
        self._code_lines: list[str] = []
        self._code_language = 'text'
        if continuation is not None:
            self._parser, self._code_language = deepcopy(continuation)
        self._continued_fence = continuation is not None and self._parser.state.is_in_code()

    def continuation(self) -> tuple[Parser, str]:
        """Snapshot parsing context before finishing this display segment."""
        return deepcopy(self._parser), self._code_language

    def line(self, line: str) -> None:
        """Render one complete source line."""
        self._render_events(self._parser.parse_line(line))

    def finish(self) -> None:
        """Close any open block, such as an unterminated fence."""
        self._render_events(self._parser.finalize())

    def _render_events(self, events: list[ParseEvent]) -> None:
        for event in events:
            if isinstance(event, CodeBlockStartEvent):
                self._code_language = (event.language or 'text').split()[0]
                self._code_lines = []
                self._continued_fence = False
            elif isinstance(event, CodeBlockLineEvent):
                self._code_lines.append(event.line)
            elif isinstance(event, CodeBlockEndEvent):
                # Steering immediately before the closing fence has no code left to display.
                if self._continued_fence and not self._code_lines:
                    self._continued_fence = False
                    continue
                self._continued_fence = False
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
                        style=Style(dim=self.thinking),
                    )
                    self.console.rule(style=theme.color(theme.MUTED))
                self.output.write(capture.get())
                self._code_lines = []
            else:
                self._renderer.render(event)


ColorSystemName = Literal['standard', '256', 'truecolor', 'windows']


def color_system(console: Console) -> ColorSystemName | None:
    """The colour system `console` renders with; `None` when it renders no colour."""
    name = console.color_system
    return name if name in ('standard', '256', 'truecolor', 'windows') else None


def render_markdown(
    *,
    source: str,
    width: int,
    thinking: bool,
    colors: ColorSystemName | None,
    continuation: tuple[Parser, str] | None = None,
) -> str:
    """Render a whole part as the stream did, for a width or theme it was not streamed at.

    `colors` must be the stream console's colour system. Rich caches a style's ANSI codes on the
    shared style instance for the first colour system that renders it, so a replay in another
    system would change what the main console emits afterwards.
    """
    output = io.StringIO()
    console = Console(file=io.StringIO(), force_terminal=True, color_system=colors, width=width)
    if thinking and source:
        output.write(thinking_heading(console))
    markdown = MarkdownPipeline(
        output=LinkOutput(output=output),
        console=console,
        thinking=thinking,
        hyperlinks=True,
        continuation=continuation,
    )
    *lines, rest = source.split('\n')
    for line in lines:
        markdown.line(line)
    if rest:
        markdown.line(rest)
    markdown.finish()
    return output.getvalue()


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
        tool_calls: ToolCallDisplay = 'detailed',
        renderers: Sequence[Callable[[AgentStreamEvent], RenderableType | None]] = (),
        smooth: bool = True,
    ) -> None:
        """`smooth=False` writes each part at once, for history that has already streamed."""
        self.console = console
        self.smooth = smooth
        self._renderers = tuple(renderers)
        self._sandbox_calls = SandboxCallOrder()
        self.show_tool_output = show_tool_output
        self.tool_arg_chars = tool_arg_chars
        self._tool_output = ToolOutput(
            console, shell_lines=shell_lines, show_output=show_tool_output, tool_calls=tool_calls
        )
        self._grep_output = GrepOutput(console, lines=grep_lines, show_output=show_tool_output)
        self._group = ToolCallGroup(console, colors=color_system(console)) if tool_calls == 'grouped' else None
        self.smooth_seconds = smooth_seconds
        self._thinking = False
        self._heading_printed = False
        self.show_thinking = show_thinking
        self.stop_loading = stop_loading
        self._writer: SmoothWriter | None = None
        self._markdown: MarkdownPipeline | None = None
        self._block: MarkdownBlock | None = None
        self._buffer = ''
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
        if isinstance(event, CapabilityEvent) and await self._render_capability(event):
            return
        if isinstance(event, PartStartEvent) and isinstance(event.part, (TextPart, ThinkingPart)):
            await self._drain()
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
            await self._drain()
        elif isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent)):
            await self._render_tool(event)

    async def _render_capability(self, event: CapabilityEvent) -> bool:
        """Return whether the event was handled. A diff ends the tool-call group."""
        await self._drain()
        if self._group is not None and self._tool_output.prints_diff(event):
            self._group.close()
        return self._tool_output.render(event)

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
        await self._drain()
        self.stop_loading()
        self._render_tool_event(event)

    def _render_tool_event(self, event: FunctionToolCallEvent | FunctionToolResultEvent) -> None:
        if isinstance(event, DelegationToolCallEvent):
            return
        if isinstance(event, FunctionToolResultEvent):
            self._tool_output.discard_call(event.part.tool_call_id)
        if self._group is not None:
            if not isinstance(event, FunctionToolCallEvent):
                return
            if not self._tool_output.prints_diff(event):
                self._group.add(event.part.tool_name)
                return
            # A count would repeat the header its diff prints under.
            self._group.close()
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

    def _start_part(self, continuation: tuple[Parser, str] | None = None) -> None:
        surface = self.console.file
        self._block = (
            surface.markdown(
                render=partial(
                    render_markdown,
                    thinking=self._thinking,
                    colors=color_system(self.console),
                    continuation=continuation,
                ),
                width=self.console.width,
            )
            if isinstance(surface, PromptSurface)
            else None
        )
        output = self._block or self.console.file
        if self.console.is_terminal and self.smooth:
            self._writer = self._make_writer(output)
            self._writer.start()
        self._markdown = MarkdownPipeline(
            output=self._writer or output,  # pyright: ignore[reportArgumentType]
            console=self.console,
            thinking=self._thinking,
            hyperlinks=self.console.is_terminal,
            continuation=continuation,
        )

    def _make_writer(self, output: IO[str]) -> SmoothWriter:
        """Reasoning keeps Code Puppy's slower thinking pace; responses use the configured catch-up."""
        output = LinkOutput(output=output)
        if self._thinking:
            return SmoothWriter(output, tick_interval=0.02, catch_up_seconds=0.4, min_chars_per_tick=2)
        return SmoothWriter(output, tick_interval=0.012, catch_up_seconds=self.smooth_seconds, min_chars_per_tick=1)

    def _feed(self, content: str) -> None:
        content = terminal_text(content)
        if content and not self._heading_printed:
            # Only a part that shows something ends the group; reasoning can arrive with no text.
            self._close_group()
        if self._block is not None:
            self._block.extend(content)
        if content and not self._heading_printed:
            if self._thinking:
                # No newline: the rendered reasoning continues on the heading's line.
                (self._writer or self._block or self.console.file).write(thinking_heading(self.console))
            self._heading_printed = True
        self._buffer += content
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._line(line)

    def _line(self, line: str) -> None:
        assert self._markdown is not None
        self._markdown.line(line)

    async def echo_prompt(self, text: str) -> None:
        """Print a steering prompt between streamed chunks without ending the active part."""
        index, thinking = self._index, self._thinking
        if self._buffer:
            self._line(self._buffer)
            self._buffer = ''
        continuation = self._markdown.continuation() if self._markdown is not None else None
        await self.finish()
        self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
        self.console.print()
        if index is not None:
            self._index = index
            self._thinking = thinking
            self._start_part(continuation)

    async def finish(self) -> None:
        """Drain rendered Markdown and end any tool-call group before a plugin rendering, a widget, or the prompt appears."""
        await self._drain()
        self._close_group()

    def _close_group(self) -> None:
        if self._group is not None:
            self._group.close()

    async def _drain(self) -> None:
        """Flush the open Markdown part. Tool calls keep counting into their group around it."""
        if self._buffer:
            self._line(self._buffer)
        if self._markdown is not None:
            self._markdown.finish()
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
        self._close_group()
        if self._block is not None:
            self._block.freeze()
        writer, self._writer = self._writer, None
        self._reset()
        if writer is not None:
            writer.abort()
        await anyio.sleep(0)

    def _reset(self) -> None:
        self._heading_printed = False
        self._buffer = ''
        self._markdown = None
        self._block = None
        self._index = None
