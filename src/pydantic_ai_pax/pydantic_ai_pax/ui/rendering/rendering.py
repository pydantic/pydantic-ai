"""Incremental Markdown rendering for native Pydantic AI events."""

import io
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from typing import IO

import anyio
from rich.console import Console, RenderableType
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from termflow import Parser, Renderer
from termflow.ansi import UNDERLINE_OFF, UNDERLINE_ON, fg_color, make_link
from termflow.parser.events import (
    BlockquoteEndEvent,
    CodeBlockEndEvent,
    CodeBlockLineEvent,
    CodeBlockStartEvent,
    EmptyLineEvent,
    HeadingEvent,
    ListEndEvent,
    ListItemEndEvent,
    ParseEvent,
    TableEndEvent,
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
from pydantic_ai_pax.config import ToolCallDisplay
from pydantic_ai_pax.runtime.sandbox_calls import DelegationToolCallEvent, SandboxCallOrder
from pydantic_ai_pax.ui.prompt.prompt_selection import trim_url
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import DROP, PREVIEW, MarkdownBlock
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.grep_output import GrepOutput
from pydantic_ai_pax.ui.rendering.markdown_table import table_lines
from pydantic_ai_pax.ui.rendering.tool_group import ToolCallGroup
from pydantic_ai_pax.ui.rendering.tool_output import (
    ColorSystemName,
    ToolOutput,
    color_system,
    terminal_text,
    tool_arguments_text,
)

PROMPT_MARKER = '> '
"""What starts an echoed prompt; its wrapped rows are indented by its width."""


def prompt_line(text: str) -> Text:
    """Text the user typed, as the transcript shows it: an accent marker and bold text set it apart from answers."""
    return Text.assemble((PROMPT_MARKER, theme.color(theme.ACCENT)), (terminal_text(text), 'bold'))


def print_prompt(console: Console, text: str) -> None:
    """Echo a submitted prompt, wrapped rows indented under its first, then a blank line.

    In the live prompt the prompt is its own transcript item, so a new width or theme wraps it again.
    """
    surface = console.file
    if console.is_terminal and isinstance(surface, PromptSurface):
        render = partial(_prompt_ansi, colors=color_system(console))
        block = surface.markdown(render=render, width=console.width)
        # Rows first: a repaint between the two calls would otherwise count an empty prompt as shown.
        block.write(render(source=text, width=console.width))
        block.extend(text)
    else:
        _print_prompt_rows(console, text)
    console.print()


def _prompt_ansi(*, source: str, width: int, colors: ColorSystemName | None) -> str:
    """The prompt's rows at `width`; `colors` must be the stream console's colour system, see `render_markdown`."""
    output = io.StringIO()
    _print_prompt_rows(Console(file=output, force_terminal=True, color_system=colors, width=width), source)
    return output.getvalue()


def _print_prompt_rows(console: Console, text: str) -> None:
    marker, body = prompt_line(text).divide([len(PROMPT_MARKER)])
    indent = Text(' ' * marker.cell_len)
    for index, row in enumerate(body.wrap(console, max(console.width - marker.cell_len, 1))):
        row.rstrip()
        console.print((marker if index == 0 else indent) + row, highlight=False, no_wrap=True)


def markdown_style() -> RenderStyle:
    """Keep the existing Markdown colours, readable on a light terminal, unless a Termflow palette is selected."""
    palette = theme.current()
    if palette is not None:
        return palette.to_render_style()
    return RenderStyle(
        bright=theme.color(theme.LITHIUM),
        head=theme.color(theme.PURPLE),
        symbol=theme.color(theme.CALCIUM),
        grey=theme.color(theme.GREY),
        dark=theme.DARK_PURPLE,
        mid=theme.ELEMENT_PURPLE,
        light=theme.color(theme.GREY),
        link=theme.color(theme.AQUA),
        error=theme.color(theme.CALCIUM),
    )


MAX_LINK = 2048
"""The longest URL Pax makes a terminal hyperlink, since the escape repeats it: per streamed chunk, or per folded row."""
_URL_RE = re.compile(r'<(https?://[^\s<>]+)>|(?<![\w/])https?://[^\s<>`]+')
_URL_START, _URL_END = '\ufdd0', '\ufdd1'
"""Marks around a URL while Termflow formats the line.

Model text cannot forge them: `terminal_text` escapes these noncharacters, and the
HTML entity decoding Termflow applies turns `&#xFDD0;` into nothing.
"""
_MARKED_URL_RE = re.compile(f'{_URL_START}([^{_URL_END}]*){_URL_END}')


_CODE_SPAN_SGR_RE = re.compile(r'\x1b\[2m([^\x1b]*)\x1b\[22m')
"""Termflow's inline code: dim on, the code, dim off (which is also bold off, so only this pair is matched)."""


class MarkdownRenderer(Renderer):
    """Termflow's renderer, also highlighting bare `https://` and `<https://...>` URLs as links."""

    _enclosing = ''
    """Codes that restore the style a link or code span interrupts, for headings Termflow styles around the text."""

    def render(self, event: ParseEvent) -> None:
        """Render one event, remembering a heading's style for links inside it."""
        self._enclosing = _heading_codes(event.level, self.style)[0] if isinstance(event, HeadingEvent) else ''
        # Termflow buffers a table's header, rows and alignments until its end event and draws it
        # there only once the table started and its header is set (`Renderer.render`); pax draws it
        # instead, then clears the header so the base class only resets its table state.
        if isinstance(event, TableEndEvent) and self._table_started and self._table_header is not None:
            for line in table_lines(
                header=[self._format_inline(cell) for cell in self._table_header],
                rows=[[self._format_inline(cell) for cell in row] for row in self._table_rows],
                alignments=self._table_alignments,
                width=self._current_width(),
                margin=self._margin(),
                style=self.style,
            ):
                self._writeln(line)
            self._table_header = None
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

        formatted = _MARKED_URL_RE.sub(self._link, super()._format_inline(_URL_RE.sub(mark, text)))
        if theme.current() is None and theme.light_background() is not None:
            # Termflow dims inline code, which leaves black text near 3:1 on white; paint it muted instead,
            # then restore a heading's colour, as after a link.
            muted = theme.sgr(theme.MUTED)
            formatted = _CODE_SPAN_SGR_RE.sub(lambda code: f'{muted}{code[1]}\x1b[39m{self._enclosing}', formatted)
        return formatted

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
        self._link = match[0] if 0 < len(match[1]) <= MAX_LINK else ''
        return self._link or '\x1b]8;;\x1b\\'

    def flush(self) -> None:
        """Forward flushes without taking ownership of the terminal."""
        self.output.flush()


class PartWriter(SmoothWriter):
    """Types a streaming part out smoothly; a finished code block replaces its preview in one write.

    When the part ends, whatever is still being typed is written at once.
    """

    def commit(self, text: str) -> None:
        """Write what is still being typed, then `text`, at once."""
        self._flush_all()
        self._target.write(text)
        self._target.flush()

    async def close(self) -> None:
        """Write what is still being typed at once, so the next tool line or the prompt does not wait for it."""
        self._flush_all()
        await super().close()


def thinking_heading(console: Console) -> str:
    """The label reasoning starts with, styled for `console`."""
    with console.capture() as capture:
        console.print('Thinking ', style=theme.color(theme.THINKING), end='')
    return capture.get()


class MarkdownPipeline:
    """Termflow's line parser and renderer, with code fences highlighted by Rich.

    With `preview`, for output a `MarkdownBlock` shows, a code block's lines appear as they arrive.
    The closing fence replaces them with the whole block lexed at once, so multiline strings and
    comments keep their state.
    """

    def __init__(
        self,
        *,
        output: IO[str],
        console: Console,
        thinking: bool,
        hyperlinks: bool,
        preview: bool = False,
        continuation: tuple[Parser, str] | None = None,
    ) -> None:
        """Render to `output` at `console`'s width."""
        self.output = output
        self.console = console
        self.thinking = thinking
        self.preview = preview
        self._parser = Parser()
        self._rendered = io.StringIO()
        """What the events of one line rendered, written to `output` at once."""
        self._renderer = MarkdownRenderer(
            output=self._rendered,
            width=console.width,
            style=markdown_style(),
            features=RenderFeatures(clipboard=False, hyperlinks=hyperlinks, images=False),
            dim=thinking,
        )
        self._code_lines: list[str] = []
        self._code_language = 'text'
        self._code_shown = False
        """Whether the open code block's lines are on screen as a preview."""
        if continuation is not None:
            self._parser, self._code_language = deepcopy(continuation)
        self._continued_fence = continuation is not None and self._parser.state.is_in_code()
        self._spaced = True
        """Whether a block starting now would follow a blank row, or nothing of this part."""
        self._fenced = False
        """Whether the last row painted closed a code block."""

    def continuation(self) -> tuple[Parser, str]:
        """Snapshot parsing context before finishing this display segment."""
        return deepcopy(self._parser), self._code_language

    def line(self, line: str) -> None:
        """Render one complete source line."""
        self._render_events(self._space_code(self._parser.parse_line(line)))

    def finish(self) -> None:
        """Close any open block, such as an unterminated fence."""
        self._render_events(self._space_code(self._parser.finalize()))

    def _space_code(self, events: list[ParseEvent]) -> list[ParseEvent]:
        """Add the blank row a code block needs before and after it when its source has none."""
        spaced: list[ParseEvent] = []
        for event in events:
            # Closing a list or quote paints no row, so a blank row before it still spaces what follows.
            paints = not isinstance(event, (ListItemEndEvent, ListEndEvent, BlockquoteEndEvent))
            opens = isinstance(event, CodeBlockStartEvent) and not self._spaced
            if paints and not isinstance(event, EmptyLineEvent) and (opens or self._fenced):
                spaced.append(EmptyLineEvent())
            spaced.append(event)
            if paints:
                self._spaced = isinstance(event, EmptyLineEvent)
                self._fenced = isinstance(event, CodeBlockEndEvent)
        return spaced

    def _render_events(self, events: list[ParseEvent]) -> None:
        for event in events:
            if isinstance(event, (CodeBlockStartEvent, CodeBlockLineEvent, CodeBlockEndEvent)):
                self._render_code(event)
            else:
                self._renderer.render(event)
        text = self._rendered.getvalue()
        self._rendered.seek(0)
        self._rendered.truncate()
        if DROP in text and isinstance(self.output, PartWriter):
            self.output.commit(text)
        elif text:
            self.output.write(text)
            self.output.flush()

    def _render_code(self, event: CodeBlockStartEvent | CodeBlockLineEvent | CodeBlockEndEvent) -> None:
        if isinstance(event, CodeBlockStartEvent):
            self._code_language = (event.language or 'text').split()[0]
            self._code_lines = []
            self._continued_fence = False
            if self.preview:
                self._show_code_rule()
        elif isinstance(event, CodeBlockLineEvent):
            self._code_lines.append(event.line)
            if self.preview:
                if not self._code_shown:
                    self._show_code_rule()
                # Lexed alone, a line inside a multiline string may be off until the fence lexes the block.
                self._rendered.write(self._code(event.line))
        elif self._continued_fence and not self._code_lines:
            # Steering immediately before the closing fence has no code left to display.
            self._continued_fence = False
        else:
            self._continued_fence = False
            if self._code_shown:
                self._rendered.write(DROP)
                self._code_shown = False
            self._rendered.write(
                self._code_rule() + self._code('\n'.join(self._code_lines)) + self._code_rule(end=True)
            )
            self._code_lines = []

    def _show_code_rule(self) -> None:
        self._rendered.write(PREVIEW + self._code_rule())
        self._code_shown = True

    def _code_rule(self, *, end: bool = False) -> str:
        with self.console.capture() as capture:
            title = '' if end else Text(self._code_language)
            self.console.rule(title, align='left', style=theme.color(theme.MUTED))
        return capture.get()

    def _code(self, code: str) -> str:
        with self.console.capture() as capture:
            self.console.print(
                Syntax(
                    code,
                    LANGUAGE_ALIASES.get(self._code_language.lower(), self._code_language.lower()),
                    theme=theme.syntax_theme(),
                    background_color='default',
                    word_wrap=True,
                ),
                style=Style(dim=self.thinking),
            )
        return capture.get()


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
        self._grep_output = GrepOutput(self._tool_output, lines=grep_lines, show_output=show_tool_output)
        self._group = ToolCallGroup(console, colors=color_system(console)) if tool_calls == 'grouped' else None
        self._delegations: set[str] = set()
        """Delegation calls in flight, whose outcome the task row already shows."""
        self.smooth_seconds = smooth_seconds
        self._thinking = False
        self._heading_printed = False
        self.show_thinking = show_thinking
        self.stop_loading = stop_loading
        self._writer: PartWriter | None = None
        self._markdown: MarkdownPipeline | None = None
        self._block: MarkdownBlock | None = None
        self._output: IO[str] | None = None
        """Where the console printed when the open part started."""
        self._buffer = ''
        self._index: int | None = None
        self.rendered_text = False

    async def on_stream_event(self, event: AgentStreamEvent) -> None:
        """Bind this callback to `Session.on_stream_event`."""
        if self._markdown is not None and self.console.file is not self._output:
            await self._follow_console(self._markdown)
        if await self._render_sandbox_call(event):
            return
        if await self._render_with_plugins(event):
            self._open_plugin_rendered_part(event)
            return
        if isinstance(event, CapabilityEvent) and await self._render_capability(event):
            return
        match event:
            case PartStartEvent() if isinstance(event.part, (TextPart, ThinkingPart)):
                await self._stream_part(event.index, event.part)
            case PartDeltaEvent() if event.index == self._index:
                self._feed_delta(event)
            case PartStartEvent():
                await self._drain()
            case PartEndEvent() if event.index == self._index:
                await self._drain()
            case FunctionToolCallEvent() | FunctionToolResultEvent():
                await self._render_tool(event)
            case _:
                pass

    def _open_plugin_rendered_part(self, event: AgentStreamEvent) -> None:
        """Open a text or thinking part a plugin rendered the start of, so the rest of it streams as usual."""
        if not (isinstance(event, PartStartEvent) and isinstance(event.part, (TextPart, ThinkingPart))):
            return
        self._thinking = isinstance(event.part, ThinkingPart)
        self._index = event.index
        self._start_part()
        if isinstance(event.part, TextPart):
            self.rendered_text = True

    async def _stream_part(self, index: int, part: TextPart | ThinkingPart) -> None:
        """Start streaming a text or thinking part, unless it is reasoning the user hid."""
        await self._drain()
        self.stop_loading()
        thinking = isinstance(part, ThinkingPart)
        if thinking and not self.show_thinking:
            return
        self._thinking = thinking
        self._index = index
        self._start_part()
        self._feed(part.content)
        if not thinking:
            self.rendered_text = True

    def _feed_delta(self, event: PartDeltaEvent) -> None:
        match event.delta:
            case TextPartDelta():
                self._feed(event.delta.content_delta)
            case ThinkingPartDelta():
                self._feed(event.delta.content_delta or '')
            case _:
                pass

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
        if isinstance(event, FunctionToolResultEvent):
            self._render_result(event)
        elif isinstance(event, DelegationToolCallEvent):
            self._delegations.add(event.part.tool_call_id)
        else:
            self._render_call(event)

    def _render_result(self, event: FunctionToolResultEvent) -> None:
        """A grouped style counts calls only; otherwise a failure gets its row."""
        delegated = event.part.tool_call_id in self._delegations
        self._delegations.discard(event.part.tool_call_id)
        if self._group is None and not delegated and not self._grep_output.render(event):
            self._tool_output.render_result(event)
        self._tool_output.discard_call(event.part.tool_call_id)

    def _render_call(self, event: FunctionToolCallEvent) -> None:
        if self._group is not None:
            if not self._tool_output.prints_diff(event):
                self._tool_output.separate()
                self._group.add(event.part.tool_name)
                return
            # A count would repeat the header its diff prints under.
            self._group.close()
        if self._grep_output.render(event):
            return
        if not self._tool_output.render_call(event):
            name = ''.join(char if char.isprintable() else ' ' for char in event.part.tool_name)
            arguments = tool_arguments_text(event.part.args_as_dict(), max_chars=self.tool_arg_chars)
            self._tool_output.header(event.part.tool_call_id, name=name, argument=arguments)

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

    async def _follow_console(self, markdown: MarkdownPipeline) -> None:
        """Go on with the open part where the console prints now, as when the terminal shows another session."""
        index, thinking, buffer = self._index, self._thinking, self._buffer
        continuation = markdown.continuation()
        # The line still arriving goes where the rest of the part does.
        self._buffer = ''
        await self.finish()
        self._index, self._thinking = index, thinking
        self._start_part(continuation)
        self._buffer = buffer
        if self._block is not None:
            self._block.extend(buffer)

    def _start_part(self, continuation: tuple[Parser, str] | None = None) -> None:
        # Before the part's transcript item opens, so the blank line lands above it.
        self._tool_output.separate()
        surface = self._output = self.console.file
        self._block = (
            surface.markdown(
                render=partial(
                    render_markdown,
                    thinking=self._thinking,
                    colors=color_system(self.console),
                    continuation=continuation,
                ),
                width=self.console.width,
                streaming=True,
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
            # History replays whole parts, so there is nothing to show early.
            preview=self._block is not None and self.smooth,
            continuation=continuation,
        )

    def _make_writer(self, output: IO[str]) -> PartWriter:
        """Reasoning keeps Code Puppy's slower thinking pace; responses use the configured catch-up."""
        output = LinkOutput(output=output)
        if self._thinking:
            return PartWriter(output, tick_interval=0.02, catch_up_seconds=0.4, min_chars_per_tick=2)
        return PartWriter(output, tick_interval=0.012, catch_up_seconds=self.smooth_seconds, min_chars_per_tick=1)

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
        print_prompt(self.console, text)
        if index is not None:
            self._index = index
            self._thinking = thinking
            self._start_part(continuation)

    async def finish(self) -> None:
        """Drain rendered Markdown and end any tool-call group before a plugin rendering, a widget, or the prompt appears."""
        await self._drain()
        self._close_group()

    def _close_group(self) -> None:
        """End what tool output left open: the group's line, or the blank line a header owes."""
        self._tool_output.separate()
        if self._group is not None:
            self._group.close()

    async def _drain(self) -> None:
        """Flush the open Markdown part. Tool calls keep counting into their group around it."""
        if self._buffer:
            self._line(self._buffer)
        if self._markdown is not None:
            self._markdown.finish()
        writer, self._writer = self._writer, None
        visible, block = self._heading_printed, self._block
        self._reset()
        try:
            if writer is not None:
                await writer.close()
        finally:
            # A cancelled close leaves the streamed rows; nothing else can reach the block to end it.
            if block is not None:
                block.end()
        if visible:
            self.console.print()
        self.console.file.flush()

    async def abort(self) -> None:
        """Discard pending output on cancellation and let the drainer terminate."""
        self._tool_output.abort()
        self._close_group()
        if self._block is not None:
            self._block.abort()
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
        self._output = None
        self._index = None
