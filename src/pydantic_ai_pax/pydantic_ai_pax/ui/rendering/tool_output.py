"""Render typed capability events.

The one model-facing tool result read here is the `shell` output that Ctrl+O shows, split from its
handles by the harness, which writes them.
"""

import io
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from pydantic import BaseModel
from rich.ansi import AnsiDecoder
from rich.console import Console, Group
from rich.text import Text

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai_harness.file_system import FileChangeRequestedEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_ai_harness.shell._persistent import command_output
from pydantic_ai_pax.config import ToolCallDisplay
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import MarkdownBlock
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.command_syntax import highlight_command

ColorSystemName = Literal['standard', '256', 'truecolor', 'windows']


def color_system(console: Console) -> ColorSystemName | None:
    """The colour system `console` renders with; `None` when it renders no colour."""
    name = console.color_system
    return name if name in ('standard', '256', 'truecolor', 'windows') else None


def terminal_text(text: str, *, keep: str = '\n\t') -> str:
    """Make untrusted control characters inert before terminal rendering, except those in `keep`."""
    return ''.join(char if char.isprintable() or char in keep else f'\\x{ord(char):02x}' for char in text)


def tool_header(*, name: str, argument: str | Text = '') -> Group:
    """Highlight the tool name, leaving its marker and a plain-string argument muted, on one row.

    A `Text` argument is appended as-is, so it must already be styled and terminal-safe. Line
    breaks in a string are escaped, so model-supplied text cannot add a forged transcript row. The
    `Group` keeps the row unwrapped: `Console.print` copies a bare `Text` without its `no_wrap`.
    """
    text = Text('● ', style=theme.color(theme.MUTED), overflow='ellipsis', no_wrap=True)
    text.append(terminal_text(name, keep=''), style=theme.color(theme.ACCENT))
    if isinstance(argument, Text):
        if argument:
            text.append(' ')
            text.append_text(argument)
    elif argument:
        text.append(f' {terminal_text(argument, keep="")}', style=theme.color(theme.MUTED))
    return Group(text)


def print_tool_header(console: Console, *, name: str, argument: str | Text = '') -> None:
    """Print `tool_header` and the blank line that separates it from what follows."""
    console.print(tool_header(name=name, argument=argument))
    console.print()


SHELL_COMMAND_ROWS = 10
"""Most terminal rows a shell command header may take before the rest is summarized."""


def print_shell_header(console: Console, command: str) -> None:
    """Wrap the whole command under a hanging indent, so long one-liners and scripts stay readable.

    The command is highlighted as Bash, with embedded code in its own language. Rows beyond
    `SHELL_COMMAND_ROWS` are replaced by a count; the model-facing command is unchanged. No blank
    line follows, so the command's result row can sit under it. In the live prompt the header is its
    own transcript item, so a new width or theme renders it again.
    """
    if not command.strip():
        console.print(tool_header(name='shell'))
        return
    surface = console.file
    if console.is_terminal and isinstance(surface, PromptSurface):
        render = partial(_shell_header_ansi, colors=color_system(console))
        block = surface.markdown(render=render, width=console.width)
        block.extend(command)
        block.write(render(source=command, width=console.width))
        return
    _print_shell_header(console, command)


def _shell_header_ansi(*, source: str, width: int, colors: ColorSystemName | None) -> str:
    """Style on a console of our own, for a width or theme the header was not printed at.

    `colors` must be the stream console's colour system; see `render_markdown`.
    """
    output = io.StringIO()
    _print_shell_header(Console(file=output, force_terminal=True, color_system=colors, width=width), source)
    return output.getvalue()


def _print_shell_header(console: Console, command: str) -> None:
    muted = theme.color(theme.MUTED)
    # Spans, not a base style, so the command's unstyled names keep the terminal foreground.
    head = Text.assemble(('● ', muted), ('shell ', theme.color(theme.ACCENT)))
    indent = Text(' ' * head.cell_len)
    width = max(console.width - head.cell_len, 1)
    # Escape control characters before highlighting, so no style can carry them to the terminal.
    command = terminal_text(command.strip('\n'))
    # Each line takes a row or more, and each row `width` cells or fewer, so lex no further than the
    # shown rows reach. One row of slack keeps the last shown word whole.
    shown = len('\n'.join(command.split('\n')[:SHELL_COMMAND_ROWS]))
    end = min(shown, (SHELL_COMMAND_ROWS + 1) * width)
    rows = highlight_command(command, end=end).wrap(console, width, overflow='fold')
    total = len(Text(command).wrap(console, width, overflow='fold'))
    for index, row in enumerate(rows[:SHELL_COMMAND_ROWS]):
        row.rstrip()
        console.print((head if index == 0 else indent) + row, overflow='ellipsis', no_wrap=True)
    if total > SHELL_COMMAND_ROWS:
        console.print(indent + Text(f'… +{total - SHELL_COMMAND_ROWS} lines', style=muted), no_wrap=True)


def tool_arguments_text(arguments: Mapping[str, object], *, max_chars: int) -> Text:
    """Show arguments as one-line `name=value` pairs, clipping each value to `max_chars`.

    Values are compact JSON, so strings are quoted and newlines stay escaped. Zero hides arguments.
    """
    text = Text()
    if max_chars <= 0:
        return text
    for name, value in arguments.items():
        rendered = json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 1] + '…'
        if text:
            text.append(' ')
        text.append(terminal_text(name, keep=''), style=theme.color(theme.ACCENT))
        text.append(f'={terminal_text(rendered, keep="")}', style=theme.color(theme.MUTED))
    return text


_SGR = re.compile(r'(\x1b\[[0-9;]*m)')


def shell_text(text: str, decoder: AnsiDecoder) -> Text:
    """Decode SGR styles only; keep other terminal controls inert."""
    parts = _SGR.split(text)
    safe = ''.join(part if index % 2 else terminal_text(part) for index, part in enumerate(parts))
    return decoder.decode_line(safe)


EXPAND_HINT = 'ctrl+o to expand'
"""Ends a live result row whose output Ctrl+O can show."""


def _draw_row(console: Console, *, text: str, role: str, label: str, log: str | None, hint: bool) -> None:
    """Print a result row at `console`'s width, naming the call `label` last when it is not under its header.

    `role` is resolved when the row is drawn, so a row drawn again after a theme change takes its colours.
    """
    muted = theme.color(theme.MUTED)
    row = Text('  ⎿ ', style=muted)
    row.append(terminal_text(text, keep=''), style=theme.color(role))
    if label:
        row.append(f' · {terminal_text(label, keep="")}', style=muted)
    if hint and row.cell_len + len(EXPAND_HINT) + 3 <= console.width:
        row.append(f' · {EXPAND_HINT}', style=muted)
    if log is None:
        console.print(row, overflow='ellipsis', no_wrap=True)
    else:
        row.truncate(console.width, overflow='ellipsis')
        # The path wraps rather than clips, so it stays usable.
        console.print(row + Text(f' · log {terminal_text(log, keep="")}', style=muted))


def _row_ansi(
    *, source: str, width: int, draw: Callable[..., None], colors: ColorSystemName | None, expanded: bool
) -> str:
    """A live result row at `width`, with a hint when `source` has output, or `expanded` with that output under it.

    `colors` must be the stream console's colour system; see `render_markdown`.
    """
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system=colors, width=width)
    draw(console, hint=not expanded and bool(source))
    if expanded and source:
        decoder = AnsiDecoder()
        for line in source.removesuffix('\n').split('\n'):
            # Printed whole: the transcript wraps it, so a wrapped URL still opens in full.
            console.print(
                Text('    ') + shell_text(line.removesuffix('\r'), decoder),
                style=theme.color(theme.MUTED),
                no_wrap=True,
                overflow='ignore',
                crop=False,
            )
    return output.getvalue()


class DisplayArgs(BaseModel):
    """Optional display fields for file and shell tools."""

    path: str | None = None
    command: str | None = None
    offset: int = 1
    limit: int | None = None
    glob: str | None = None
    recursive: bool = True


@dataclass
class ShellPreview:
    """Count displayed logical lines across output chunks."""

    completed_lines: int = 0
    cut: bool = False
    """Whether a preview line was wider than the terminal and lost its end."""
    pending: str = ''
    carriage_return: bool = False
    decoder: AnsiDecoder = field(default_factory=AnsiDecoder)

    @property
    def shown(self) -> int:
        """Include a displayed unterminated line."""
        return self.completed_lines


class ToolOutput:
    """Present bounded shell chunks, Termflow-highlighted file diffs, and one result row per command or failure.

    A header's separating blank line is owed rather than printed, so a result row can sit right under
    it; `separate` pays it before anything else prints. A row that lands under another call's header
    names its own call.
    """

    def __init__(
        self,
        console: Console,
        *,
        shell_lines: int = 20,
        show_output: bool = False,
        tool_calls: ToolCallDisplay = 'detailed',
    ) -> None:
        """Use the conversation's output stream, not global stdout.

        A `grouped` tool-call style counts shell calls, so it shows no shell header, output, or result row.
        """
        self.console = console
        self.shell_lines = shell_lines
        self.show_output = show_output and tool_calls == 'detailed'
        self.tool_calls = tool_calls
        self._shells: dict[str | None, ShellPreview] = {}
        self._headers: set[tuple[str | None, str]] = set()
        self._writes: dict[tuple[str | None, str, str], FileChangeRequestedEvent] = {}
        self._changes: dict[str | None, FileChangeRequestedEvent] = {}
        """Each call's latest file change request, which a listener may cancel before the result."""
        self._labels: dict[str | None, str] = {}
        """What each shown call's header said, for a row that does not land under it."""
        self._owed: str | None = None
        """The call whose header still owes its blank line."""
        self._blank_owed = False
        self._outputs: dict[str | None, tuple[MarkdownBlock, int]] = {}
        """Live rows of finished commands, and their PIDs, waiting for the output their tool result brings."""

    def separate(self) -> None:
        """Print the blank line a header still owes, before other output follows it."""
        if self._blank_owed:
            self.console.print()
        self._blank_owed = False
        self._owed = None

    def header(self, tool_call_id: str | None, *, name: str, argument: str | Text = '') -> None:
        """Print a tool's header, owing the blank line that separates it from what follows."""
        self.separate()
        if name == 'shell' and isinstance(argument, str):
            print_shell_header(self.console, argument)
            label = argument.strip().split('\n', 1)[0]
        else:
            self.console.print(tool_header(name=name, argument=argument))
            label = f'{name} {argument.plain if isinstance(argument, Text) else argument}'.rstrip()
        self._labels[tool_call_id] = label
        self._owed = tool_call_id
        self._blank_owed = True

    def _row(
        self,
        tool_call_id: str | None,
        text: str,
        *,
        role: str,
        log: str | None = None,
        name: str = '',
        expandable: bool = False,
    ) -> MarkdownBlock | None:
        """Print one result row under the call's header, naming the call last when another header is in between.

        `name` stands in for a call whose header this renderer did not print. An `expandable` row in the live
        prompt is returned as a transcript block, for the output Ctrl+O shows under it.
        """
        under_header = self._blank_owed and tool_call_id is not None and self._owed == tool_call_id
        if not under_header:
            self.separate()
        label = '' if under_header else self._labels.get(tool_call_id, name)
        draw = partial(_draw_row, text=text, role=role, label=label, log=log)
        self._blank_owed = False
        self._owed = None
        surface = self.console.file
        if not (expandable and self.console.is_terminal and isinstance(surface, PromptSurface)):
            draw(self.console, hint=False)
            return None
        colors = color_system(self.console)
        collapsed = partial(_row_ansi, draw=draw, colors=colors, expanded=False)
        expanded = partial(_row_ansi, draw=draw, colors=colors, expanded=True)
        block = surface.markdown(render=collapsed, width=self.console.width, expanded=expanded)
        # Written without its source, so a row whose output is dropped over the transcript budget shows no hint.
        block.write(collapsed(source='', width=self.console.width))
        return block

    def prints_diff(self, event: AgentStreamEvent) -> bool:
        """Whether `event` is a file's diff, or a call to a tool that prints one, in every tool-call style."""
        if isinstance(event, FunctionToolCallEvent):
            return event.part.tool_name in ('write_file', 'edit_file')
        return isinstance(event, FileWrittenEvent)

    def render_call(self, event: FunctionToolCallEvent) -> bool:
        """Show arguments once, before execution, including for failed calls."""
        name = event.part.tool_name
        if name not in ('shell', 'write_file', 'edit_file', 'read_file', 'list_files'):
            return False
        try:
            args = DisplayArgs.model_validate_json(event.part.args_as_json_str())
        except ValueError:
            return False
        if name in ('read_file', 'list_files'):
            return self._inspection_header(event.part.tool_call_id, name, args)
        argument = args.command if name == 'shell' else args.path
        if argument is None:
            return False
        self.header(event.part.tool_call_id, name=name, argument=argument)
        self._headers.add((event.part.tool_call_id, name))
        return True

    def _inspection_header(self, tool_call_id: str, name: str, args: DisplayArgs) -> bool:
        """Name what is read or listed, with only the arguments the model set."""
        if name == 'read_file':
            if args.path is None:
                return False
            details = ''
            if args.limit is not None:
                # The default read limit is 2000 lines, whatever the model asks for.
                details = f' lines {args.offset}-{args.offset + min(args.limit, 2000) - 1}'
            elif args.offset > 1:
                details = f' from line {args.offset}'
            argument = args.path + details
        else:
            first = None if args.limit is None else f'first {args.limit}'
            parts = (args.path or '.', args.glob, None if args.recursive else 'not recursive', first)
            argument = ' '.join(part for part in parts if part)
        self.header(tool_call_id, name=name, argument=argument)
        return True

    def render_result(self, event: FunctionToolResultEvent) -> None:
        """Mark a call that failed, was refused, was interrupted, or has to be retried; a success shows nothing more."""
        if event.part.outcome != 'success':
            lines = event.part.model_response_str(wrap_if_error=False).strip().splitlines()
            reason = lines[0] if lines else event.part.outcome
            if event.part.retry is not None and event.part.retry.cause == 'validation_error':
                # Its reason is a dump of validation errors, not a sentence.
                reason = 'invalid arguments'
        else:
            # A refused file change still succeeds, telling the model why nothing changed.
            request = self._changes.get(event.part.tool_call_id)
            refusal = (request.cancel_reason or '').strip().split('\n', 1)[0] if request else ''
            reason = f'refused: {refusal or "not approved"}' if request and request.cancelled else ''
        pending = self._outputs.pop(event.part.tool_call_id, None)
        if pending is not None and isinstance(event.part.content, str):
            block, pid = pending
            # The model's copy has secrets masked; its handles are left out, since the row already gives them.
            block.extend(command_output(event.part.content, pid=pid))
        if reason:
            self._row(event.part.tool_call_id, reason, role=theme.ERROR, name=event.part.tool_name)
            self.console.print()

    def _shell_chunk(self, event: CommandOutputEvent) -> None:
        preview = self._shells.setdefault(event.tool_call_id, ShellPreview())
        for char in event.text:
            if preview.completed_lines >= self.shell_lines:
                break
            if char == '\n':
                self._shell_line(event.tool_call_id, preview)
                preview.carriage_return = False
            elif char == '\r':
                preview.carriage_return = True
            else:
                if preview.carriage_return:
                    shell_text(preview.pending, preview.decoder)
                    preview.pending = ''
                    preview.carriage_return = False
                preview.pending += char

    def _shell_line(self, tool_call_id: str | None, preview: ShellPreview) -> None:
        if not (self._blank_owed and self._owed == tool_call_id):
            # Output under another call's header starts its own block.
            self.separate()
        line = shell_text(preview.pending, preview.decoder)
        preview.cut = preview.cut or line.cell_len > self.console.width
        self.console.print(
            line,
            style=theme.color(theme.MUTED),
            markup=False,
            highlight=False,
            overflow='ellipsis',
            no_wrap=True,
        )
        preview.pending = ''
        preview.completed_lines += 1

    def _shell_finished(self, event: CommandFinishedEvent) -> None:
        """One row: the exit status and line count, and the log when output was cut or keeps coming.

        Without `show_output` no output is shown: the events carry the raw output, before the harness masks secrets.
        """
        preview = self._shells.pop(event.tool_call_id, ShellPreview())
        if preview.pending and preview.completed_lines < self.shell_lines:
            self._shell_line(event.tool_call_id, preview)
        failed = event.exit_code not in (None, 0)
        parts = [f'exit {event.exit_code}' if event.exit_code is not None else 'running in background']
        if event.line_count:
            lines = f'{event.line_count} line{"s" if event.line_count != 1 else ""}'
            if self.show_output and preview.shown < event.line_count:
                lines += f', {preview.shown} shown'
            parts.append(lines)
        clipped = self.show_output and (preview.cut or preview.shown < (event.line_count or 0))
        log = event.output_path if event.truncated or event.exit_code is None or clipped else None
        # A background command has no output yet, and a full preview needs no second copy.
        hidden = event.exit_code is not None and (not self.show_output or clipped or event.truncated)
        role = theme.ERROR if failed else theme.MUTED
        block = self._row(event.tool_call_id, ' · '.join(parts), role=role, log=log, expandable=hidden)
        if block is not None:
            self._outputs[event.tool_call_id] = (block, event.pid)
        self.console.print()

    def _diff(self, diff: str, *, truncated: bool) -> None:
        self.separate()
        # Neither form ends in a newline, so the blank line printed below is a row of its own.
        safe_diff = terminal_text(diff).rstrip('\n')
        if safe_diff:
            if self.console.is_terminal:
                # Termflow drops file and hunk headers, so a diff of headers only renders no rows.
                if rendered := theme.diff_renderer().render(safe_diff):
                    self.console.file.write(rendered + '\n')
                    self.console.file.flush()
            else:
                self.console.print(safe_diff, markup=False, highlight=False)
        if truncated:
            self.console.print('Diff truncated.', style=theme.color(theme.MUTED))
        self.console.print()

    def abort(self) -> None:
        """Release pending events when a run ends without tool results."""
        self._writes.clear()
        self._changes.clear()
        self._headers.clear()
        self._shells.clear()
        self._labels.clear()
        self._outputs.clear()

    def discard_call(self, tool_call_id: str) -> None:
        """Release proposed diffs and the header label after a tool result, including refusals and retries."""
        self._writes = {key: value for key, value in self._writes.items() if key[0] != tool_call_id}
        self._changes.pop(tool_call_id, None)
        self._labels.pop(tool_call_id, None)

    def render(self, event: AgentStreamEvent) -> bool:
        """Return whether this event belongs to the specialized tool display."""
        match event:
            case CommandStartedEvent():
                self._shell_started(event)
            case CommandOutputEvent():
                if self.show_output:
                    self._shell_chunk(event)
            case CommandFinishedEvent():
                if self.tool_calls == 'detailed':
                    self._shell_finished(event)
                else:
                    self._shells.pop(event.tool_call_id, None)
            case FileChangeRequestedEvent():
                self._changes[event.tool_call_id] = event
                if event.operation == 'write':
                    self._writes[event.tool_call_id, event.root_dir, event.path] = event
            case FileEditedEvent():
                self._file_edited(event)
            case FileWrittenEvent():
                self._file_written(event)
            case _:
                return False
        return True

    def _shell_started(self, event: CommandStartedEvent) -> None:
        if self.show_output:
            self._shells[event.tool_call_id] = ShellPreview()
        key = (event.tool_call_id, 'shell')
        if key not in self._headers and self.tool_calls == 'detailed':
            self.header(event.tool_call_id, name='shell', argument=event.command)
        self._headers.discard(key)

    def _file_edited(self, event: FileEditedEvent) -> None:
        key = (event.tool_call_id, 'edit_file')
        if key not in self._headers:
            self.header(event.tool_call_id, name='edit_file', argument=event.path)
        self._headers.discard(key)
        self._diff(event.diff, truncated=event.truncated)

    def _file_written(self, event: FileWrittenEvent) -> None:
        key = (event.tool_call_id, 'write_file')
        if key not in self._headers:
            self.header(event.tool_call_id, name='write_file', argument=event.path)
        self._headers.discard(key)
        request = self._writes.pop((event.tool_call_id, event.root_dir, event.path), None)
        if request is not None and not request.cancelled:
            self._diff(request.diff, truncated=request.truncated)
        else:
            self.separate()
