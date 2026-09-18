"""Render typed capability events without parsing model-facing tool results."""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.text import Text
from termflow.diff import DiffRenderer, DiffTheme  # pyright: ignore[reportMissingTypeStubs]

from . import theme


def terminal_text(text: str) -> str:
    """Make untrusted control characters inert before terminal rendering."""
    return ''.join(char if char.isprintable() or char in '\n\t' else f'\\x{ord(char):02x}' for char in text)


_SGR = re.compile(r'(\x1b\[[0-9;]*m)')


def shell_text(text: str, decoder: AnsiDecoder) -> Text:
    """Decode SGR styles only; keep other terminal controls inert."""
    parts = _SGR.split(text)
    safe = ''.join(part if index % 2 else terminal_text(part) for index, part in enumerate(parts))
    return decoder.decode_line(safe)


class DisplayArguments(BaseModel):
    """Optional display fields for file and shell tools."""

    path: str | None = None
    command: str | None = None
    offset: int = 0
    limit: int | None = None
    glob: str | None = None
    recursive: bool = True


@dataclass
class ShellPreview:
    """Count displayed logical lines across output chunks."""

    completed_lines: int = 0
    pending: str = ''
    carriage_return: bool = False
    decoder: AnsiDecoder = field(default_factory=AnsiDecoder)

    @property
    def shown(self) -> int:
        """Include a displayed unterminated line."""
        return self.completed_lines


class ToolOutput:
    """Present bounded shell chunks and Termflow-highlighted file diffs."""

    def __init__(self, console: Console, *, shell_lines: int = 20, show_output: bool = False) -> None:
        """Use the conversation's output stream, not global stdout."""
        self.console = console
        self.shell_lines = shell_lines
        self.show_output = show_output
        self._shells: dict[str | None, ShellPreview] = {}
        self._headers: set[tuple[str | None, str]] = set()
        self._writes: dict[tuple[str | None, str, str], FileChangeRequestEvent] = {}

    def _header(self, name: str, argument: str) -> None:
        lines = argument.splitlines()
        summary = lines[0] if lines else ''
        if len(lines) > 1:
            summary += f' (+{len(lines) - 1} command lines)'
        text = Text(f'● {name} ', style=theme.MUTED)
        text.append(terminal_text(summary), style=theme.ACCENT)
        self.console.print(text, overflow='ellipsis', no_wrap=True)
        if self.show_output:
            self.console.print()

    def render_call(self, event: FunctionToolCallEvent) -> bool:
        """Show arguments once, before execution, including for failed calls."""
        name = event.part.tool_name
        if name not in ('shell', 'write_file', 'edit_file', 'read_file', 'list_files'):
            return False
        try:
            args = DisplayArguments.model_validate_json(event.part.args_as_json_str())
        except ValueError:
            return False
        if name in ('read_file', 'list_files'):
            return self._inspection_header(name, args)
        argument = args.command if name == 'shell' else args.path
        if argument is None:
            return False
        self._header(name, argument)
        self._headers.add((event.part.tool_call_id, name))
        return True

    def _inspection_header(self, name: str, args: DisplayArguments) -> bool:
        if name == 'read_file':
            if args.path is None:
                return False
            limit = min(args.limit, 2000) if args.limit is not None else 2000
            details = f'offset={args.offset} limit={limit} lines'
            if args.limit is not None and args.limit > 2000:
                details += f' (requested {args.limit})'
            path = args.path
        else:
            path = args.path or '.'
            limit = args.limit if args.limit is not None else 200
            details = f'recursive={str(args.recursive).lower()} limit={limit}'
            if args.glob is not None:
                details += f' glob={args.glob!r}'
        self._header(name, f'{path!r} {details}')
        return True

    def _shell_chunk(self, event: CommandOutputEvent) -> None:
        preview = self._shells.setdefault(event.tool_call_id, ShellPreview())
        for char in event.text:
            if preview.completed_lines >= self.shell_lines:
                break
            if char == '\n':
                self._shell_line(preview)
                preview.carriage_return = False
            elif char == '\r':
                preview.carriage_return = True
            else:
                if preview.carriage_return:
                    shell_text(preview.pending, preview.decoder)
                    preview.pending = ''
                    preview.carriage_return = False
                preview.pending += char

    def _shell_line(self, preview: ShellPreview) -> None:
        self.console.print(
            shell_text(preview.pending, preview.decoder),
            style=theme.MUTED,
            markup=False,
            highlight=False,
            overflow='ellipsis',
            no_wrap=True,
        )
        preview.pending = ''
        preview.completed_lines += 1

    def _shell_finished(self, event: CommandFinishedEvent) -> None:
        preview = self._shells.pop(event.tool_call_id, ShellPreview())
        if preview.pending and preview.completed_lines < self.shell_lines:
            self._shell_line(preview)
        omitted = max(0, event.total_lines - preview.shown) if event.total_lines is not None else 0
        if omitted:
            self.console.print(f'Truncated {omitted} lines', style=theme.MUTED)
        state = f'exit {event.exit_code}' if event.exit_code is not None else 'running in background'
        self.console.print(f'{state} | PID {event.pid}', style=theme.MUTED, markup=False, highlight=False)
        self.console.print(
            f'Output: {terminal_text(event.output_path)}', style=theme.MUTED, markup=False, highlight=False
        )
        self.console.print(
            f'Status: {terminal_text(event.status_path)}', style=theme.MUTED, markup=False, highlight=False
        )
        if event.truncated and not omitted:
            self.console.print('Output preview truncated; full output is in the command log.', style=theme.MUTED)
        self.console.print()

    def _diff(self, diff: str, *, truncated: bool) -> None:
        if not self.show_output:
            return
        safe_diff = terminal_text(diff)
        if safe_diff:
            if self.console.is_terminal:
                self.console.file.write(
                    DiffRenderer(
                        theme=DiffTheme(
                            addition=theme.DIFF_ADDITION,
                            deletion=theme.DIFF_DELETION,
                            marker_brighten=2.0,
                        )
                    ).render(safe_diff)
                )
                self.console.file.flush()
            else:
                self.console.print(safe_diff, markup=False, highlight=False)
        if truncated:
            self.console.print('Diff truncated.', style=theme.MUTED)
        self.console.print()

    def abort(self) -> None:
        """Release pending events when a run ends without tool results."""
        self._writes.clear()
        self._headers.clear()
        self._shells.clear()

    def discard_call(self, tool_call_id: str) -> None:
        """Release proposed diffs after a tool result, including refusals and retries."""
        self._writes = {key: value for key, value in self._writes.items() if key[0] != tool_call_id}

    def render(self, event: AgentStreamEvent) -> bool:
        """Return whether this event belongs to the specialized tool display."""
        if isinstance(event, CommandStartedEvent):
            self._shells[event.tool_call_id] = ShellPreview()
            key = (event.tool_call_id, 'shell')
            if key not in self._headers:
                self._header('shell', event.command)
            self._headers.discard(key)
        elif isinstance(event, CommandOutputEvent):
            if self.show_output:
                self._shell_chunk(event)
        elif isinstance(event, CommandFinishedEvent):
            if not self.show_output:
                self._shells.pop(event.tool_call_id, None)
                return True
            self._shell_finished(event)
        elif isinstance(event, FileChangeRequestEvent):
            if self.show_output and event.operation == 'write':
                self._writes[event.tool_call_id, event.root_dir, event.path] = event
        elif isinstance(event, FileEditedEvent):
            key = (event.tool_call_id, 'edit_file')
            if key not in self._headers:
                self._header('edit_file', event.path)
            self._headers.discard(key)
            self._diff(event.diff, truncated=event.truncated)
        elif isinstance(event, FileWrittenEvent):
            key = (event.tool_call_id, 'write_file')
            if key not in self._headers:
                self._header('write_file', event.path)
            self._headers.discard(key)
            if not self.show_output:
                return True
            request = self._writes.pop((event.tool_call_id, event.root_dir, event.path), None)
            if request is not None and not request.cancelled:
                self._diff(request.diff, truncated=request.truncated)
            else:
                self.console.print()
        else:
            return False
        return True
