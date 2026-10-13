"""The grouped tool-call display: consecutive calls counted by tool name on one live line."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import MarkdownBlock
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text

if TYPE_CHECKING:
    from pydantic_clai2.ui.rendering._rendering import ColorSystemName

_Streak = tuple[str, int]
"""A tool name and how many consecutive calls it received."""


def _line(streaks: Sequence[_Streak]) -> Text:
    text = Text('● ', style=theme.color(theme.MUTED))
    for index, (name, count) in enumerate(streaks):
        if index:
            text.append(', ', style=theme.color(theme.MUTED))
        text.append(name, style=theme.color(theme.ACCENT))
        text.append(f' {count}', style=theme.color(theme.MUTED))
    return text


def _count(rows: list[list[_Streak]], name: str, *, width: int) -> None:
    """Count one call into the last row, or start a row when a new tool would not fit, so a redraw never wraps.

    A new tool is measured with a three-digit count, leaving its count room to grow in place.
    """
    row = rows[-1]
    if row and row[-1][0] == name:
        row[-1] = (name, row[-1][1] + 1)
    elif row and _line([*row, (name, 999)]).cell_len >= width:
        rows.append([(name, 1)])
    else:
        row.append((name, 1))


class ToolCallGroup:
    """Print `● shell 4, grep 2, shell 3`, growing the last count in place as calls arrive.

    A count is only final once a different tool, or other output, follows it. On a terminal the
    line is redrawn from column zero with each call; elsewhere it prints once, when it ends.
    In the live prompt the group is its own transcript item, like a streamed Markdown part, so
    output printed while it is open lands after the line instead of on it.
    """

    def __init__(self, console: Console, *, colors: ColorSystemName | None) -> None:
        """Count into `console`, styled in `colors`, which must be `console`'s colour system."""
        self.console = console
        self.colors: ColorSystemName | None = colors
        self._rows: list[list[_Streak]] = [[]]
        self._block: MarkdownBlock | None = None

    def add(self, name: str) -> None:
        """Count one call, extending the last streak when it is the same tool."""
        name = terminal_text(name, keep='')
        if not self._rows[-1] and self.console.is_terminal and isinstance(self.console.file, PromptSurface):
            self._block = self.console.file.markdown(render=self._render, width=self.console.width)
        if self._block is not None:
            # One `+name` line per call, so an empty name stays distinct from the empty line `close` adds.
            self._block.extend(f'+{name}\n')
        rows = len(self._rows)
        _count(self._rows, name, width=self.console.width)
        if len(self._rows) > rows:
            self._end_line(self._rows[-2])
        if self.console.is_terminal:
            self._write('\r' + self._ansi(self._rows[-1], width=self.console.width, end=''))

    def close(self) -> None:
        """End the line and leave a blank one, like any other tool output. Does nothing when empty."""
        if self._rows[-1]:
            self._end_line(self._rows[-1])
            if self._block is None:
                self.console.print()
            else:
                # The blank line stays part of the group, above anything printed while it was open.
                self._block.extend('\n')
                self._write('\n')
            self._rows = [[]]
            self._block = None

    def _end_line(self, row: list[_Streak]) -> None:
        """A terminal already shows the line, so only move past it."""
        self._write('\n' if self.console.is_terminal else self._ansi(row, width=self.console.width))

    def _write(self, text: str) -> None:
        output = self._block or self.console.file
        output.write(text)
        output.flush()

    def _ansi(self, row: list[_Streak], *, width: int, end: str = '\n') -> str:
        """Style on a console of our own: a replay can run while the stream's console is mid-print."""
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, color_system=self.colors, width=width)
        line = _line(row)
        if line.cell_len > width:
            # Only the last streak can outgrow the row; cut its name, never its count.
            *rest, (name, count) = row
            short = Text(name)
            short.truncate(width - _line([*rest, ('', count)]).cell_len, overflow='ellipsis')
            line = _line([*rest, (short.plain, count)])
        console.print(line, end=end, overflow='ellipsis', no_wrap=True)
        return output.getvalue()

    def _render(self, *, source: str, width: int) -> str:
        """The whole group again, for a width or theme it was not drawn at; `source` holds one `+name` line per call."""
        rows: list[list[_Streak]] = [[]]
        for line in source.splitlines():
            if line:
                _count(rows, line[1:], width=width)
        # `close` ends the source with an empty line for the group's blank one.
        blank = '\n' if source.endswith('\n\n') else ''
        return ''.join(self._ansi(row, width=width) for row in rows) + blank
