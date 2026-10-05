"""The grouped tool-call display: consecutive calls counted by tool name on one live line."""

from rich.console import Console
from rich.text import Text

from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text


class ToolCallGroup:
    """Print `● shell 4, grep 2, shell 3`, growing the last count in place as calls arrive.

    A count is only final once a different tool, or other output, follows it. On a terminal the
    line is redrawn from column zero with each call; elsewhere it prints once, when it ends.
    A tool that no longer fits the row starts the next line, so a redraw never wraps.
    """

    def __init__(self, console: Console) -> None:
        """Count into `console`; nothing prints until the first call."""
        self.console = console
        self._runs: list[tuple[str, int]] = []

    def add(self, name: str) -> None:
        """Count one call, extending the last run when it is the same tool."""
        name = terminal_text(name, keep='')
        if self._runs and self._runs[-1][0] == name:
            self._runs[-1] = (name, self._runs[-1][1] + 1)
        else:
            if self._runs and self._line([*self._runs, (name, 1)]).cell_len >= self.console.width:
                self._end_line()
                self._runs = []
            self._runs.append((name, 1))
        if self.console.is_terminal:
            self.console.file.write('\r')
            self._draw(end='')

    def close(self) -> None:
        """End the line and leave a blank one, like any other tool output. Does nothing when empty."""
        if self._runs:
            self._end_line()
            self.console.print()
            self._runs = []

    def _line(self, runs: list[tuple[str, int]]) -> Text:
        text = Text('● ', style=theme.color(theme.MUTED))
        for index, (name, count) in enumerate(runs):
            if index:
                text.append(', ', style=theme.color(theme.MUTED))
            text.append(name, style=theme.color(theme.ACCENT))
            text.append(f' {count}', style=theme.color(theme.MUTED))
        return text

    def _draw(self, *, end: str) -> None:
        self.console.print(self._line(self._runs), end=end, overflow='ellipsis', no_wrap=True)

    def _end_line(self) -> None:
        """A terminal already shows the line, so only move past it."""
        if self.console.is_terminal:
            self.console.print()
        else:
            self._draw(end='\n')
