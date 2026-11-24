"""Grep invocation and bounded result previews from native tool events."""

from pydantic import BaseModel, ValidationError

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.tool_output import ToolOutput, terminal_text


class GrepArgs(BaseModel):
    """Only the display fields; other search options belong to the tool."""

    pattern: str
    path: str = '.'


class GrepOutput:
    """Keep concurrent grep results associated with their invocation."""

    def __init__(self, tool_output: ToolOutput, *, lines: int = 20, show_output: bool = False) -> None:
        """Limit display only, without changing the result sent to the model.

        Headers go through `tool_output`, so a failed search gets its result row under its own header.
        """
        self.tool_output = tool_output
        self.console = tool_output.console
        self.lines = lines
        self.show_output = show_output
        self._calls: dict[str, str] = {}

    def render(self, event: FunctionToolCallEvent | FunctionToolResultEvent) -> bool:
        """Return false for unrelated or unsupported tool calls."""
        if isinstance(event, FunctionToolCallEvent):
            if event.part.tool_name != 'grep':
                return False
            try:
                args = GrepArgs.model_validate_json(event.part.args_as_json_str())
            except (ValidationError, ValueError):
                return False
            label = f'grep {args.pattern!r} in {args.path!r}'
            self._calls[event.part.tool_call_id] = label
            self.tool_output.header(event.part.tool_call_id, name='grep', argument=f'{args.pattern!r} in {args.path!r}')
            return True
        label = self._calls.pop(event.part.tool_call_id, None)
        if label is None or event.part.outcome != 'success':
            # A failure gets the result row every tool does.
            return False
        if not self.show_output:
            return True
        self.tool_output.separate()
        if not isinstance(event.part.content, str):
            self.console.print(
                f'{terminal_text(label)}: tool did not return text results.',
                style=theme.color(theme.MUTED),
                markup=False,
            )
            self.console.print()
            return True
        rows = event.part.content.splitlines()
        tool_truncated = bool(rows and rows[-1] == '[truncated; narrow the search]')
        if tool_truncated:
            rows.pop()
        self.console.print(
            f'Results: {terminal_text(label)}', style=theme.color(theme.MUTED), markup=False, highlight=False
        )
        for row in rows[: self.lines]:
            self.console.print(terminal_text(row), style=theme.color(theme.MUTED), markup=False, highlight=False)
        hidden = max(0, len(rows) - self.lines)
        if hidden:
            self.console.print(f'Truncated {hidden} result lines', style=theme.color(theme.MUTED))
        if tool_truncated:
            self.console.print(
                'Tool also truncated the search; additional result count unknown.', style=theme.color(theme.MUTED)
            )
        elif not rows:
            self.console.print('No matches.', style=theme.color(theme.MUTED))
        self.console.print()
        return True
