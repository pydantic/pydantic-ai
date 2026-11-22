"""A sample of each `display.tool_calls` style for the `/set` choice picker."""

from io import StringIO

from rich.console import Console

from pydantic_ai_harness.shell import CommandFinishedEvent
from pydantic_ai_pax.ui.rendering.tool_group import ToolCallGroup
from pydantic_ai_pax.ui.rendering.tool_output import ToolOutput

_CALLS = (
    ('shell', 'git status', 2),
    ('shell', 'git diff', 14),
    ('shell', 'pytest -q', 3),
    ('read_file', "'src/app.py'", 0),
    ('read_file', "'src/util.py'", 0),
    ('shell', 'ruff check', 1),
)


def tool_calls_preview(style: str, *, width: int) -> str:
    """Print the same calls the way `style` would, using the shell's own tool-line renderers.

    The console is not a terminal, so a group shows only its final line, not each redraw.
    """
    output = StringIO()
    console = Console(file=output, width=width, color_system='truecolor', highlight=False)
    if style == 'grouped':
        group = ToolCallGroup(console, colors='truecolor')
        for name, _, _ in _CALLS:
            group.add(name)
        group.close()
    else:
        tool_output = ToolOutput(console)
        for index, (name, argument, lines) in enumerate(_CALLS):
            tool_output.header(str(index), name=name, argument=argument)
            if name == 'shell':
                tool_output.render(
                    CommandFinishedEvent(
                        tool_call_id=str(index),
                        pid=0,
                        output_path='',
                        status_path='',
                        exit_code=0,
                        truncated=False,
                        line_count=lines,
                    )
                )
    return output.getvalue().rstrip('\n')
