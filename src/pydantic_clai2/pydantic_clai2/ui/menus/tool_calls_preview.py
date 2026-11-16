"""A sample of each `display.tool_calls` style for the `/set` choice picker."""

from io import StringIO

from rich.console import Console

from pydantic_clai2.ui.rendering.tool_group import ToolCallGroup
from pydantic_clai2.ui.rendering.tool_output import print_tool_header

_CALLS = (
    ('shell', 'git status'),
    ('shell', 'git diff'),
    ('shell', 'pytest -q'),
    ('read_file', "'src/app.py' offset=0 limit=2000 lines"),
    ('read_file', "'src/util.py' offset=0 limit=2000 lines"),
    ('shell', 'ruff check'),
)


def tool_calls_preview(style: str, *, width: int) -> str:
    """Print the same calls the way `style` would, using the shell's own tool-line renderers.

    The console is not a terminal, so a group shows only its final line, not each redraw.
    """
    output = StringIO()
    console = Console(file=output, width=width, color_system='truecolor', highlight=False)
    if style == 'grouped':
        group = ToolCallGroup(console, colors='truecolor')
        for name, _ in _CALLS:
            group.add(name)
        group.close()
    else:
        for name, argument in _CALLS:
            print_tool_header(console, name=name, argument=argument)
    return output.getvalue().rstrip('\n')
