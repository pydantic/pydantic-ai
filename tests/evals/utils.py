from io import StringIO

from rich.console import Console, RenderableType


def trim_trailing_whitespace(rendered: str) -> str:
    """Trim end-of-line whitespace to prevent snapshot diffs after pre-commit removes the whitespace."""
    return '\n'.join([line.rstrip() for line in rendered.split('\n')])


def render_table(table: RenderableType) -> str:
    """Render a rich renderable as a string."""
    string_io = StringIO()
    Console(width=300, file=string_io).print(table)
    return trim_trailing_whitespace(string_io.getvalue())
