from io import StringIO

from rich.console import Console, RenderableType


def render_table(table: RenderableType) -> str:
    """Render a rich renderable as a string."""
    string_io = StringIO()
    Console(width=300, file=string_io).print(table)
    rendered = string_io.getvalue()
    # Need to trim end-of-line whitespace to prevent snapshot diffs after pre-commit removes the whitespace
    trimmed = '\n'.join([line.rstrip() for line in rendered.split('\n')])
    return trimmed
