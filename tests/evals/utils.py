from io import StringIO

from rich.console import Console
from rich.table import Table


def render_table(table: Table) -> str:
    """Render a rich Table as a string."""
    string_io = StringIO()
    Console(width=300, file=string_io).print(table)
    rendered = string_io.getvalue()
    # Need to trim end-of-line whitespace to prevent snapshot diffs after pre-commit removes the whitespace
    trimmed = '\n'.join([line.rstrip() for line in rendered.split('\n')])
    return trimmed
