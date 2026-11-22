"""Markdown tables that never split a word: wrapped columns when the words fit, one block per row when not."""

from collections.abc import Sequence

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible_length, wrap_ansi
from termflow.ansi.utils import visible
from termflow.render.style import RenderStyle
from termflow.render.table import (
    TABLE_HORIZ,
    TableRenderState,
    render_table_bottom,
    render_table_complete,
    render_table_row,
    render_table_separator,
    render_table_top,
)


def table_lines(
    *,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignments: Sequence[str],
    width: int,
    margin: str,
    style: RenderStyle,
) -> list[str]:
    """Render a table whose cells are already styled, `width` cells wide after `margin`.

    A table whose cells fit unwrapped renders as Termflow draws it. Otherwise every column
    gets at least its longest word and the room left goes to the columns that wrap, so text
    breaks only between words. A table whose longest words cannot sit side by side at this
    width becomes one block of `Header: value` lines per row, so no word is cut in half.
    """
    cells = [header, *rows]
    columns = max(len(row) for row in cells)
    natural = [
        max((visible_length(row[i].strip()) for row in cells if i < len(row)), default=0) for i in range(columns)
    ]
    words = [max((_longest_word(row[i]) for row in cells if i < len(row)), default=0) for i in range(columns)]
    # Each cell is padded by a space on either side, and every column has a border on its left plus one on the right.
    room = width - (3 * columns + 1)
    if sum(natural) <= room:
        return render_table_complete(header, rows, alignments, width, margin, style)
    if sum(words) > room:
        return _stacked(header=header, rows=rows, width=width, margin=margin, style=style)
    state = TableRenderState(column_widths=_share(natural=natural, words=words, room=room), alignments=list(alignments))
    lines = [render_table_top(state, margin, style)]
    lines += render_table_row(header, state, width, margin, style, is_header=True)
    lines.append(render_table_separator(state, margin, style))
    for row in rows:
        lines += render_table_row(row, state, width, margin, style)
    lines.append(render_table_bottom(state, margin, style))
    return lines


def _longest_word(cell: str) -> int:
    """The widest run a wrap may not break: a run of narrow characters between spaces, or one wide character.

    Lines may break between any two wide (CJK or emoji) characters, as text in those scripts does.
    """
    longest = run = 0
    for char in visible(cell):
        cells = visible_length(char)
        if char.isspace() or cells > 1:
            longest = max(longest, run, cells if cells > 1 else 0)
            run = 0
        else:
            run += cells
    return max(longest, run)


def _share(*, natural: list[int], words: list[int], room: int) -> list[int]:
    """Column widths that start at each column's longest word and split what is left by how much more each wants."""
    widths = list(words)
    wants = [full - least for full, least in zip(natural, words)]
    spare = room - sum(widths)
    total = sum(wants)
    for i, want in enumerate(wants):
        widths[i] += spare * want // total
    # Rounding down leaves fewer cells than columns; give them to the first columns still short of their full width.
    left = room - sum(widths)
    for i, full in enumerate(natural):
        if left and widths[i] < full:
            widths[i] += 1
            left -= 1
    return widths


def _stacked(
    *, header: Sequence[str], rows: Sequence[Sequence[str]], width: int, margin: str, style: RenderStyle
) -> list[str]:
    if not rows:
        return [
            margin + part
            for part in wrap_ansi(f'{BOLD_ON}{"  ".join(cell.strip() for cell in header)}{BOLD_OFF}', width)
        ]
    rule = f'{margin}{fg_color(style.symbol)}{TABLE_HORIZ * width}{RESET}'
    lines: list[str] = []
    for index, row in enumerate(rows):
        if index:
            lines.append(rule)
        for i, cell in enumerate(row):
            label = header[i].strip() if i < len(header) else ''
            line = f'{BOLD_ON}{label}:{BOLD_OFF} {cell.strip()}'
            lines += [margin + part for part in wrap_ansi(line, width)]
    return lines
