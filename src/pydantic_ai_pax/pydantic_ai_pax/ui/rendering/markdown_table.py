"""Markdown tables that keep words whole: wrapped columns when the words fit, one block per row when not.

This works around Termflow 1.0's `TableRenderState.cap_widths_to_max`, which shrinks an
overflowing table's columns to a fixed minimum of 8 cells instead of the column's longest
word, so the cell wrap then splits words mid-character. Remove this module once Termflow
keeps a column at least as wide as its longest word.
"""

import os
import re
from collections.abc import Sequence
from itertools import zip_longest

from termflow.ansi import BOLD_OFF, BOLD_ON, RESET, fg_color, visible, visible_length, wrap_ansi
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

_BREAKS = re.compile('[ \t]+')
"""Where Termflow's wrap breaks a line; a word longer than its column is split anywhere."""


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
    width becomes one block of `Header: value` lines per row, where
    only a word wider than the whole width is split.
    """
    # The cap Termflow's own table renderer applies, so every layout stays within it.
    try:
        cap = int(os.environ['TERMFLOW_MAX_TABLE_WIDTH']) - visible_length(margin)
    except (KeyError, ValueError):
        cap = width
    width = min(width, cap) if cap > 0 else width
    natural: list[int] = []
    words: list[int] = []
    for row in (header, *rows):
        for i, cell in enumerate(row):
            if i == len(natural):
                natural.append(0)
                words.append(0)
            natural[i] = max(natural[i], visible_length(cell.strip()))
            # Termflow's wrap adds up each code point's width, so a joined emoji counts as all its parts.
            words[i] = max(words[i], *(sum(map(visible_length, word)) for word in _BREAKS.split(visible(cell))))
    columns = len(natural)
    # Each cell is padded by a space on either side, and every column has a border on its left plus one on the right.
    room = width - (3 * columns + 1)
    if sum(natural) <= room:
        return render_table_complete(header, rows, alignments, width, margin, style)
    if sum(words) > room:
        return _stacked(header=header, rows=rows, width=width, margin=margin, style=style)
    state = TableRenderState(
        column_widths=_share(natural=[max(pair) for pair in zip(natural, words)], words=words, room=room),
        alignments=list(alignments),
    )
    lines = [render_table_top(state, margin, style)]
    lines += render_table_row(header, state, width, margin, style, is_header=True)
    lines.append(render_table_separator(state, margin, style))
    for row in rows:
        lines += render_table_row(row, state, width, margin, style)
    lines.append(render_table_bottom(state, margin, style))
    return lines


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
    rule = f'{margin}{fg_color(style.symbol)}{TABLE_HORIZ * width}{RESET}'
    lines: list[str] = []
    for index, row in enumerate(rows or [[]]):
        if index:
            lines.append(rule)
        for label, cell in zip_longest(header, row, fillvalue=''):
            label = label.strip()
            line = f'{BOLD_ON}{label}:{BOLD_OFF} {cell.strip()}' if label else cell.strip()
            lines += [margin + part for part in wrap_ansi(line, width)]
    return lines
