"""Mouse selection over the live panel's painted cells.

The panel reports mouse buttons to scroll with the wheel, which stops most terminals from
selecting text themselves. So a left-button drag selects here instead, in reading order like a
terminal's own selection, and the release copies it out. It also stops them opening links, so a
click on a URL finds it in the painted cells to open.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from termflow.live import ScreenBuffer
from termflow.live.buffer import REVERSE

Cell: TypeAlias = tuple[int, int]
"""A zero-based `(row, column)` on screen."""

LEFT = 0
MOTION = 32
"""Added to the button while it moves held down."""
WHEEL_UP = 64
WHEEL_DOWN = 65
_MODIFIERS = 4 | 8 | 16
"""Shift, Alt, and Ctrl add these to the button."""
_REPORT = re.compile(r'\x1b\[<(\d{1,5});(\d{1,5});(\d{1,5})([mM])')
"""Bounded fields: a malformed report too long for `int` is not a report, rather than an error."""
_URL = re.compile(r'https?://[^\s<>`]+')


@dataclass(frozen=True, kw_only=True)
class MouseReport:
    """One decoded SGR mouse report."""

    button: int
    """The button, motion, and wheel code, without modifiers."""
    cell: Cell
    released: bool


def mouse_report(data: str) -> MouseReport | None:
    """Decode `CSI < button ; column ; row M|m`, or `None` for anything else."""
    report = _REPORT.fullmatch(data)
    if report is None:
        return None
    button, column, row, final = report.groups()
    return MouseReport(button=int(button) & ~_MODIFIERS, cell=(int(row) - 1, int(column) - 1), released=final == 'm')


@dataclass(kw_only=True)
class Selection:
    """A left-button drag from `anchor` to `head`, kept highlighted until cleared."""

    anchor: Cell | None = None
    head: Cell | None = None
    """Where the drag is now; `None` until the held button moves, so a click selects nothing."""
    held: bool = False

    def clear(self) -> bool:
        """Forget the selection, as when the cells under it move; `True` if cells were highlighted."""
        highlighted = self.anchor is not None and self.head is not None
        self.anchor = self.head = None
        self.held = False
        return highlighted

    def feed(self, report: MouseReport) -> bool:
        """Track a press, drag, and release of the left button; `True` when a release ends a drag."""
        if report.released:
            if report.button != LEFT:
                return False  # SGR reports name the released button; the left one may still be held.
            ended = self.held and self.head is not None
            self.held = False
            return ended
        if report.button == LEFT:
            self.anchor, self.head, self.held = report.cell, None, True
        elif report.button == LEFT | MOTION and self.held:
            self.head = report.cell
        return False

    def span(self, *, width: int, rows: int) -> range:
        """Frame indexes from the earlier end to the later one, inclusive, within the top `rows`.

        Only the transcript is selectable: a drag into the editor or footer below it stops at its
        last row, so a copy never picks up the draft, borders, or status.
        """
        if self.anchor is None or self.head is None:
            return range(0)
        start, end = sorted((self.anchor, self.head))

        def index(cell: Cell) -> int:
            row, column = cell
            if row >= rows:
                return rows * width - 1
            return max(row, 0) * width + min(max(column, 0), width - 1)

        return range(index(start), index(end) + 1)

    def highlight(self, frame: ScreenBuffer, *, rows: int, previous: ScreenBuffer | None) -> None:
        """Show the selected cells in reverse video, or drop the selection if output moved them.

        The selection holds screen cells, not transcript rows, so new output that scrolls the
        view, or a widget that redraws, must not leave it over text the user never dragged across.
        """
        span = self.span(width=frame.width, rows=rows)
        if previous is not None and (
            previous.size != frame.size or previous.chars[span.start : span.stop] != frame.chars[span.start : span.stop]
        ):
            self.clear()
            return
        for index in span:
            # Set, not toggled: a cell that is already reverse video, such as the cursor, still looks selected.
            frame.attrs[index] |= REVERSE

    def text(self, frame: ScreenBuffer, *, rows: int) -> str:
        """The selected characters, one line per row, without trailing blanks or blank rows after them."""
        span = self.span(width=frame.width, rows=rows)
        width = frame.width
        return '\n'.join(
            ''.join(frame.chars[max(span.start, row * width) : min(span.stop, (row + 1) * width)]).rstrip()
            for row in range(span.start // width, (span.stop - 1) // width + 1)
        ).rstrip('\n')


def trim_url(url: str) -> str:
    """Leave trailing punctuation and unbalanced closing brackets out of a bare URL, as GFM does."""
    unopened = {')': url.count(')') - url.count('('), ']': url.count(']') - url.count('[')}
    end = len(url)
    while url[end - 1] in '.,;:!?\'"*_~)]':
        char = url[end - 1]
        if char in unopened:
            if unopened[char] <= 0:
                break
            unopened[char] -= 1
        end -= 1
    return url[:end]


def url_at(frame: ScreenBuffer, cell: Cell, *, rows: int, joins: Sequence[bool]) -> str | None:
    """The `http(s)` URL painted under `cell` in the frame's top `rows`, if all of it is on screen.

    `joins[i]` says whether row `i` wraps on from row `i - 1`, and `joins[rows]` whether a row below
    the transcript wraps on from its last one. Only those genuine wraps join rows, so a URL that ends
    at the right edge never runs into the next line. A URL cut off by the top or bottom of the
    transcript is not returned, since its address is incomplete. Painted cells carry no hyperlink,
    so the URL is read from the visible text.
    """
    row, column = cell
    width = frame.width
    if not (0 <= row < rows and 0 <= column < width):
        return None

    def joined(index: int) -> bool:
        return index < len(joins) and joins[index]

    first, last = row, row
    while first > 0 and joined(first):
        first -= 1
    while last + 1 < rows and joined(last + 1):
        last += 1
    cells = frame.chars[first * width : (last + 1) * width]
    text = ''.join(cells)
    offset = len(''.join(cells[: (row - first) * width + column]))
    for match in _URL.finditer(text):
        url = trim_url(match[0])
        if match.start() <= offset < match.start() + len(url):
            above = match.start() == 0 and first == 0 and joined(0)
            below = match.end() == len(text) and last == rows - 1 and joined(rows)
            return None if above or below else url
    return None
