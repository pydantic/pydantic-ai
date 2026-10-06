"""Mouse selection over the live panel's painted cells.

The panel reports mouse buttons to scroll with the wheel, which stops most terminals from
selecting text themselves. So a left-button drag selects here instead, in reading order like a
terminal's own selection, and the release copies it out.
"""

import re
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
