"""Mouse selection over the live panel's painted cells.

The panel reports mouse buttons to scroll with the wheel, which stops most terminals from
selecting text themselves. So the selection happens here instead, as in a terminal's own: a
left-button drag in reading order, a double-click on a word, a triple-click on a line, and a drag
after either extends by words or lines. The release copies it out. Reporting also stops terminals
opening links, so a click on a URL finds it in the painted cells to open.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from termflow.live import ScreenBuffer
from termflow.live.buffer import REVERSE, WIDE_TAIL

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
URL = re.compile(r'https?://[^\s<>`]+')
"""A bare `http(s)` URL in plain text, before `trim_url` leaves out its trailing punctuation."""
REPEAT_INTERVAL = 0.5
"""Seconds from one press to the next on the same cell that make a double- or triple-click, as in Ghostty."""
WORD_BOUNDARIES = frozenset('\'"│`|:;,()[]{}<>$')
"""With blanks, the characters a double-click's word stops at: Ghostty's default `selection-word-chars`."""


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
    """A left-button selection from `anchor` to `head`, kept highlighted until cleared."""

    anchor: Cell | None = None
    head: Cell | None = None
    """Where the selection ends now; after a single press `None` until the held button moves, so a click selects nothing."""
    held: bool = False
    clicks: int = 1
    """The presses that started it: 1 selects cells, 2 whole words, and 3 whole lines."""
    pressed_at: float = -math.inf
    """When the last press came, so a press on its cell soon after repeats it."""

    def clear(self) -> bool:
        """Forget the selection, as when the cells under it move; `True` if cells were highlighted."""
        highlighted = self.anchor is not None and self.head is not None
        self.anchor = self.head = None
        self.held = False
        return highlighted

    def feed(self, report: MouseReport, *, now: float) -> bool:
        """Track presses, drags, and releases of the left button; `True` when a release ends a selection.

        A second press on the same cell within `REPEAT_INTERVAL` selects the word there, a third the
        line, and a fourth starts over.
        """
        if report.released:
            if report.button != LEFT:
                return False  # SGR reports name the released button; the left one may still be held.
            ended = self.held and self.head is not None
            self.held = False
            return ended
        if report.button == LEFT:
            repeat = report.cell == self.anchor and now - self.pressed_at <= REPEAT_INTERVAL and self.clicks < 3
            self.clicks = self.clicks + 1 if repeat else 1
            self.anchor, self.held, self.pressed_at = report.cell, True, now
            self.head = report.cell if self.clicks > 1 else None
        elif report.button == LEFT | MOTION and self.held:
            self.head = report.cell
            if report.cell != self.anchor:
                self.pressed_at = -math.inf  # A press after a drag starts a new selection, wherever it lands.
        return False

    def span(self, frame: ScreenBuffer, *, rows: int, joins: Sequence[bool]) -> range:
        """Frame indexes from the earlier end to the later one, inclusive, within the top `rows`.

        The surface supplies the selectable rows: normally just the transcript, or the question
        widget too while it owns the panel. Ordinary editor and status rows remain excluded.
        After a double- or triple-click, both ends widen to the word or line under them; `joins`
        says which rows wrap on from the row above, as `url_at` reads it.
        """
        if self.anchor is None or self.head is None:
            return range(0)
        width = frame.width

        def index(cell: Cell) -> int:
            row, column = cell
            if row >= rows:
                return rows * width - 1
            return max(row, 0) * width + min(max(column, 0), width - 1)

        start, end = (index(cell) for cell in sorted((self.anchor, self.head)))
        if start % width and frame.chars[start] == WIDE_TAIL:
            start -= 1  # The right half of a double-width character holds no text; its left half does.
        if self.clicks > 1:
            within = _in_word if self.clicks == 2 else _any_cell
            start = _reach(frame, start, -1, rows=rows, joins=joins, within=within)
            end = _reach(frame, end, 1, rows=rows, joins=joins, within=within)
        return range(start, end + 1)

    def highlight(
        self, frame: ScreenBuffer, *, rows: int, joins: Sequence[bool], previous: ScreenBuffer | None
    ) -> None:
        """Show the selected cells in reverse video, or drop the selection if output moved them.

        The selection holds screen cells, not transcript rows, so new output that scrolls the
        view, or a widget that redraws, must not leave it over text the user never selected.
        """
        span = self.span(frame, rows=rows, joins=joins)
        if previous is not None and (
            previous.size != frame.size or previous.chars[span.start : span.stop] != frame.chars[span.start : span.stop]
        ):
            self.clear()
            return
        for index in span:
            # Set, not toggled: a cell that is already reverse video, such as the cursor, still looks selected.
            frame.attrs[index] |= REVERSE

    def text(self, frame: ScreenBuffer, *, rows: int, joins: Sequence[bool]) -> str:
        """The selected characters, without trailing blanks or blank rows after them.

        Each row ends its line unless the next one wraps on from it, as `joins` says.
        """
        span = self.span(frame, rows=rows, joins=joins)
        width = frame.width
        last = (span.stop - 1) // width
        text = line = ''
        for row in range(span.start // width, last + 1):
            cells = ''.join(frame.chars[max(span.start, row * width) : min(span.stop, (row + 1) * width)])
            if row < last and _joined(joins, row + 1):
                line += cells[:-1] if _padding(frame, (row + 1) * width - 1, joins) else cells
            else:
                text += (line + cells).rstrip() + '\n'
                line = ''
        return text.rstrip('\n')


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
    first, last = row, row
    while first > 0 and _joined(joins, first):
        first -= 1
    while last + 1 < rows and _joined(joins, last + 1):
        last += 1
    cells = frame.chars[first * width : (last + 1) * width]
    text = ''.join(cells)
    offset = len(''.join(cells[: (row - first) * width + column]))
    for match in URL.finditer(text):
        url = trim_url(match[0])
        if match.start() <= offset < match.start() + len(url):
            above = match.start() == 0 and first == 0 and _joined(joins, 0)
            below = match.end() == len(text) and last == rows - 1 and _joined(joins, rows)
            return None if above or below else url
    return None


def _joined(joins: Sequence[bool], index: int) -> bool:
    """Whether row `index` wraps on from the row above, which a row beyond `joins` never does."""
    return index < len(joins) and joins[index]


def _in_word(char: str) -> bool:
    """Whether a cell belongs to a double-click's word; a double-width character's tail cell does."""
    return not char.isspace() and char not in WORD_BOUNDARIES


def _any_cell(char: str) -> bool:
    return True


def _reach(
    frame: ScreenBuffer, index: int, step: int, *, rows: int, joins: Sequence[bool], within: Callable[[str], bool]
) -> int:
    """The farthest cell from `index` in direction `step` over cells `within` accepts, staying in its line.

    A row joins the row above only where it wraps on from it, so a line ends at a row that does not.
    """
    width, end = frame.width, rows * frame.width
    while within(frame.chars[index]):
        after = index + step
        if 0 <= after < end and _padding(frame, after, joins):
            after += step
        if not (0 <= after < end and within(frame.chars[after])):
            break
        if after // width != index // width and not _joined(joins, max(index, after) // width):
            break
        index = after
    return index


def _padding(frame: ScreenBuffer, index: int, joins: Sequence[bool]) -> bool:
    """Whether cell `index` is the blank the panel leaves at a row's end when a double-width character opens the next row.

    That blank is no text, though a space that happened to land there reads the same.
    """
    width = frame.width
    return (
        index % width == width - 1
        and frame.chars[index] == ' '
        and _joined(joins, index // width + 1)
        and frame.chars[index + 2 : index + 3] == [WIDE_TAIL]
    )
