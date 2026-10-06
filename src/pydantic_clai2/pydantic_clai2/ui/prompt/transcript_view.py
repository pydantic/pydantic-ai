"""The transcript as a termflow.live widget: follows new output, or stays where the user scrolled."""

from termflow.live import Region, Widget

from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.rendering import theme

_Position = tuple[int, int]
"""A row: the item's id and the row within it."""

SCROLLED_HINT = ' ↓ more below · PgDn '


class TranscriptView(Widget):
    """Draw the rows that end at the anchor, or the newest rows while following.

    The anchor is the bottom row's item id and row, so output arriving while the user reads
    older rows leaves them in place. Scrolling back to the end follows again.
    """

    focusable = False

    def __init__(self, transcript: TranscriptBuffer) -> None:
        """Start following."""
        self.transcript = transcript
        self.anchor: _Position | None = None
        self._width = 1
        self._height = 0
        self._memo: dict[int, tuple[str, ...]] = {}

    def follow(self) -> None:
        """Show the newest rows again."""
        self.anchor = None

    def _rows(self, item: int) -> tuple[str, ...]:
        if item not in self._memo:
            self._memo[item] = self.transcript.rows(item, width=self._width)
        return self._memo[item]

    def _last(self) -> _Position:
        end = self.transcript.end
        return end, len(self._rows(end)) - 1

    def _before(self, position: _Position) -> _Position | None:
        item, row = position
        if row > 0:
            return item, row - 1
        for earlier in range(item - 1, self.transcript.ids().start - 1, -1):
            if rows := self._rows(earlier):
                return earlier, len(rows) - 1
        return None

    def _after(self, position: _Position) -> _Position | None:
        item, row = position
        if row + 1 < len(self._rows(item)):
            return item, row + 1
        for later in range(item + 1, self.transcript.end + 1):
            if self._rows(later):
                return later, 0
        return None

    def _bottom(self) -> _Position:
        """The anchor, clamped to what still exists at this width, or the newest row."""
        if self.anchor is None or self.anchor[0] not in self.transcript.ids():
            self.anchor = None
            return self._last()
        item, row = self.anchor
        rows = len(self._rows(item))
        if rows == 0:
            return self._before(self.anchor) or self._last()
        return item, min(row, rows - 1)

    def scroll(self, rows: int) -> None:
        """Move back (positive) or forward (negative) by `rows`; reaching the end follows again."""
        self._memo = {}
        position: _Position | None = self._bottom()
        step = self._before if rows > 0 else self._after
        for _ in range(abs(rows)):
            moved = step(position)
            if moved is None:
                break
            position = moved
        # Keep a full page above the bottom row: scrolling stops at the oldest row.
        first: _Position | None = (self.transcript.ids().start, 0)
        if not self._rows(first[0]):
            first = self._after(first)
        for _ in range(self._height - 1):
            first = self._after(first) if first is not None else None
        if first is None or position >= self._last():
            self.anchor = None
            return
        self.anchor = max(position, first)

    def window(self, *, width: int, height: int) -> list[str]:
        """The rows to show, top-aligned when the transcript is shorter than `height`."""
        if (width, height) != (self._width, self._height):
            self._width, self._height = width, height
        self._memo = {}
        if height <= 0:
            return []
        position: _Position | None = self._bottom()
        rows: list[str] = []
        while position is not None and len(rows) < height:
            rows.append(self._rows(position[0])[position[1]])
            position = self._before(position)
        rows.reverse()
        return rows

    def draw(self, region: Region, focused: bool) -> None:
        """Paint into `region`, with a hint while scrolled away from the newest output."""
        for y, row in enumerate(self.window(width=max(1, region.width), height=region.height)):
            region.ansi(0, y, row)
        if self.anchor is not None and region.height > 0:
            hint = SCROLLED_HINT[: region.width]
            region.ansi(region.width - len(hint), region.height - 1, f'{theme.sgr(theme.MUTED)}\x1b[7m{hint}\x1b[0m')
