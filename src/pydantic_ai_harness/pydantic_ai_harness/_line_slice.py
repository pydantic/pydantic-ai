"""Windows over stored text, bounded in both axes, for tools that page through a large result.

The caps are the ones `ToolOutputLimits`' `read_tool_result` enforces; `WorkLog`'s `read_work_log`
slices with the functions here. `pattern` is a literal substring (not a regex), so a model-supplied
value cannot hang the host with catastrophic backtracking. `limit` is clamped and the joined output is
capped, so one call can never return an unbounded amount of text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai.exceptions import ModelRetry

MAX_READ_LINES = 1_000
"""Hard cap on lines returned by one read."""

MAX_READ_CHARS = 50_000
"""Hard cap on characters (or bytes, with `unit='bytes'`) returned by one read."""

ReadUnit = Literal['lines', 'bytes']
"""What a read's `offset` and `limit` count."""


def check_window(offset: int, limit: int, unit: ReadUnit = 'lines', pattern: str | None = None) -> int:
    """Validate a model-supplied window and return `limit` clamped to the unit's cap.

    Called before the stored text is read, so a bad argument costs no I/O.
    """
    if offset < 0:
        raise ModelRetry('`offset` must be >= 0.')
    if limit < 1:
        raise ModelRetry('`limit` must be >= 1.')
    if unit == 'bytes':
        if pattern is not None:
            raise ModelRetry("`pattern` filters lines, so it cannot be combined with `unit='bytes'`.")
        return min(limit, MAX_READ_CHARS)
    return min(limit, MAX_READ_LINES)


def _utf8_len(text: str) -> int:
    return len(text.encode('utf-8', 'surrogatepass'))


@dataclass(frozen=True)
class LineSlice:
    """The lines a window selected, joined and capped at `MAX_READ_CHARS`."""

    text: str
    shown: int
    """Lines in the window, before the character cap."""
    total: int
    """Lines matching `pattern`, or all lines without one."""
    capped: bool
    """Whether the character cap cut `text`."""
    whole: int
    """Lines of the window that `text` holds whole, counted from the side the cap kept: `shown` unless `capped`."""
    cut_at: int | None
    """The byte offset where the part of a line the cap cut, and `text` leaves out, starts.

    `None` when no line was cut."""


def slice_lines(
    text: str, *, offset: int, limit: int, from_end: bool, pattern: str | None, cap_keeps_end: bool = False
) -> LineSlice:
    """Filter `text` by `pattern`, then take `limit` lines after skipping `offset`, from the start or the end.

    `limit` is used as given: clamp it with `check_window` first. A window over `MAX_READ_CHARS` keeps its
    first characters, or with `cap_keeps_end` its last, so a read from the end returns the lines nearest it.
    The line the cap cuts is named by `cut_at`, so a `unit='bytes'` read can return the rest of it.
    """
    starts: list[int] = []
    lines: list[str] = []
    position = 0
    for line in text.splitlines(keepends=True):
        body = line.splitlines()[0]
        if pattern is None or pattern in body:
            starts.append(position)
            lines.append(body)
        position += _utf8_len(line)

    total = len(lines)
    if from_end:
        stop = max(0, total - offset)
        first = max(0, stop - limit)
    else:
        first = min(offset, total)
        stop = first + limit
    window = lines[first:stop]

    body = '\n'.join(window)
    if len(body) <= MAX_READ_CHARS:
        return LineSlice(text=body, shown=len(window), total=total, capped=False, whole=len(window), cut_at=None)
    # A cut that falls on a line break leaves the line before it whole: the character past the cut is that break.
    if cap_keeps_end:
        kept, edge = body[-MAX_READ_CHARS:], body[-MAX_READ_CHARS - 1]
    else:
        kept, edge = body[:MAX_READ_CHARS], body[MAX_READ_CHARS]
    whole = kept.count('\n') + (1 if edge == '\n' else 0)
    cut_at: int | None = None
    if cap_keeps_end and edge != '\n' and not kept.startswith('\n'):
        # The cut line's head is missing: it starts at the line's start.
        cut_at = starts[first + len(window) - whole - 1]
    elif not cap_keeps_end and edge != '\n' and not kept.endswith('\n'):
        cut_at = starts[first + whole] + _utf8_len(kept[kept.rfind('\n') + 1 :])
    return LineSlice(text=kept, shown=len(window), total=total, capped=True, whole=whole, cut_at=cut_at)


@dataclass(frozen=True)
class ByteSlice:
    """A window of `text`'s UTF-8 bytes, decoded with a multi-byte sequence cut at either edge as U+FFFD."""

    text: str
    start: int
    end: int
    total: int


def slice_bytes(text: str, *, offset: int, limit: int, from_end: bool) -> ByteSlice:
    """Take `limit` bytes of `text` after skipping `offset`, from the start or the end.

    For output without useful line breaks, such as minified JSON. `limit` is used as given: clamp it with
    `check_window` first.
    """
    data = text.encode('utf-8', 'surrogatepass')
    total = len(data)
    if from_end:
        end = max(0, total - offset)
        start = max(0, end - limit)
    else:
        start = min(offset, total)
        end = min(total, start + limit)
    return ByteSlice(text=data[start:end].decode('utf-8', errors='replace'), start=start, end=end, total=total)
