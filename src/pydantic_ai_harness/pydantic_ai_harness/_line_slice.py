"""Line windows over stored text, bounded in both axes, for tools that page through a large result.

Shared by `ToolOutputLimits`' `read_tool_result` and `WorkLog`'s `read_work_log`. `pattern` is a
literal substring (not a regex), so a model-supplied value cannot hang the host with catastrophic
backtracking. `limit` is clamped and the joined output is capped, so one call can never return an
unbounded amount of text.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.exceptions import ModelRetry

MAX_READ_LINES = 1_000
"""Hard cap on lines returned by one read."""

MAX_READ_CHARS = 50_000
"""Hard cap on characters returned by one read."""


def check_window(offset: int, limit: int) -> int:
    """Validate a model-supplied window and return `limit` clamped to `MAX_READ_LINES`.

    Called before the stored text is read, so a bad argument costs no I/O.
    """
    if offset < 0:
        raise ModelRetry('`offset` must be >= 0.')
    if limit < 1:
        raise ModelRetry('`limit` must be >= 1.')
    return min(limit, MAX_READ_LINES)


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


def slice_lines(
    text: str, *, offset: int, limit: int, from_end: bool, pattern: str | None, cap_keeps_end: bool = False
) -> LineSlice:
    """Filter `text` by `pattern`, then take `limit` lines after skipping `offset`, from the start or the end.

    `limit` is used as given: clamp it with `check_window` first. A window over `MAX_READ_CHARS` keeps its
    first characters, or with `cap_keeps_end` its last, so a read from the end returns the lines nearest it.
    """
    lines = text.splitlines()
    if pattern is not None:
        lines = [line for line in lines if pattern in line]

    total = len(lines)
    if from_end:
        end = max(0, total - offset)
        window = lines[max(0, end - limit) : end]
    else:
        window = lines[offset : offset + limit]

    body = '\n'.join(window)
    if len(body) <= MAX_READ_CHARS:
        return LineSlice(text=body, shown=len(window), total=total, capped=False, whole=len(window))
    # A cut that falls on a line break leaves the line before it whole: the character past the cut is that break.
    if cap_keeps_end:
        kept, edge = body[-MAX_READ_CHARS:], body[-MAX_READ_CHARS - 1]
    else:
        kept, edge = body[:MAX_READ_CHARS], body[MAX_READ_CHARS]
    whole = kept.count('\n') + (1 if edge == '\n' else 0)
    return LineSlice(text=kept, shown=len(window), total=total, capped=True, whole=whole)
