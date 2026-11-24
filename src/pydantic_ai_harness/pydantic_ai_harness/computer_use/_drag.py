"""How desktop backends move the pointer through a drag, so a web page's drag-and-drop registers it."""

from __future__ import annotations as _annotations

import itertools
import math
from collections.abc import Sequence

from pydantic_ai.computer_use import Point

DRAG_STEP_PAUSE = 0.05
"""Seconds before each move of a drag and before letting go. A web page's drag-and-drop misses a drag whose events all
arrive at once."""

_MAX_STEP_PIXELS = 200
"""The longest single move in a drag, in pixels."""


def drag_steps(path: Sequence[Point]) -> list[Point]:
    """The points to move through along `path` after pressing at its first point, ending at its last.

    Each straight part of the path is split into moves of at most `_MAX_STEP_PIXELS` pixels, and into at least two
    moves: a browser starts a drag only after the pointer has moved more than once. The drag ends 1 pixel past the
    last point and back. Chromium answers each pointer position with what the page said about an earlier one, so
    without one more position after the page accepts the drop, it cancels the drop on release. In live runs, a web
    page's drag-and-drop still missed a few drags in 100.
    """
    steps: list[Point] = []
    for start, end in itertools.pairwise(path):
        count = max(2, math.ceil(math.dist((start.x, start.y), (end.x, end.y)) / _MAX_STEP_PIXELS))
        steps += [
            Point(x=start.x + (end.x - start.x) * step // count, y=start.y + (end.y - start.y) * step // count)
            for step in range(1, count + 1)
        ]
    end = path[-1]
    return [*steps, Point(x=end.x + 1, y=end.y), end]
