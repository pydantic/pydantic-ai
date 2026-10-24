"""How desktop backends move the pointer through a drag, so a web page's drag-and-drop registers it."""

from __future__ import annotations as _annotations

import itertools
import math
from collections.abc import Sequence

from pydantic_ai.workspaces import Point

DRAG_PRESS_PAUSE = 0.1
"""Seconds between pressing the button and the first move, and between the last move and letting go: a web page's
drag-and-drop misses a drag whose events all arrive at once (checked live in Firefox and Chromium)."""

DRAG_STEP_PAUSE = 0.05
"""Seconds between the moves of a drag, for the same reason."""

_STEP_PIXELS = 200
"""The longest single move in a drag."""


def drag_steps(path: Sequence[Point]) -> list[Point]:
    """The points to move through along `path` after pressing at its first point, ending at its last.

    Each leg is split into moves of at most `_STEP_PIXELS`, and at least two: a browser only starts a drag, and only
    lets it land, after the pointer moves more than once.
    """
    steps: list[Point] = []
    for start, end in itertools.pairwise(path):
        count = max(2, math.ceil(math.dist((start.x, start.y), (end.x, end.y)) / _STEP_PIXELS))
        steps += [
            Point(x=start.x + (end.x - start.x) * step // count, y=start.y + (end.y - start.y) * step // count)
            for step in range(1, count + 1)
        ]
    return steps
