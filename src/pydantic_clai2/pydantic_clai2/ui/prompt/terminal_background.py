"""The terminal's own background, which the default theme otherwise cannot know."""

import os
from typing import IO

_DARK_SLOTS = frozenset((0, 1, 2, 3, 4, 5, 6, 8))
"""`COLORFGBG` background slots that are dark in the usual ANSI palettes."""


def detect_background(*, stdout: IO[str]) -> str | None:
    """The background of the terminal `stdout` is, as `#000000` or `#FFFFFF`; `None` when it is unknown.

    `COLORFGBG`, which some terminals set, tells only light from dark. It describes the terminal, so
    output that is not a terminal has no background.
    """
    slot = os.environ.get('COLORFGBG', '').rpartition(';')[2]
    if not slot.isdigit() or not stdout.isatty():
        return None
    return '#000000' if int(slot) in _DARK_SLOTS else '#FFFFFF'
