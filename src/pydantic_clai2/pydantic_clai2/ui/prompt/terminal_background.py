"""Ask the terminal for its own background colour, which the default theme otherwise cannot know.

Standard library only: the entry point asks before the splash, ahead of the heavy imports.
"""

import os
import re
import select
import sys
import time
from typing import IO

TIMEOUT = 0.25
"""Seconds to wait for a terminal that answers neither query, so startup never hangs on one."""

_QUERY = '\x1b]11;?\x1b\\\x1b[c'
"""OSC 11 asks for the background. Terminals answer in order and all answer DA1 (`CSI c`), so its reply ends the wait."""
_COLOUR = re.compile(
    rb'\x1b\]11;rgba?:([0-9a-f]{1,4})/([0-9a-f]{1,4})/([0-9a-f]{1,4})(?:/[0-9a-f]{1,4})?(?:\x07|\x1b\\)',
    re.IGNORECASE,
)
_ATTRIBUTES = re.compile(rb'\x1b\[\?[0-9;]*c')
_DARK_SLOTS = frozenset((0, 1, 2, 3, 4, 5, 6, 8))
"""`COLORFGBG` background slots that are dark in the usual ANSI palettes."""


def detect_background(*, stdin: IO[str], stdout: IO[str], timeout: float = TIMEOUT) -> str | None:
    """The background of the terminal `stdin` and `stdout` are, as `#RRGGBB`; `None` when it is unknown.

    The terminal's answer to OSC 11 comes first, then `COLORFGBG`, which tells only light from dark.
    Nothing is asked unless both are a terminal, so piped or redirected output never receives the query.
    """
    if sys.platform == 'win32' or os.environ.get('TERM') == 'dumb' or not (stdout.isatty() and stdin.isatty()):
        return None
    if (answer := _ask(stdin.fileno(), stdout=stdout, timeout=timeout)) is not None:
        return answer
    slot = os.environ.get('COLORFGBG', '').rpartition(';')[2]
    if not slot.isdigit():
        return None
    return '#000000' if int(slot) in _DARK_SLOTS else '#FFFFFF'


def _ask(fd: int, *, stdout: IO[str], timeout: float) -> str | None:
    """Ask with OSC 11, never over keys already typed, and wait at most `timeout` seconds.

    Keys typed while the answer is awaited arrive mixed with it and are read with it.
    """
    import fcntl
    import termios
    import tty

    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    answer = b''
    try:
        # `TCSANOW`, not `setcbreak`'s default `TCSAFLUSH`, keeps keys typed ahead of the prompt queued for it.
        tty.setcbreak(fd, termios.TCSANOW)
        # Reading the answer would swallow keys already typed, which the editor shows in its draft.
        if int.from_bytes(fcntl.ioctl(fd, termios.FIONREAD, bytes(4)), sys.byteorder):
            return None
        stdout.write(_QUERY)
        stdout.flush()
        deadline = time.monotonic() + timeout
        while not _ATTRIBUTES.search(answer) and (left := deadline - time.monotonic()) > 0:
            if not select.select([fd], [], [], left)[0]:
                break
            # One byte at a time, so keys typed after the DA1 reply stay queued for the editor.
            answer += os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
    if (match := _COLOUR.search(answer)) is None:
        return None
    return '#' + ''.join(f'{round(int(part, 16) * 255 / (16 ** len(part) - 1)):02X}' for part in match.groups())
