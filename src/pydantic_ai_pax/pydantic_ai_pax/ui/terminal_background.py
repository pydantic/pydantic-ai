"""Find the terminal's background colour, so the default theme can pick colours that read on it."""

import os
import re
import select
import sys
import time

_QUERY = b'\x1b]11;?\x07\x1b[c'
"""Ask for the background colour (OSC 11), then for the device attributes (DA1).

Terminals that do not report their background still answer DA1, so its reply ends the wait at once.
"""
_BACKGROUND_REPLY = re.compile(
    rb'\x1b\]11;rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})[^\x07\x1b]*(?:\x07|\x1b\\)'
)
_ATTRIBUTES_REPLY = re.compile(rb'\x1b\[\?[0-9;]*c')
_TIMEOUT = 0.5
"""How long to wait for a terminal that answers neither query, such as one behind a slow SSH link."""


def terminal_background() -> str | None:
    """The background as `#RRGGBB`: what the terminal reports, else what `COLORFGBG` says, else `None` (unknown)."""
    return _reported() or _from_colorfgbg(os.getenv('COLORFGBG', ''))


def _reported() -> str | None:
    """Ask the terminal on stdin and stdout, when both are one and no keys typed ahead are waiting to be read."""
    if sys.platform == 'win32' or os.getenv('TERM') == 'dumb' or not (os.isatty(0) and os.isatty(1)):
        return None
    import termios

    saved = termios.tcgetattr(0)
    modes = termios.tcgetattr(0)
    modes[3] &= ~(termios.ICANON | termios.ECHO)  # The reply is read as it comes, and is not echoed.
    termios.tcsetattr(0, termios.TCSANOW, modes)
    try:
        # Out of line mode, keys typed ahead without Enter are readable: reading the reply would take them.
        if select.select([0], [], [], 0)[0]:
            return None
        os.write(1, _QUERY)
        received = b''
        deadline = time.monotonic() + _TIMEOUT
        while (
            not _ATTRIBUTES_REPLY.search(received)
            and select.select([0], [], [], max(0, deadline - time.monotonic()))[0]
        ):
            received += os.read(0, 1024)
    finally:
        termios.tcsetattr(0, termios.TCSANOW, saved)  # Not `TCSAFLUSH`, which would drop keys typed meanwhile.
    if (background := _BACKGROUND_REPLY.search(received)) is None:
        return None
    return '#' + ''.join(f'{int(channel, 16) * 255 // (16 ** len(channel) - 1):02X}' for channel in background.groups())


def _from_colorfgbg(value: str) -> str | None:
    """The background `COLORFGBG` (`FG;BG`, ANSI slot numbers) names: white for slot 7 or 15, else `None`."""
    background = value.rpartition(';')[2]
    return '#FFFFFF' if background in ('7', '15') else None
