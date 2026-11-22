"""Find the terminal's background colour, so the default theme can pick colours that read on it."""

import os
import re
import select
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPress

_QUERY = b'\x1b]11;?\x07\x1b[c'
"""Ask for the background colour (OSC 11), then for the device attributes (DA1).

Terminals that do not report their background still answer DA1, so its reply ends the wait at once.
"""
_BACKGROUND_REPLY = re.compile(
    rb'\x1b\]11;rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})[^\x07\x1b]*(?:\x07|\x1b\\)'
)
_ANY_BACKGROUND_REPLY = re.compile(rb'\x1b\]11;[^\x07\x1b]*(?:\x07|\x1b\\)?')
"""Any background reply, also in a form `_BACKGROUND_REPLY` does not read, which must not reach the prompt."""
_ATTRIBUTES_REPLY = re.compile(rb'\x1b\[\?[0-9;]*c')
_TIMEOUT = 0.5
"""How long to wait for a terminal that answers neither query, such as one behind a slow SSH link."""


def terminal_background() -> str | None:
    """The background as `#RRGGBB`: what the terminal reports, else what `COLORFGBG` says, else `None` (unknown)."""
    return _reported() or _from_colorfgbg(os.getenv('COLORFGBG', ''))


def _reported() -> str | None:
    """Ask the terminal on stdin and stdout, when both are one; keys typed meanwhile go on to the prompt."""
    if sys.platform == 'win32' or os.getenv('TERM') == 'dumb' or not (os.isatty(0) and os.isatty(1)):
        return None
    import termios

    saved = termios.tcgetattr(0)
    modes = termios.tcgetattr(0)
    modes[3] &= ~(termios.ICANON | termios.ECHO)  # The reply is read as it comes, and is not echoed.
    modes[0] &= ~termios.ICRNL  # Enter typed meanwhile stays Enter for the prompt, not Ctrl-J.
    termios.tcsetattr(0, termios.TCSANOW, modes)
    try:
        os.write(1, _QUERY)
        received = b''
        deadline = time.monotonic() + _TIMEOUT
        while (
            not _ATTRIBUTES_REPLY.search(received)
            and (remaining := deadline - time.monotonic()) > 0
            and select.select([0], [], [], remaining)[0]
            and (chunk := os.read(0, 1024))  # Empty once the terminal is gone.
        ):
            received += chunk
    finally:
        termios.tcsetattr(0, termios.TCSANOW, saved)  # Not `TCSAFLUSH`, which would drop keys typed meanwhile.
    background = _BACKGROUND_REPLY.search(received)
    if typed := _ATTRIBUTES_REPLY.sub(b'', _ANY_BACKGROUND_REPLY.sub(b'', received)):
        _keep_for_the_prompt(typed)
    if background is None:
        return None
    return '#' + ''.join(f'{int(channel, 16) * 255 // (16 ** len(channel) - 1):02X}' for channel in background.groups())


def _keep_for_the_prompt(typed: bytes) -> None:
    """Hand keys typed while waiting for the reply to the prompt, through prompt-toolkit's type-ahead store."""
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    keys: list[KeyPress] = []
    parser = Vt100Parser(keys.append)
    parser.feed(typed.decode(errors='replace'))
    parser.flush()
    store_typeahead(get_app_session().input, keys)


def _from_colorfgbg(value: str) -> str | None:
    """The background `COLORFGBG` (`FG;BG`, ANSI slot numbers) names: white for slot 7 or 15, else `None`."""
    background = value.rpartition(';')[2]
    return '#FFFFFF' if background in ('7', '15') else None
