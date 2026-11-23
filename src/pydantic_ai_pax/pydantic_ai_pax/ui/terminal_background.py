"""Find the terminal's background colour, so the default theme can pick colours that read on it."""

import os
import re
import select
import sys
import time

from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.input.typeahead import store_typeahead
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding import KeyPress

if sys.platform != 'win32':  # pragma: no branch  # No `termios` on Windows, where pax never asks.
    import termios

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
    if not _on_a_terminal():
        return None
    background = _BACKGROUND_REPLY.search(_read(query=_QUERY, timeout=_TIMEOUT))
    if background is None:
        return None
    return '#' + ''.join(f'{int(channel, 16) * 255 // (16 ** len(channel) - 1):02X}' for channel in background.groups())


def drain_late_reply() -> None:
    """Before the prompt reads keys, drop a reply that came after `terminal_background` stopped waiting.

    A terminal behind a slow link can answer after the wait; its reply would otherwise reach the prompt
    as typed text. Keys typed meanwhile go on to the prompt.
    """
    if _on_a_terminal():
        _read(query=b'', timeout=0)


def _on_a_terminal() -> bool:
    return sys.platform != 'win32' and os.getenv('TERM') != 'dumb' and os.isatty(0) and os.isatty(1)


def _read(*, query: bytes, timeout: float) -> bytes:
    """Send `query`, then read stdin until the device attributes reply or `timeout`; typed keys go to the prompt."""
    saved = termios.tcgetattr(0)
    modes = termios.tcgetattr(0)
    modes[3] &= ~(termios.ICANON | termios.ECHO)  # The reply is read as it comes, and is not echoed.
    modes[0] &= ~termios.ICRNL  # Enter typed meanwhile stays Enter for the prompt, not Ctrl+J.
    termios.tcsetattr(0, termios.TCSANOW, modes)
    try:
        os.write(1, query)
        received = b''
        deadline = time.monotonic() + timeout
        while (
            not _ATTRIBUTES_REPLY.search(received)
            and select.select([0], [], [], max(0, deadline - time.monotonic()))[0]
            and (chunk := os.read(0, 1024))  # Empty once the terminal is gone.
        ):
            received += chunk
            if timeout and time.monotonic() >= deadline:  # A paste does not extend the wait; a drain reads all.
                break
    finally:
        termios.tcsetattr(0, termios.TCSANOW, saved)  # Not `TCSAFLUSH`, which would drop keys typed meanwhile.
    if typed := _ATTRIBUTES_REPLY.sub(b'', _ANY_BACKGROUND_REPLY.sub(b'', received)):
        _keep_for_the_prompt(typed)
    return received


def _keep_for_the_prompt(typed: bytes) -> None:
    """Hand keys typed while waiting for the reply to the prompt, through prompt-toolkit's type-ahead store."""
    keys: list[KeyPress] = []
    parser = Vt100Parser(keys.append)
    parser.feed(typed.decode(errors='replace'))
    parser.flush()
    store_typeahead(get_app_session().input, keys)


def _from_colorfgbg(value: str) -> str | None:
    """The background `COLORFGBG` (`FG;BG`, ANSI slot numbers) names: white for slot 7 or 15, else `None`."""
    background = value.rpartition(';')[2]
    return '#FFFFFF' if background in ('7', '15') else None
