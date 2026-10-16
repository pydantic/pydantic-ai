"""Run a `!` command on a pseudo-terminal, handing it CLAI's terminal once it goes full screen.

A command that sees a terminal behaves as it does in a shell: it can prompt, and a full-screen program
such as `nano`, `vim`, `less`, or `htop` can draw. Its output is captured as with a pipe, without
colours, until the program switches to the alternate screen, as full-screen programs do. From then on
it owns CLAI's terminal and nothing more is captured. No list of programs is involved.

POSIX only: Windows has no pseudo-terminal in the standard library, so commands there keep pipes.
"""

import asyncio
import contextlib
import os
import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_clai2.ui.menus.menu_worker import full_terminal
from pydantic_clai2.ui.prompt.prompt_resize import resize_notifications

FULL_SCREEN = re.compile(rb'\x1b\[\?(?:1049|1047|47)h')
"""Switching to the alternate screen, the moment a program starts drawing a full screen."""

_ESCAPE = re.compile(
    rb'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[PX^_][^\x1b]*\x1b\\|[ -/]*[0-OQ-WYZ\\`-~])'
)
"""Control and colour sequences, which a captured transcript cannot draw."""
_MAX_ESCAPE = 4096
_LINE_END = re.compile(rb'\r+\n')
_INTERRUPT = b'\x03'
_DRAIN_GRACE = 0.1

COMMAND_ENVIRONMENT = {'PAGER': 'cat', 'GIT_PAGER': 'cat'}
"""A terminal makes `git log` and friends page; their output is wanted in the transcript instead."""


@dataclass(frozen=True, kw_only=True)
class Terminal:
    """The terminal CLAI runs on: the file descriptors keys arrive on and output goes to."""

    input: int = 0
    output: int = 1


def attached_terminal() -> Terminal | None:
    """CLAI's terminal, or `None` on Windows or when its input or output is not a terminal."""
    if sys.platform == 'win32' or not (os.isatty(0) and os.isatty(1)):
        return None
    return Terminal()


class TerminalText:
    """Turn terminal output into transcript text: no control sequences, and newline line ends.

    A sequence or line end split across reads is held until the rest arrives.
    """

    def __init__(self) -> None:
        """Start with nothing held."""
        self._held = b''

    def release(self, data: bytes) -> bytes:
        """What an earlier call held back, followed by `data`, now held no longer."""
        data, self._held = self._held + data, b''
        return data

    def feed(self, data: bytes, *, final: bool = False) -> bytes:
        """The text in `data`, after what an earlier call held back."""
        data, self._held = self._held + data, b''
        if not final:
            cut = len(data)
            escape = data.rfind(b'\x1b')
            if escape >= 0 and not _ESCAPE.match(data, escape) and cut - escape < _MAX_ESCAPE:
                cut = escape
            cut = len(data[:cut].rstrip(b'\r'))
            data, self._held = data[:cut], data[cut:]
        # The terminal turns a program's `\r\n` into `\r\r\n`; a stray escape byte is not text.
        return _LINE_END.sub(b'\n', _ESCAPE.sub(b'', data)).replace(b'\x1b', b'')


class PtyRun:
    """One command's pseudo-terminal, and the relay between it and CLAI's terminal.

    Until the command goes full screen, its output goes to `capture` and keys go to it, except Ctrl-C,
    which calls `interrupt` so CLAI interrupts the command as it does any other. Once it goes full
    screen, `full_terminal()` takes CLAI's display off the screen and the relay passes everything through.
    """

    def __init__(
        self, *, terminal: Terminal, capture: Callable[[bytes], None], interrupt: Callable[[], object]
    ) -> None:
        """Open the pseudo-terminal at the size of CLAI's terminal."""
        self.terminal = terminal
        self.capture = capture
        self.interrupt = interrupt
        self.handed_over = False
        self.leader, self.follower = os.openpty()
        # Keys the command is not reading are dropped rather than blocking CLAI.
        os.set_blocking(self.leader, False)
        self.closed = asyncio.Event()
        self._text = TerminalText()
        self._stack = contextlib.ExitStack()
        self._resize()

    def command(self, command: str) -> str:
        """`command` for the system shell, starting by making the pseudo-terminal its controlling terminal.

        The shell runs in a new session, and a session leader opening a terminal takes it as its own.
        Programs that read `/dev/tty`, such as `less` and `sudo`, need that.
        """
        return f'exec 3<>{shlex.quote(os.ttyname(self.follower))} 3>&-\n{command}'

    def start(self) -> None:
        """Relay output and keys, with CLAI's terminal in raw mode so each key reaches the command."""
        from prompt_toolkit.input.vt100 import raw_mode

        loop = self._loop = asyncio.get_running_loop()
        self._close_follower()
        self._stack.enter_context(raw_mode(self.terminal.input))
        self._stack.enter_context(resize_notifications(self._resize_soon))
        loop.add_reader(self.leader, self._output)
        self._stack.callback(loop.remove_reader, self.leader)
        loop.add_reader(self.terminal.input, self._key)
        self._stack.callback(loop.remove_reader, self.terminal.input)

    def pass_through(self, data: bytes) -> None:
        """Write `data` to CLAI's terminal as it is."""
        view = memoryview(data)
        while view:
            view = view[os.write(self.terminal.output, view) :]

    async def drain(self) -> None:
        """Wait briefly for the last output; a background job may hold the pseudo-terminal open."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.closed.wait(), _DRAIN_GRACE)

    def close(self) -> None:
        """Stop relaying, give CLAI's display its screen back, and restore the terminal's modes."""
        if not self.handed_over:
            self.capture(self._text.feed(b'', final=True))
        self._stack.close()
        self._close_follower()
        os.close(self.leader)

    def _close_follower(self) -> None:
        """Close CLAI's copy of the command's end, so the command's exit ends the output."""
        if self.follower >= 0:
            os.close(self.follower)
            self.follower = -1

    def _output(self) -> None:
        try:
            data = os.read(self.leader, 65536)
        except BlockingIOError:  # pragma: no cover -- a spurious wakeup.
            return
        except OSError:  # pragma: lax no cover -- Linux reports the other end's last close as an error.
            data = b''
        if not data:
            asyncio.get_running_loop().remove_reader(self.leader)
            self.closed.set()
        elif self.handed_over:
            self.pass_through(data)
        elif full := FULL_SCREEN.search(data := self._text.release(data)):
            self._hand_over(data, full.start())
        else:
            self.capture(self._text.feed(data))

    def _hand_over(self, data: bytes, start: int) -> None:
        self.capture(self._text.feed(data[:start], final=True))
        self.handed_over = True
        self._stack.enter_context(full_terminal())
        self.pass_through(data[start:])

    def _key(self) -> None:
        data = os.read(self.terminal.input, 4096)
        if not self.handed_over and _INTERRUPT in data:
            data = data[: data.index(_INTERRUPT)]
            self.interrupt()
        with contextlib.suppress(OSError):  # The command exited, or is not reading what is typed.
            os.write(self.leader, data)

    def _resize_soon(self) -> None:
        """Resize from the event loop: this runs in a signal handler."""
        self._loop.call_soon_threadsafe(self._resize)

    def _resize(self) -> None:
        """Give the pseudo-terminal CLAI's size, which signals the command when it changed."""
        import fcntl
        import termios

        with contextlib.suppress(OSError):
            fcntl.ioctl(
                self.leader, termios.TIOCSWINSZ, fcntl.ioctl(self.terminal.output, termios.TIOCGWINSZ, b'\0' * 8)
            )
