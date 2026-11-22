"""Paint the panel's first frame from the last launch before this launch's imports, so the prompt shows at once.

A bare `pax` in a terminal saves the shape of its inline panel's first frame (`PromptSurface.startup_frame`), with
the terminal's size and colour settings. The next bare `pax` in a terminal with the same ones paints that shape before
importing anything heavy, and the panel paints over it when it is ready. Keys typed meanwhile wait in the terminal, unechoed,
for the editor. Standard library only: this runs before the heavy imports it covers.
"""

import os
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO, cast

INLINE_ON = '\x1b[?25l\x1b[?7l'
"""Hidden cursor and no autowrap for the inline panel, so each row takes exactly one terminal row."""
INLINE_OFF = '\x1b[0m\x1b[?7h\x1b[?25h'
PASTE_ON = '\x1b[?2004h'
"""Bracketed paste, so a paste typed before the editor reads waits as one paste, not as lines each sending Enter."""
PASTE_OFF = '\x1b[?2004l'
_KEY_VARIABLES = ('TERM', 'COLORTERM', 'NO_COLOR', 'FORCE_COLOR')
"""Variables that change how the rows' colours are written, so a frame saved under other values is not reused."""


class FirstFrame:
    """The frame painted early, until the panel takes its rows over."""

    def __init__(self) -> None:
        """Nothing saved or shown; `paint` arms a launch."""
        self.path: Path | None = None
        """Where this launch saves its first frame; `None` outside a bare `pax` in a terminal."""
        self.rows = 0
        """Rows painted early, which the panel has not taken over yet."""
        self.held: list[tuple[TextIO, str]] = []
        """What was written to standard output or error while the rows show, with the stream it was for."""
        self._theme = ''
        self._size = (0, 0)
        """The terminal's columns and lines when the rows were painted: a resize since then reflowed them."""
        self._modes: list[int | list[bytes | int]] | None = None
        """The terminal's input modes before echo was turned off, given back before the editor reads."""
        self._streams: tuple[TextIO, TextIO] | None = None
        """Standard output and error, while `sys.stdout` and `sys.stderr` hold what is written to them."""

    @property
    def holding(self) -> bool:
        """Whether output waits for the panel, which paints it over the early rows."""
        return self.rows > 0

    def paint(self, path: Path, *, theme: str) -> None:
        """Arm this launch to save its first frame at `path`, and paint the one saved there if it fits.

        `theme` is the configured theme, whose colours the saved rules are drawn in.
        """
        if os.name == 'nt' or not (sys.stdin.isatty() and sys.stdout.isatty()) or os.getenv('TERM') == 'dumb':
            return
        self.path, self._theme = path, theme
        try:
            key, *rows = path.read_text(encoding='utf-8').split('\n')
        except (OSError, UnicodeDecodeError, ValueError):
            return
        if key != self._key() or not rows:
            return
        import termios  # Not on Windows, which returned above.

        modes = termios.tcgetattr(sys.stdin)
        quiet = termios.tcgetattr(sys.stdin)
        # Typed keys wait unechoed, Enter as `\r`, as the editor's raw mode reads them; Ctrl-C still stops the
        # launch. TCSANOW keeps what was typed before this; the default TCSAFLUSH would discard it.
        quiet[0] &= ~termios.ICRNL
        quiet[3] &= ~(termios.ECHO | termios.ICANON)
        # Each change is recorded before it is made, so `close` undoes it however startup ends, Ctrl-C included.
        self._modes = modes
        termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
        size = os.get_terminal_size(sys.stdout.fileno())
        self._size = (size.columns, size.lines)
        self.rows = len(rows)
        sys.stdout.write(
            '\x1b[?2026h' + INLINE_ON + PASTE_ON + '\r' + '\r\n'.join(f'{row}\x1b[0m' for row in rows) + '\x1b[?2026l'
        )
        sys.stdout.flush()
        # A print, traceback, or usage error would land among the early rows, so it waits until they are gone;
        # standard error redirected away from the terminal goes where it was sent.
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = cast(TextIO, _Holding(sys.stdout, self))
        if sys.stderr.isatty():
            sys.stderr = cast(TextIO, _Holding(sys.stderr, self))

    def adopt(self) -> tuple[int, tuple[int, int], str]:
        """Hand the early rows to the panel: how many, the terminal size they were painted at, and what was held."""
        rows, self.rows = self.rows, 0
        return rows, self._size, ''.join(text for _, text in self._release())

    def save(self, frame: Callable[[], list[str] | None]) -> None:
        """Save the panel's first frame for the next launch, once; `None`, for a panel on the alternate screen, forgets it.

        `frame` is only called when this launch saves one, as it wraps the transcript.
        """
        path, self.path = self.path, None
        if path is None:
            return
        rows = frame()
        with suppress(OSError):
            if rows is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text('\n'.join([self._key(), *rows]), encoding='utf-8')

    def restore_input(self) -> None:
        """Give the terminal back its input modes, before the editor's raw mode saves them."""
        modes, self._modes = self._modes, None
        if modes is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSANOW, modes)

    def close(self) -> None:
        """At exit: erase early rows the panel never took over, then write what was held to its stream."""
        self.restore_input()
        if rows := self.rows:
            self.rows = 0
            sys.stdout.write(
                '\x1b[0m\r' + (f'\x1b[{rows - 1}A' if rows > 1 else '') + '\x1b[J' + INLINE_OFF + PASTE_OFF
            )
            sys.stdout.flush()
        for stream, text in self._release():
            stream.write(text)
            stream.flush()

    def _release(self) -> list[tuple[TextIO, str]]:
        if self._streams is not None:
            (sys.stdout, sys.stderr), self._streams = self._streams, None
        held, self.held = self.held, []
        return held

    def _key(self) -> str:
        """The terminal's size, colour settings, and theme, which a saved frame must match to be painted."""
        size = os.get_terminal_size(sys.stdout.fileno())
        values = (str(size.columns), str(size.lines), *(os.getenv(name, '') for name in _KEY_VARIABLES), self._theme)
        return '\t'.join(values)


class _Holding:
    """Stands in for a standard stream while the early rows show: writes wait, then pass straight through.

    Whatever takes the stream meanwhile, such as the console the panel writes through, keeps working after.
    """

    def __init__(self, stream: TextIO, frame: FirstFrame) -> None:
        self._stream = stream
        self._frame = frame

    def write(self, text: str) -> int:
        if not self._frame.holding:
            return self._stream.write(text)
        self._frame.held.append((self._stream, text))
        return len(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


current = FirstFrame()
"""This process's early frame: `__main__` paints it, the live prompt takes it over.

On the module because it exists before anything the launch builds, as `stderr_relay`'s terminal does.
"""
