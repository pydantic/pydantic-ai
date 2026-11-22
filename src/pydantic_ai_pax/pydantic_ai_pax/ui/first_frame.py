"""Paint the panel's first frame from the last launch before this launch's imports, so the prompt shows at once.

A bare `pax` in a terminal saves the rows its inline panel first painted, with the terminal's size and colour
settings. The next bare `pax` in a terminal with the same ones paints those rows again before importing anything
heavy, and the panel paints over them when it is ready. Keys typed meanwhile wait in the terminal, unechoed,
for the editor. Standard library only: this runs before the heavy imports it covers.
"""

import io
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TextIO

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
        self._size = (0, 0)
        """The terminal's columns and lines when the rows were painted: a resize since then reflowed them."""
        self._modes: list[int | list[bytes | int]] | None = None
        """The terminal's input modes before echo was turned off, given back before the editor reads."""
        self._stderr: TextIO | None = None
        """Standard error, while `sys.stderr` holds what is written to it until the early rows are gone."""
        self._held = io.StringIO()

    @property
    def holding(self) -> bool:
        """Whether output waits for the panel, which paints it over the early rows."""
        return self.rows > 0

    def paint(self, path: Path) -> None:
        """Arm this launch to save its first frame at `path`, and paint the one saved there if it fits."""
        if os.name == 'nt' or not (sys.stdin.isatty() and sys.stdout.isatty()) or os.getenv('TERM') == 'dumb':
            return
        self.path = path
        try:
            key, *rows = path.read_text(encoding='utf-8').split('\n')
        except (OSError, UnicodeDecodeError, ValueError):
            return
        if key != _key() or not rows:
            return
        import termios  # Not on Windows, which returned above.

        modes = termios.tcgetattr(sys.stdin)
        quiet = termios.tcgetattr(sys.stdin)
        # Typed keys wait unechoed, Enter as `\r`, as the editor's raw mode reads them; Ctrl-C still stops the
        # launch. TCSANOW keeps what was typed before this; the default TCSAFLUSH would discard it.
        quiet[0] &= ~termios.ICRNL
        quiet[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
        self._modes = modes
        sys.stdout.write(
            '\x1b[?2026h' + INLINE_ON + PASTE_ON + '\r' + '\r\n'.join(f'{row}\x1b[0m' for row in rows) + '\x1b[?2026l'
        )
        sys.stdout.flush()
        self.rows = len(rows)
        size = os.get_terminal_size(sys.stdout.fileno())
        self._size = (size.columns, size.lines)
        # A traceback or usage error would print among the early rows, so it waits until they are gone; one
        # redirected away from the terminal goes where it was sent.
        if sys.stderr.isatty():
            self._stderr, sys.stderr = sys.stderr, self._held

    def adopt(self) -> tuple[int, tuple[int, int], str]:
        """Hand the early rows to the panel: how many, the terminal size they were painted at, and what stderr got."""
        rows, self.rows = self.rows, 0
        return rows, self._size, self._release_stderr()

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
                path.write_text('\n'.join([_key(), *rows]), encoding='utf-8')

    def restore_input(self) -> None:
        """Give the terminal back its input modes, before the editor's raw mode saves them."""
        modes, self._modes = self._modes, None
        if modes is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSANOW, modes)

    def close(self) -> None:
        """At exit: erase early rows the panel never took over, then print what standard error got meanwhile."""
        self.restore_input()
        if rows := self.rows:
            self.rows = 0
            sys.stdout.write(
                '\x1b[0m\r' + (f'\x1b[{rows - 1}A' if rows > 1 else '') + '\x1b[J' + INLINE_OFF + PASTE_OFF
            )
            sys.stdout.flush()
        if held := self._release_stderr():
            sys.stderr.write(held)
            sys.stderr.flush()

    def _release_stderr(self) -> str:
        original, self._stderr = self._stderr, None
        if original is None:
            return ''
        sys.stderr = original
        held, self._held = self._held.getvalue(), io.StringIO()
        return held


def _key() -> str:
    """The terminal's size and colour settings, which a saved frame must match to be painted."""
    size = os.get_terminal_size(sys.stdout.fileno())
    return '\t'.join([str(size.columns), str(size.lines), *(os.getenv(name, '') for name in _KEY_VARIABLES)])


current = FirstFrame()
"""This process's early frame: `__main__` paints it, the live prompt takes it over.

On the module because it exists before anything the launch builds, as `stderr_relay`'s terminal does.
"""
