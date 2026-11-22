"""Paint the panel's first frame from the last launch before this launch's imports, so the prompt shows at once.

A bare `pax` in a terminal saves the rows its inline panel first painted, with the terminal's size and colour
settings. The next bare `pax` in a terminal with the same ones paints those rows again before importing anything
heavy, and the panel paints over them when it is ready. Keys typed meanwhile wait in the terminal, unechoed,
for the editor. Standard library only: this runs before the heavy imports it covers.
"""

import io
import json
import os
import sys
from pathlib import Path

INLINE_ON = '\x1b[?25l\x1b[?7l'
"""Hidden cursor and no autowrap for the inline panel, so each row takes exactly one terminal row."""
INLINE_OFF = '\x1b[0m\x1b[?7h\x1b[?25h'
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
        self._attrs: list[int | list[bytes | int]] | None = None
        """The terminal's input modes before echo was turned off, restored before the editor reads."""
        self._stderr: tuple[io.TextIOBase, io.StringIO] | None = None
        """Standard error and the buffer holding what was written to it while the frame shows."""

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
            saved = json.loads(path.read_text(encoding='utf-8'))
            rows = saved['rows'] if saved['key'] == _key() else None
        except (OSError, ValueError, LookupError, TypeError):
            return
        if not isinstance(rows, list) or not rows or not all(isinstance(row, str) for row in rows):
            return
        import termios  # Not on Windows, which returned above.

        attrs = termios.tcgetattr(sys.stdin)
        quiet = termios.tcgetattr(sys.stdin)
        # Typed keys wait unechoed, Enter as `\r`, as the editor's raw mode would read them. Ctrl-C still stops
        # the launch. TCSANOW keeps what was typed before this.
        quiet[0] &= ~termios.ICRNL
        quiet[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
        self._attrs = attrs
        sys.stdout.write(
            '\x1b[?2026h' + INLINE_ON + '\r' + '\r\n'.join(f'{row}\x1b[0m' for row in rows) + '\x1b[?2026l'
        )
        sys.stdout.flush()
        self.rows = len(rows)
        # A traceback or usage error would print among the early rows; it waits for them to be erased.
        self._stderr = (sys.stderr, io.StringIO())
        sys.stderr = self._stderr[1]

    def adopt(self) -> tuple[int, str]:
        """Hand the early rows to the panel: how many there are, and what standard error got meanwhile."""
        rows, self.rows = self.rows, 0
        return rows, self._release_stderr()

    def save(self, rows: list[str] | None) -> None:
        """Save the panel's first frame for the next launch once, or forget the saved one when `rows` is `None`.

        `None` means the panel started on the alternate screen, where inline rows painted early would not be.
        """
        path, self.path = self.path, None
        if path is None:
            return
        try:
            if rows is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(json.dumps({'key': _key(), 'rows': rows}), encoding='utf-8')
        except OSError:
            pass

    def restore_input(self) -> None:
        """Give the terminal back its input modes, before the editor's raw mode saves them."""
        attrs, self._attrs = self._attrs, None
        if attrs is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSANOW, attrs)

    def close(self) -> None:
        """At exit: erase rows the panel never took over, and print what standard error got meanwhile."""
        self.restore_input()
        if rows := self.rows:
            self.rows = 0
            sys.stdout.write('\x1b[0m\r' + (f'\x1b[{rows - 1}A' if rows > 1 else '') + '\x1b[J' + INLINE_OFF)
            sys.stdout.flush()
        if held := self._release_stderr():
            sys.stderr.write(held)
            sys.stderr.flush()

    def _release_stderr(self) -> str:
        if self._stderr is None:
            return ''
        (original, held), self._stderr = self._stderr, None
        if sys.stderr is held:
            sys.stderr = original
        return held.getvalue()


def _key() -> list[object]:
    size = os.get_terminal_size(sys.stdout.fileno())
    return [size.columns, size.lines, *(os.getenv(name) for name in _KEY_VARIABLES)]


current = FirstFrame()
"""This process's early frame: `__main__` paints it, the live prompt takes it over."""
