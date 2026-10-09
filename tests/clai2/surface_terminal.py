"""Small VT screen fixture for the cursor and screen operations the live panel emits.

Unlike StringIO assertions, this checks what remains on screen after rows move.
Resize deliberately retains existing row coordinates, reproducing the stale-band
case that bottom-anchoring tmux can hide. The alternate screen is kept apart from
the main one and its history, as terminals do. This is not a general terminal emulator.
"""

import io
import re


class SurfaceTerminal(io.StringIO):
    """Track screen cells, scrolling and the cursor for surface regression tests."""

    def __init__(self, *, width: int, height: int) -> None:
        super().__init__()
        self.width, self.height = width, height
        self.cells = [[' '] * width for _ in range(height)]
        self.row = self.column = 0
        self.wrap = True
        self.history: list[str] = []
        self.main: tuple[list[list[str]], tuple[int, int]] | None = None
        """The main screen and its cursor, kept while the alternate screen shows."""

    @property
    def alternate(self) -> bool:
        return self.main is not None

    def resize(self, *, width: int, height: int, bottom_anchored: bool = False) -> None:
        """Resize without moving old UI rows to the new screen bottom."""
        if bottom_anchored:  # pragma: lax no cover
            delta = height - self.height
            if delta > 0:
                restored = self.history[-delta:]
                del self.history[max(0, len(self.history) - delta) :]
                prefix = [[' '] * self.width for _ in range(delta - len(restored))]
                prefix.extend([list(line.ljust(self.width)) for line in restored])
                self.cells = prefix + self.cells
            elif delta < 0:
                self.history.extend(''.join(row).rstrip() for row in self.cells[:-delta])
                self.cells = self.cells[-delta:]
            self.row = max(0, self.row + delta)
        self.cells = [(row[:width] + [' '] * width)[:width] for row in self.cells[:height]]
        self.cells.extend([[' '] * width for _ in range(height - len(self.cells))])
        self.width, self.height = width, height
        self.row, self.column = min(self.row, height - 1), min(self.column, width - 1)

    def isatty(self) -> bool:
        return True

    def lines(self) -> list[str]:
        return [''.join(row).rstrip() for row in self.cells]

    def write(self, text: str) -> int:
        super().write(text)
        for token in re.findall(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?<>]*[A-Za-z]|\x1b.|[^\x1b]', text):
            if token.startswith('\x1b]'):
                continue  # OSC: hyperlinks and palettes have no cells
            if token in ('\n', '\x1bD'):
                self.advance()
            elif token == '\r':
                self.column = 0
            elif token.startswith('\x1b['):
                self.control(token)
            elif not token.startswith('\x1b') and token.isprintable():  # pragma: no branch
                if self.column == self.width:
                    self.column = 0
                    self.advance()
                self.cells[self.row][self.column] = token
                self.column = min(self.width if self.wrap else self.width - 1, self.column + 1)
        return len(text)

    def advance(self) -> None:
        if self.row == self.height - 1:
            if self.main is None:  # pragma: no branch
                self.history.append(''.join(self.cells[0]).rstrip())
            del self.cells[0]
            self.cells.append([' '] * self.width)
        else:
            self.row = min(self.row + 1, self.height - 1)

    def control(self, token: str) -> None:
        params, code = token[2:-1], token[-1]
        if token == '\x1b[?1049h' and self.main is None:
            self.main = (self.cells, (self.row, self.column))
            self.cells = [[' '] * self.width for _ in range(self.height)]
        elif token == '\x1b[?1049l' and self.main is not None:
            self.cells, (self.row, self.column) = self.main
            self.main = None
        elif code == 'H':
            row, col = (int(part) for part in params.split(';'))
            self.row = min(max(0, row - 1), self.height - 1)
            self.column = min(max(0, col - 1), self.width - 1)
        elif code == 'J' and params == '2':
            self.cells = [[' '] * self.width for _ in range(self.height)]
        elif token in ('\x1b[?7h', '\x1b[?7l'):
            self.wrap = code == 'h'
