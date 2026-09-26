"""Stdlib-only startup splash adapted from Code Puppy, not a run-time spinner."""

import io
import os
import shutil
import sys
import threading
from typing import TextIO

from ._splash_art import LABEL, PYRAMID
from .theme import LIGHT_PURPLE, LITHIUM, PURPLE, SUGAR, sgr


class Splash:
    """Own the alternate screen while heavyweight imports run."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Inspect terminal suitability without importing UI dependencies."""
        self._stream = sys.stdout
        self._originals = (sys.stdout, sys.stderr)
        self._buffers = (io.StringIO(), io.StringIO())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._size = shutil.get_terminal_size()
        self._enabled = (
            enabled
            and self._stream.isatty()
            and not os.getenv('NO_COLOR')
            and os.getenv('TERM') != 'dumb'
            and self._size.columns >= 46
            and self._size.lines >= 22
            and os.name != 'nt'
        )

    def start(self) -> None:
        """Start one animation thread and defer import-time output."""
        if not self._enabled or self._thread is not None:
            return
        sys.stdout, sys.stderr = self._buffers
        self._thread = threading.Thread(target=self._run, name='clai-startup-splash', daemon=True)
        self._thread.start()

    def frame(self, phase: int) -> str:
        """Render the same diagonal neon sheen and tiered pyramid as Code Puppy."""
        rows = [''.join(' ░▒█'[int(tier)] for tier in row).ljust(44) for row in PYRAMID]
        if self._size.lines >= 30:
            rows.extend(['', *[line.center(44) for line in LABEL]])
        base = (sgr(PURPLE), sgr(LITHIUM), sgr(LIGHT_PURPLE, bold=True))
        hot = (sgr(LITHIUM), sgr(LIGHT_PURPLE), sgr(SUGAR, bold=True))
        lines: list[str] = []
        for y, row in enumerate(rows):
            line = ' ' * max(0, (self._size.columns - 44) // 2)
            for x, glyph in enumerate(row):
                tier = 0 if glyph == '░' else 2 if glyph == '█' else 1
                palette = hot if (x + 2 * y - phase) % 70 < 7 else base
                line += glyph if glyph == ' ' else palette[tier] + glyph
            lines.append(line + '\x1b[0m')
        top = max(1, (self._size.lines - len(rows)) // 2 + 1)
        return f'\x1b[{top};1H' + '\n'.join(lines)

    def _run(self) -> None:
        try:
            self._stream.write('\x1b[?1049h\x1b[?25l\x1b[2J')
            phase = 0
            while not self._stop.is_set():
                self._stream.write('\x1b[?2026h' + self.frame(phase) + '\x1b[?2026l')
                self._stream.flush()
                phase = (phase + 2) % 70
                self._stop.wait(0.033)
        except OSError:
            pass

    def stop(self) -> None:
        """Join the painter, restore the screen, and replay captured output."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        try:
            self._stream.write('\x1b[?1049l\x1b[?25h')
            self._stream.flush()
        finally:
            if sys.stdout is self._buffers[0]:
                sys.stdout = self._originals[0]
            if sys.stderr is self._buffers[1]:
                sys.stderr = self._originals[1]
            for buffer, original in zip(self._buffers, self._originals):
                self._replay(buffer, original)

    @staticmethod
    def _replay(buffer: io.StringIO, original: TextIO) -> None:
        original.write(buffer.getvalue())
        original.flush()
