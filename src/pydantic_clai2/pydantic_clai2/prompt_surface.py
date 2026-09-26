"""A pinned editor with a blank, debounced viewport during terminal resize."""

import io
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile
from threading import RLock
from typing import IO

from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]

from .prompt_transcript import TranscriptBuffer


class PromptSurface(io.StringIO):
    """Own terminal writes; redraw from retained output rather than old coordinates."""

    def __init__(
        self,
        *,
        output: IO[str],
        size: Callable[[], tuple[int, int]],
        clock: Callable[[], float] = time.monotonic,
        transcript: TranscriptBuffer | None = None,
    ) -> None:
        """Bind terminal IO and injectable geometry/time sources."""
        super().__init__()
        self.output = output
        self.size = size
        self.clock = clock
        self.transcript = transcript if transcript is not None else TranscriptBuffer()
        self._lock = RLock()
        self._geometry = (0, 0)
        self._rows: tuple[str, ...] = ()
        self._active = False
        self._partial = False
        self._resize_at: float | None = None
        self._resize_notice = False
        self._observed_size = (0, 0)
        self._deferred: IO[str] | None = None
        self._holds = 0

    def isatty(self) -> bool:
        """Preserve Rich and Termflow terminal detection."""
        return self.output.isatty()

    def resize_notice(self) -> None:
        """Mark a resize notification; signal handlers must not draw or erase."""
        if self._active:
            self._resize_notice = True
            self._resize_at = self.clock()

    def _check_resize(self, *, size: tuple[int, int]) -> None:
        if self._resize_notice or size != self._observed_size:
            self._resize_notice = False
            self._observed_size = size
            self._resize_at = self.clock()
            self._spool()
            self._transaction('\x1b[?25l\x1b[r\x1b[2J\x1b[1;1H')

    def _spool(self) -> None:
        if self._deferred is None:
            # Large output during a long drag or an open menu spills to a private
            # temp file rather than growing memory without bound or being dropped.
            self._deferred = SpooledTemporaryFile(max_size=1_000_000, mode='w+t', encoding='utf-8', newline='')

    @contextmanager
    def held(self) -> Generator[None]:
        """Spool writes while another widget owns the terminal, then replay them in order.

        Holds nest; output is replayed when the outermost one exits, unless a resize
        rebuild is still pending, which replays it once the viewport settles.
        """
        with self._lock:
            self._holds += 1
            self._spool()
        try:
            yield
        finally:
            with self._lock:
                self._holds -= 1
                if self._resize_at is None:
                    self._flush_deferred()

    def _emit(self, text: str) -> None:
        self.transcript.write(text)
        self.output.write(text.replace('\n', '\r\n') if self._active and self.output.isatty() else text)
        self.output.flush()

    def write(self, text: str) -> int:
        """Stream normally, or spool writes while the visible viewport is blank."""
        with self._lock:
            if self._active:
                self._check_resize(size=self.size())
            if self._deferred is not None:
                self._deferred.write(text)
            else:
                self._emit(text)
            if text:
                self._partial = not text.endswith('\n')
        return len(text)

    def flush(self) -> None:
        """Flush without repainting or ending an incomplete line."""
        with self._lock:
            self.output.flush()

    async def drain(self) -> None:
        """Settle an incomplete transcript line before a turn/menu boundary."""
        if self._partial:
            self.write('\n')

    def paint(self, rows: tuple[str, ...]) -> None:
        """Wait for 250 ms of stable size before rebuilding the viewport once."""
        with self._lock:
            width, height = self.size()
            width, height = max(1, width), max(2, height)
            rows = tuple(truncate(row, width) for row in rows[-(height - 2) :]) if height > 2 else ()
            if self._active:
                self._check_resize(size=(width, height))
            if self._resize_at is not None:
                if self.clock() - self._resize_at >= 0.25:
                    self._rebuild(rows=rows, width=width, height=height)
                return
            self._paint(rows=rows, width=width, height=height)

    def _paint(self, *, rows: tuple[str, ...], width: int, height: int) -> None:
        bottom = height - len(rows)
        changed_geometry = self._geometry != (width, height) or len(rows) != len(self._rows)
        parts: list[str] = []
        if not self._active:
            # Scroll only as far as the rows need, then return to the writer's row.
            # Jumping to the region bottom instead left a blank band under short
            # history, visible after startup and whenever a menu hands the screen back.
            up = f'\x1b[{len(rows)}A' if rows else ''
            parts.extend(
                ['\x1b[?25l\x1b[?2004h\x1b[>4;1m', '\r\n' * len(rows), up, '\x1b7', f'\x1b[1;{bottom}r', '\x1b8']
            )
            self._active = True
        elif changed_geometry:
            old_bottom = self._geometry[1] - len(self._rows)
            growth = max(0, old_bottom - bottom)
            parts.append('\x1b7')
            if growth:
                # Make room from the writer's actual position, not the old
                # region bottom. Reopening a popup can reuse the gap left when
                # it closed without scrolling another batch of blank lines.
                parts.extend(['\x1bD' * growth, f'\x1b[{growth}A', '\x1b7'])
            parts.extend([f'\x1b[1;{bottom}r', '\x1b8'])
            if bottom > old_bottom:
                parts.append('\x1b7')
                for row in range(old_bottom + 1, bottom + 1):
                    parts.append(f'\x1b[{row};1H\x1b[2K')
                parts.append('\x1b8')
        parts.append(self._row_changes(rows=rows, bottom=bottom, width=width, force=changed_geometry))
        if any(parts):
            self._transaction(''.join(parts))
        self._rows, self._geometry = rows, (width, height)
        self._observed_size = (width, height)

    def _row_changes(self, *, rows: tuple[str, ...], bottom: int, width: int, force: bool) -> str:
        parts: list[str] = []
        for index, row in enumerate(rows):
            if force or index >= len(self._rows) or row != self._rows[index]:
                clear = '\x1b[K' if visible_length(row) < width else ''
                parts.append(f'\x1b[{bottom + index + 1};1H\x1b[0m{row}\x1b[0m{clear}')
        return '\x1b7\x1b[?7l' + ''.join(parts) + '\x1b[?7h\x1b8' if parts else ''

    def _rebuild(self, *, rows: tuple[str, ...], width: int, height: int) -> None:
        bottom = height - len(rows)
        frame = self.transcript.frame(width=width, height=bottom)
        parts = ['\x1b[r\x1b[2J\x1b[1;1H', f'\x1b[1;{bottom}r']
        parts.append(self._row_changes(rows=rows, bottom=bottom, width=width, force=True))
        # Address each transcript row directly. Replaying it must not scroll old
        # output into native history a second time. Last row retains the writer's
        # column and delayed-wrap state for the next streaming chunk.
        for index, row in enumerate(frame.rows, start=bottom - len(frame.rows) + 1):
            parts.append(f'\x1b[{index};1H\x1b[0m{row}')
        parts.append(frame.continuation_style)
        self._transaction(''.join(parts))
        self._rows, self._geometry = rows, (width, height)
        self._observed_size = (width, height)
        self._resize_at = None
        self._resize_notice = False
        self._flush_deferred()

    def _flush_deferred(self) -> None:
        if self._holds:
            return
        deferred, self._deferred = self._deferred, None
        if deferred is not None:
            try:
                deferred.seek(0)
                while chunk := deferred.read(65536):
                    self._emit(chunk)
            finally:
                deferred.close()

    def _transaction(self, text: str) -> None:
        self.output.write('\x1b[?2026h' + text + '\x1b[?2026l')
        self.output.flush()

    def release(self) -> None:
        """Flush pending output and restore modes before a menu or shell takes over."""
        with self._lock:
            if not self._active:
                return
            try:
                width, height = self.size()
                width, height = max(1, width), max(2, height)
                if self._resize_at is not None or (width, height) != self._geometry:
                    rows = self._rows[-(height - 2) :] if height > 2 else ()
                    self._rebuild(rows=rows, width=width, height=height)
                if self._partial:
                    self._emit('\n')
                    self._partial = False
                bottom = self._geometry[1] - len(self._rows)
                # Resetting the margins homes the cursor, so keep the writer's own position.
                parts = ['\x1b7\x1b[r']
                for row in range(bottom + 1, self._geometry[1] + 1):
                    parts.append(f'\x1b[{row};1H\x1b[2K')
                parts.extend(['\x1b8', '\x1b[>4;0m\x1b[0m\x1b[?2004l\x1b[?25h'])
                self._transaction(''.join(parts))
            finally:
                if self._deferred is not None and not self._holds:
                    self._deferred.close()
                    self._deferred = None
                self._rows = ()
                self._active = False
                self._resize_at = None
