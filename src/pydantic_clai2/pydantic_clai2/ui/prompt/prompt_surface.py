"""The live panel: transcript and pinned editor painted as termflow.live frames on the alternate screen."""

import io
import math
import re
import time
import webbrowser
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import RLock, Thread
from typing import IO

from termflow.live import Rect, ScreenBuffer, render_diff
from termflow.tui.layout import truncate

from pydantic_clai2.ui.prompt.prompt_selection import LEFT, WHEEL_DOWN, WHEEL_UP, Selection, mouse_report, url_at
from pydantic_clai2.ui.prompt.prompt_transcript import MarkdownBlock, Render, TranscriptBuffer
from pydantic_clai2.ui.prompt.text_clipboard import copy_text
from pydantic_clai2.ui.prompt.transcript_view import TranscriptView

ENTER = '\x1b[?1049h\x1b[?25l\x1b[?7l'
"""Alternate screen, hidden cursor, and no autowrap, which `render_diff` assumes."""
LEAVE = '\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l'
MODES_ON = '\x1b[?2004h\x1b[>4;1m\x1b[>5u\x1b[?1000h\x1b[?1002h\x1b[?1006h'
"""Bracketed paste, xterm modified keys, Kitty disambiguation with alternate keys, and SGR mouse buttons and drags.

Kitty keeps a flag stack per screen, so these are pushed after `ENTER` and popped before `LEAVE`.
"""
MODES_OFF = '\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[<u\x1b[>4;0m\x1b[?2004l'
FRAME_INTERVAL = 1 / 60
"""Writes repaint at most this often; the editor's refresh loop paints what is left."""
# Palette and other non-hyperlink OSC commands are meant for the terminal, not the transcript.
_TERMINAL_OSC = re.compile(r'\x1b\](?!8;)[^\x07\x1b]*(?:\x07|\x1b\\)')
_UNFINISHED_OSC = re.compile(r'(?:\x1b\][^\x07\x1b]*\x1b?|\x1b)\Z')
"""An OSC, or a lone ESC that may start one, still waiting for its terminator at the end of a write."""
MAX_HELD_OSC = 4096
"""An unterminated control longer than this is malformed and dropped, as the transcript does."""
TRANSCRIPT_KEYS = frozenset({'pageup', 'pagedown', 'mouse'})
"""Decoded keys that scroll or select the transcript rather than reach the widget pinned under it."""
WHEEL_ROWS = 3


def open_in_browser(url: str) -> None:
    """Open `url` without stalling input on a slow browser launch."""
    Thread(target=webbrowser.open, args=(url,), daemon=True).start()


class PromptSurface(io.StringIO):
    """Own the terminal: every write lands in the transcript, and frames show it.

    Frames are a termflow.live `ScreenBuffer`: the transcript view fills the rows above the editor's,
    and `render_diff` sends only changed cells. Nothing scrolls the terminal, so a resize, a theme
    change, or a closed menu repaints from the transcript. `restore` prints what the terminal's own
    scrollback has not seen yet.
    """

    def __init__(
        self,
        *,
        output: IO[str],
        size: Callable[[], tuple[int, int]],
        clock: Callable[[], float] = time.monotonic,
        transcript: TranscriptBuffer | None = None,
        open_url: Callable[[str], object] = open_in_browser,
    ) -> None:
        """Bind terminal IO and injectable geometry/time sources."""
        super().__init__()
        self.output = output
        self.size = size
        self.clock = clock
        self.open_url = open_url
        self.transcript = transcript if transcript is not None else TranscriptBuffer()
        self.view = TranscriptView(self.transcript)
        self._lock = RLock()
        self._rows: tuple[str, ...] = ()
        self._previous: ScreenBuffer | None = None
        self._live = False
        """Painted since opening, so writes repaint."""
        self._screen = False
        self._modes = False
        self._holds = 0
        self._dirty = False
        self._painted_at = -math.inf
        self._partial = False
        self._held = ''
        """The start of a control split across writes, kept until its terminator arrives."""
        self.selection = Selection()
        self._frame: ScreenBuffer | None = None
        """The cells last painted, which a selection copies from."""
        self._transcript_rows = 0
        """How many of the frame's top rows show the transcript, the only ones a selection covers."""

    def isatty(self) -> bool:
        """Preserve Rich and Termflow terminal detection."""
        return self.output.isatty()

    def resize_notice(self) -> None:
        """Repaint every cell next frame; signal handlers must not draw."""
        self._previous = None
        self.selection.clear()

    @contextmanager
    def held(self, *, leave_screen: bool = True) -> Generator[None]:
        """Stop painting while another widget owns the terminal; writes still reach the transcript.

        Full-screen widgets enter the alternate screen themselves and leave it for the main one,
        so the panel steps out first and enters again on its next frame. An inline widget that
        paints through this surface keeps it on screen with `leave_screen=False`. Holds nest.
        """
        with self._lock:
            self._holds += 1
            if leave_screen:
                self._leave()
        try:
            yield
        finally:
            with self._lock:
                self._holds -= 1

    def write(self, text: str) -> int:
        """Record output and repaint, forwarding palette controls to the terminal.

        A write of controls alone repaints on the next frame instead of now: a palette change
        arrives as several writes, and a frame painted between them would split the sequence.
        """
        with self._lock:
            data = self._held + text
            unfinished = _UNFINISHED_OSC.search(data)
            cut = unfinished.start() if unfinished else len(data)
            data, self._held = data[:cut], data[cut:]
            if len(self._held) > MAX_HELD_OSC:
                self._held = ''
            controls = _TERMINAL_OSC.findall(data)
            for control in controls:
                self.output.write(control)
            if controls:
                self.output.flush()
                # Every cell's colours changed, so the next frame repaints them all.
                self._previous = None
            self.transcript.write(text)
            if content := _TERMINAL_OSC.sub('', data):
                self._partial = not content.endswith('\n')
                self.changed()
            elif controls:
                self._dirty = True
        return len(text)

    def changed(self) -> None:
        """Note new transcript content, painting now unless the last frame was too recent."""
        with self._lock:
            self._dirty = True
            if self.clock() - self._painted_at >= FRAME_INTERVAL:
                self.refresh()

    def refresh(self) -> None:
        """Paint output that arrived since the last frame, unless another widget owns the screen."""
        with self._lock:
            if self._dirty and self._live and not self._holds:
                self._paint()

    def flush(self) -> None:
        """Flush without painting; the frame rate is the surface's own."""
        with self._lock:
            self.output.flush()

    async def drain(self) -> None:
        """Settle an incomplete transcript line before a turn/menu boundary."""
        if self._partial:
            self.write('\n')

    def markdown(self, *, render: Render, width: int) -> MarkdownBlock:
        """Start a Markdown part that renders again for a new width or theme."""
        with self._lock:
            return self.transcript.markdown(render=render, width=width, changed=self.changed)

    def clear(self, *, keep_current: bool = False) -> None:
        """Forget the transcript and repaint every cell next frame; see `TranscriptBuffer.clear`."""
        with self._lock:
            self.transcript.clear(keep_current=keep_current)
            self.view.follow()
            self._partial = self._partial and keep_current
            self._previous = None
            self.selection.clear()
            self._dirty = True

    def scroll(self, rows: int) -> None:
        """Scroll the transcript back (positive) or forward (negative)."""
        with self._lock:
            self.view.scroll(rows)
            self.selection.clear()
            if self._live and not self._holds:
                self._paint()

    def transcript_key(self, key: str, data: str = '') -> str | None:
        """Page, wheel, drag-select, or click a URL for one of `TRANSCRIPT_KEYS`; return the text a drag copied.

        Reporting the wheel stops most terminals from selecting text themselves, so a left-button
        drag highlights cells here and its release copies them to the clipboard. It also stops them
        opening links on Cmd/Ctrl+click, and mouse reports cannot say Cmd was held, so a click
        without a drag on a URL opens it here.
        """
        if key != 'mouse':
            self.scroll(self.page if key == 'pageup' else -self.page)
            return None
        report = mouse_report(data)
        if report is None:
            return None
        if report.button in (WHEEL_UP, WHEEL_DOWN):
            self.scroll(WHEEL_ROWS if report.button == WHEEL_UP else -WHEEL_ROWS)
            return None
        with self._lock:
            clicked = report.released and report.button == LEFT and self.selection.held and self.selection.head is None
            if self.selection.feed(report) and self._frame is not None:
                text = self.selection.text(self._frame, rows=self._transcript_rows)
                if text.strip():
                    copy_text(text, output=self.output)
                    return text
            elif clicked and self._frame is not None and (url := self._url_at(report.cell)):
                self.open_url(url)
            elif self._live and not self._holds:
                self._paint()
        return None

    def _url_at(self, cell: tuple[int, int]) -> str | None:
        assert self._frame is not None
        # The scrolled-away hint covers the bottom transcript row, so a URL there reads as cut off.
        rows = self._transcript_rows - (self.view.anchor is not None)
        return url_at(self._frame, cell, rows=rows, joins=self.view.joins())

    @property
    def page(self) -> int:
        """Rows one PageUp moves: the transcript's height, keeping a row of context."""
        _, height = self._geometry()
        return max(1, height - len(self._rows) - 1)

    def _geometry(self) -> tuple[int, int]:
        width, height = self.size()
        return max(1, width), max(2, height)

    def paint(self, rows: tuple[str, ...]) -> None:
        """Pin `rows` under the transcript and enable the editor's input modes."""
        with self._lock:
            width, height = self._geometry()
            self._rows = tuple(truncate(row, width) for row in rows[-(height - 2) :]) if height > 2 else ()
            self._live = True
            self._paint(modes=True)

    def _paint(self, *, modes: bool = False) -> None:
        width, height = self._geometry()
        parts: list[str] = []
        if not self._screen:
            parts.append(ENTER)
            self._screen = True
            self._previous = None
        if modes and not self._modes:
            parts.append(MODES_ON)
            self._modes = True
        rows = self._rows[-(height - 2) :] if height > 2 else ()
        bottom = height - len(rows)
        frame = ScreenBuffer(width, height)
        self.view.draw(frame.region(Rect(0, 0, width, bottom)), False)
        for index, row in enumerate(rows):
            frame.region(Rect(0, bottom + index, width, 1)).ansi(0, 0, row)
        self.selection.highlight(frame, rows=bottom, previous=self._frame)
        parts.append(render_diff(self._previous, frame))
        self._previous = self._frame = frame
        self._transcript_rows = bottom
        self._dirty = False
        self._painted_at = self.clock()
        if text := ''.join(parts):
            self.output.write('\x1b[?2026h' + text + '\x1b[?2026l')
            self.output.flush()

    def _leave(self) -> None:
        parts: list[str] = []
        if self._modes:
            parts.append(MODES_OFF)
            self._modes = False
        if self._screen:
            parts.append(LEAVE)
            self._screen = False
            self._previous = None
        if parts:
            self.output.write(''.join(parts))
            self.output.flush()

    def release(self) -> None:
        """Drop the editor rows and input modes before a menu, command, or shell takes over.

        The transcript stays on screen, and output the command prints still appears.
        """
        with self._lock:
            if self._partial:
                self.transcript.write('\n')
                self._partial = False
            if self.selection.clear():
                self._dirty = True
            if self._modes:
                self.output.write(MODES_OFF)
                self.output.flush()
                self._modes = False
            if self._rows:
                self._rows = ()
                self._dirty = True
            self.refresh()

    def restore(self) -> None:
        """Leave the alternate screen and print the session into the terminal's own scrollback.

        Not `close`: `io.IOBase` calls that when the object is collected.
        """
        with self._lock:
            self.release()
            self._leave()
            self._live = False
            width, _ = self._geometry()
            if printed := self.transcript.printed(width=width):
                self.output.write(printed.replace('\n', '\r\n') if self.output.isatty() else printed)
                self.output.flush()
