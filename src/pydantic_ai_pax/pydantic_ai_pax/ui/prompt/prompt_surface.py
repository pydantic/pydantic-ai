"""The live panel: transcript and pinned editor, inline until they fill the terminal, then on the alternate screen."""

import asyncio
import io
import math
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import RLock, Thread
from typing import IO

from termflow.live import Rect, ScreenBuffer, render_diff
from termflow.tui.layout import truncate

from pydantic_ai_pax.ui.browser import open_browser
from pydantic_ai_pax.ui.prompt.prompt_selection import LEFT, WHEEL_DOWN, WHEEL_UP, Selection, mouse_report, url_at
from pydantic_ai_pax.ui.prompt.prompt_transcript import MarkdownBlock, Render, TranscriptBuffer
from pydantic_ai_pax.ui.prompt.text_clipboard import copy_text
from pydantic_ai_pax.ui.prompt.transcript_view import SCROLLED_HINT, TranscriptView
from pydantic_ai_pax.ui.rendering import theme

ENTER = '\x1b[?1049h\x1b[?25l\x1b[?7l'
"""Alternate screen, hidden cursor, and no autowrap, which `render_diff` assumes."""
LEAVE = '\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l'
KEYS_ON = '\x1b[?2004h\x1b[>4;1m\x1b[>5u'
"""Bracketed paste, xterm modified keys, and Kitty disambiguation with alternate keys.

The inline panel leaves the mouse to the terminal, so its own selection and scrollback keep working.
"""
MODES_ON = '\x1b[?2004h\x1b[>4;1m\x1b[>5u\x1b[?1000h\x1b[?1002h\x1b[?1006h'
"""`KEYS_ON` plus SGR mouse buttons and drags.

Kitty keeps a flag stack per screen, so these are pushed after `ENTER` and popped before `LEAVE`.
"""
MODES_OFF = '\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[<u\x1b[>4;0m\x1b[?2004l'
INLINE_ON = '\x1b[?25l\x1b[?7l'
"""Hidden cursor and no autowrap for the inline panel, so each row takes exactly one terminal row."""
INLINE_OFF = '\x1b[0m\x1b[?7h\x1b[?25h'
FRAME_INTERVAL = 1 / 60
"""Writes repaint at most this often; the editor's refresh loop paints what is left."""
# Palette and other non-hyperlink OSC commands are meant for the terminal, not the transcript.
_TERMINAL_OSC = re.compile(r'\x1b\](?!8;)[^\x07\x1b]*(?:\x07|\x1b\\)')
_UNFINISHED_OSC = re.compile(r'(?:\x1b\][^\x07\x1b]*\x1b?|\x1b)\Z')
"""An OSC, or a lone ESC that may start one, still waiting for its terminator at the end of a write."""
MAX_HELD_OSC = 4096
"""An unterminated control longer than this is malformed and dropped, as the transcript does."""
TRANSCRIPT_KEYS = frozenset({'pageup', 'pagedown', 'mouse'})
"""Decoded keys that scroll the transcript or select painted text rather than act as widget input."""
WHEEL_ROWS = 3


def open_in_browser(url: str) -> None:
    """Open `url` without stalling input on a slow browser launch."""
    Thread(target=open_browser, args=(url,), daemon=True).start()


class PromptSurface(io.StringIO):
    """Own the terminal: every write lands in the transcript, and frames show it.

    The panel starts inline, below the output already in the terminal's scrollback, and grows
    with the transcript. Once it would fill the terminal, or the terminal narrows, widens or shrinks, it moves
    to the alternate screen for good; setting `full_screen` first starts there.
    There, frames are a termflow.live `ScreenBuffer`: the transcript view fills the rows above the editor's,
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
        self.full_screen = False
        """Paint on the alternate screen, from the next frame on; set it before the first to start there.

        The inline panel sets it once it would fill the terminal or the width changes, and it stays set.
        """
        self._inline_rows: int = 0
        """How many main-screen rows the inline panel occupies, ending at the cursor's row."""
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
        """How many of the frame's top rows show the transcript."""
        self._hint_on_transcript = False
        """Whether `SCROLLED_HINT` covers the last transcript row, as when no editor rule is pinned."""
        self._select_pinned = False
        """Whether a drag can also highlight and copy pinned rows, not only the transcript."""
        self._paint_scheduled = False
        self._paint_delayed = False
        """A paint waits for the frame interval to end; see `changed`."""
        self._painted_as: tuple[int, int, str] | None = None
        """The geometry and theme of the last frame; transcript rows wrap and colour by both."""

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
        so the panel steps out first (erasing its inline rows) and paints again on its next frame.
        An inline widget that paints through this surface keeps it on screen with `leave_screen=False`.
        Holds nest.
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
        """Note new transcript content, painting now, or once the frame interval ends if the last frame was too recent."""
        with self._lock:
            self._dirty = True
            wait = FRAME_INTERVAL - (self.clock() - self._painted_at)
            if wait <= 0:
                self.refresh()
            else:
                self._paint_later(wait)

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

    def markdown(
        self, *, render: Render, width: int, streaming: bool = False, expanded: Render | None = None
    ) -> MarkdownBlock:
        """Start a Markdown part that renders again for a new width or theme; see `MarkdownBlock` for `streaming` and `expanded`."""
        with self._lock:
            return self.transcript.markdown(
                render=render, width=width, changed=self.changed, streaming=streaming, expanded=expanded
            )

    def toggle_details(self) -> bool:
        """Expand or collapse the output of every finished command in the transcript, for Ctrl+O; returns whether they are expanded.

        The terminal's scrollback cannot change, so results already printed there move the panel to
        the alternate screen, which paints them from the transcript.
        """
        with self._lock:
            expanded = self.transcript.toggle_details()
            self.full_screen = self.full_screen or self.transcript.printed_details()
            self.selection.clear()
            self._dirty = True
        return expanded

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
            self._dirty = True
        self._paint_soon()

    def transcript_key(self, key: str, data: str = '') -> str | None:
        """Page, wheel, select, or click a URL for one of `TRANSCRIPT_KEYS`; return the text a selection copied.

        Reporting the wheel stops most terminals from selecting text themselves, so a left-button
        drag, double-click, or triple-click highlights cells here and its release copies them to the
        clipboard. It also stops them opening links on Cmd/Ctrl+click, and mouse reports cannot say
        Cmd was held, so a click without a drag on a URL opens it here; the second click of a
        double-click selects instead.
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
            if self.selection.feed(report, now=self.clock()) and self._frame is not None:
                text = self.selection.text(
                    self._frame,
                    rows=self._frame.height if self._select_pinned else self._transcript_rows,
                    joins=self._wraps(),
                )
                if text.strip():
                    copy_text(text, output=self.output)
                    return text
            elif clicked and self._frame is not None and (url := self._url_at(report.cell)):
                self.open_url(url)
            else:
                self._dirty = True
        self._paint_soon()
        return None

    def _url_at(self, cell: tuple[int, int]) -> str | None:
        assert self._frame is not None
        return url_at(self._frame, cell, rows=self._transcript_rows - self._hint_on_transcript, joins=self.view.joins())

    def _wraps(self) -> tuple[bool, ...]:
        """Which of the frame's transcript rows wrap on from the row above, for a selection.

        Pinned rows never do, and neither does a row `SCROLLED_HINT` covers.
        """
        return self.view.joins()[: self._transcript_rows - self._hint_on_transcript]

    def _paint_soon(self) -> None:
        """Paint once the event loop finishes the current callback, or now outside one.

        The input reader handles every pending report in one callback, and a trackpad sends wheel
        and drag reports in bursts. A frame costs milliseconds, so painting per report stalled input
        for a burst's worth of frames that were each replaced at once; this paints the last one.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.refresh()
            return
        if not self._paint_scheduled:
            self._paint_scheduled = True
            loop.call_soon(self._scheduled_paint)

    def _paint_later(self, delay: float) -> None:
        """Paint after `delay` seconds; outside the event loop, as from a worker thread, the editor's next tick does."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if not self._paint_delayed:
            self._paint_delayed = True
            loop.call_later(delay, self._delayed_paint)

    def _delayed_paint(self) -> None:
        self._paint_delayed = False
        self.refresh()

    def _scheduled_paint(self) -> None:
        self._paint_scheduled = False
        self.refresh()

    @property
    def page(self) -> int:
        """Rows one PageUp moves: the transcript's height, keeping a row of context."""
        _, height = self._geometry()
        return max(1, height - len(self._rows) - 1)

    def _geometry(self) -> tuple[int, int]:
        width, height = self.size()
        return max(1, width), max(2, height)

    def paint(self, rows: tuple[str, ...], *, select_pinned: bool = False) -> None:
        """Pin `rows` under the transcript and enable the editor's input modes.

        `select_pinned` lets a drag highlight and copy the pinned rows too. The default excludes
        editor and status rows from selection.

        The editor also repaints on a timer to pick up status changes, so a frame identical to the
        last one is skipped rather than rebuilt and diffed cell by cell.
        """
        with self._lock:
            width, height = self._geometry()
            if select_pinned != self._select_pinned:
                self.selection.clear()
                self._dirty = True
            self._select_pinned = select_pinned
            pinned = tuple(truncate(row, width) for row in rows[-(height - 2) :]) if height > 2 else ()
            unchanged = (
                pinned == self._rows
                and self._modes
                and not self._dirty
                and (self._previous is not None or self._inline_rows > 0)
                and self._painted_as == (width, height, theme.name())
            )
            self._rows = pinned
            self._live = True
            if not unchanged:
                self._paint(modes=True)

    def _paint(self, *, modes: bool = False) -> None:
        width, height = self._geometry()
        if not self.full_screen and self._paint_compact(modes=modes, width=width, height=height):
            return
        self.full_screen = True
        parts: list[str] = []
        if not self._screen:
            if self._modes:
                # The inline panel pushed its modes on the main screen; Kitty keeps a flag stack per screen.
                parts.append(MODES_OFF)
                self._modes, modes = False, True
            parts.append(self._erase_inline() + ENTER)
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
        # A URL the hint covers part of is not opened.
        self._hint_on_transcript = self.view.anchor is not None and _show_scrolled_hint(frame, top=bottom)
        self._transcript_rows = bottom
        self.selection.highlight(
            frame, rows=height if self._select_pinned else bottom, joins=self._wraps(), previous=self._frame
        )
        parts.append(render_diff(self._previous, frame))
        self._previous = self._frame = frame
        self._dirty = False
        self._painted_at = self.clock()
        self._painted_as = (width, height, theme.name())
        if text := ''.join(parts):
            self.output.write('\x1b[?2026h' + text + '\x1b[?2026l')
            self.output.flush()

    def _paint_compact(self, *, modes: bool, width: int, height: int) -> bool:
        """Paint the inline panel, or print a released one, unless it has to move to the alternate screen."""
        if self._inline_moved(width=width, height=height):
            return False
        if not self._rows:
            self._print_released(width=width, height=height)
            return True
        pinned = self._rows[-(height - 2) :] if height > 2 else ()
        if (shown := self._unprinted(width=width, limit=height - len(pinned))) is None:
            return False
        self._paint_inline([*shown, *pinned], modes=modes, width=width, height=height)
        return True

    def _inline_moved(self, *, width: int, height: int) -> bool:
        """Whether a resize may have moved the inline rows, so erasing them by count would guess.

        A narrower or wider terminal reflows them, and a shorter one can push some into scrollback.
        """
        painted = self._painted_as
        return self._inline_rows > 0 and painted is not None and (painted[0] != width or painted[1] > height)

    def _unprinted(self, *, width: int, limit: int) -> list[str] | None:
        """The transcript rows not yet in the terminal's scrollback, or `None` once there are `limit` of them."""
        shown: list[str] = []
        for item in self.transcript.unprinted():
            shown += self.transcript.rows(item, width=width)
            if len(shown) >= limit:
                return None
        return shown

    def _paint_inline(self, lines: list[str], *, modes: bool, width: int, height: int) -> None:
        """Paint `lines` over the inline panel's rows on the main screen, scrolling earlier output up as it grows."""
        self.view.follow()  # Every unprinted row shows, so there is nothing to scroll back to.
        parts: list[str] = []
        if modes and not self._modes:
            parts.append(KEYS_ON)
            self._modes = True
        parts += [INLINE_ON, self._erase_inline() or '\r', '\r\n'.join(f'{line}\x1b[0m' for line in lines)]
        self.output.write('\x1b[?2026h' + ''.join(parts) + '\x1b[?2026l')
        self.output.flush()
        self._inline_rows = len(lines)
        self._dirty = False
        self._painted_at = self.clock()
        self._painted_as = (width, height, theme.name())

    def _print_released(self, *, width: int, height: int) -> None:
        """Print the completed transcript into scrollback while nothing is pinned, and paint only an unfinished line.

        A command run meanwhile may write to the terminal itself, so no more rows stay than that line,
        which a later erase would count; a progress line such as a `!` command's dots still shows.
        """
        parts = [self._erase_inline() + INLINE_OFF] if self._inline_rows else []
        if printed := self.transcript.printed(width=width):
            parts.append(printed.replace('\n', '\r\n') if self.output.isatty() else printed)
        if self._partial:
            tail = self.transcript.rows(self.transcript.end, width=width)
            parts.append('\x1b[?2026h' + INLINE_ON + '\r\n'.join(f'{row}\x1b[0m' for row in tail) + '\x1b[?2026l')
            self._inline_rows = len(tail)
        if parts:
            self.output.write(''.join(parts))
            self.output.flush()
        self._dirty = False
        self._painted_at = self.clock()
        self._painted_as = (width, height, theme.name())

    def _erase_inline(self) -> str:
        """Return to the inline panel's first row and clear it and everything below, or nothing without one."""
        rows, self._inline_rows = self._inline_rows, 0
        if not rows:
            return ''
        return '\x1b[0m\r' + (f'\x1b[{rows - 1}A' if rows > 1 else '') + '\x1b[J'

    def _leave(self) -> None:
        parts: list[str] = []
        if self._modes:
            parts.append(MODES_OFF)
            self._modes = False
        if self._screen:
            parts.append(LEAVE)
            self._screen = False
            self._previous = None
        elif self._inline_rows:
            parts.append(self._erase_inline() + INLINE_OFF)
        if parts:
            self.output.write(''.join(parts))
            self.output.flush()

    def release(self, *, turn_running: bool = False) -> None:
        """Drop the editor rows and input modes before a menu, command, or shell takes over.

        The transcript stays on screen, and output the command prints still appears. An inline panel
        prints it into scrollback instead, keeping no rows: what takes over may write to the terminal
        itself, below rows the panel would later erase by count. While `turn_running`, the turn may
        still extend what it streamed, so an inline panel moves to the alternate screen instead.
        """
        with self._lock:
            if turn_running and not self.full_screen:
                self.full_screen = self._dirty = True
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
        """Leave the alternate screen or erase the inline panel, then print the session into the terminal's scrollback.

        Not `close`: `io.IOBase` calls that when the object is collected.
        """
        with self._lock:
            self._live = False
            self._leave()
            self.release()
            width, _ = self._geometry()
            if printed := self.transcript.printed(width=width):
                self.output.write(printed.replace('\n', '\r\n') if self.output.isatty() else printed)
                self.output.flush()


def _show_scrolled_hint(frame: ScreenBuffer, *, top: int) -> bool:
    """Paint `SCROLLED_HINT` at the right end of the editor's bottom rule; return whether it went on the transcript.

    That rule is the last pinned row made only of `─`, so the hint covers no text and stays put
    while the working title above the draft changes. Pinned rows without a rule, such as line input
    on the alternate screen, get it on the last transcript row.
    """
    rules = [y for y in range(top, frame.height) if frame.row_text(y) == '─' * frame.width]
    y = rules[-1] if rules else top - 1
    hint = f'{theme.sgr(theme.MUTED)}\x1b[7m{SCROLLED_HINT[: frame.width]}\x1b[0m'
    frame.region(Rect(0, y, frame.width, 1)).ansi(max(0, frame.width - len(SCROLLED_HINT)), 0, hint)
    return y < top
