"""Bounded, styled transcript that the live panel paints, scrolls, and prints to scrollback on exit."""

import io
import re
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Protocol, cast

from rich.ansi import AnsiDecoder
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from termflow.ansi.utils import ANSI_ESCAPE_RE, visible_length

from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.recolor import recolor

# Rich does not recognize palette OSC commands and renders their payload as text.
_OSC = re.compile(r'\x1b\]([^\x07\x1b]*)(?:\x07|\x1b\\)')
_CONSOLE = Console(file=io.StringIO(), force_terminal=True, color_system='truecolor')
"""Only decodes recorded styling into segments; it never writes to a terminal."""


def replay_osc(match: re.Match[str]) -> str:
    """Keep only hyperlink metadata, normalizing BEL for Rich's ANSI decoder."""
    payload = match[1]
    if payload.startswith('8;'):
        _, separator, url = payload[2:].partition(';')
        if separator and (not url or url.isprintable()):
            return f'\x1b]8;;{url}\x1b\\'
    return ''


@dataclass(frozen=True, kw_only=True)
class TranscriptFrame:
    """Visible rows and the styling needed by the next streaming write."""

    rows: tuple[str, ...]
    continuation_style: str


def style_prefix(style: Style) -> str:
    """Render an SGR prefix without replaying hyperlinks or visible text."""
    return render_ansi(text=' ', style=style.update_link(None)).split(' ', 1)[0]


def render_ansi(*, text: str, style: Style) -> str:
    """Copy style attributes without Rich's color-system-specific ANSI cache."""
    fresh = Style(
        color=style.color,
        bgcolor=style.bgcolor,
        bold=style.bold,
        dim=style.dim,
        italic=style.italic,
        underline=style.underline,
        blink=style.blink,
        blink2=style.blink2,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        underline2=style.underline2,
        frame=style.frame,
        encircle=style.encircle,
        overline=style.overline,
        link=style.link,
    )
    return fresh.render(text, color_system=ColorSystem.TRUECOLOR)


def _clean(line: Text) -> Text:
    """A copy safe to replay: control characters shown as `?`, tabs expanded."""
    text = line.copy()
    text.plain = ''.join(char if char.isprintable() or char == '\t' else '?' for char in text.plain)
    text.expand_tabs(8)
    return text


def _encode(text: Text) -> str:
    """Only SGR and hyperlinks are replayed, never other controls."""
    return ''.join(
        render_ansi(text=segment.text, style=segment.style) if segment.style else segment.text
        for segment in text.render(_CONSOLE)
    )


def wrap(line: Text, *, width: int) -> tuple[str, ...]:
    """Split a styled line into rows of at most `width` cells, counting wide characters as two."""
    text = _clean(line)
    offsets: list[int] = []
    cells = 0
    for index, char in enumerate(text.plain):
        size = visible_length(char)
        if cells and cells + size > width:
            offsets.append(index)
            cells = 0
        cells += size
    rows: list[str] = []
    for piece in text.divide(offsets):
        piece.truncate(width, overflow='crop')
        rows.append(_encode(piece))
    return tuple(rows)


class TranscriptOutput(io.StringIO):
    """Capture non-editor console output while forwarding it immediately."""

    def __init__(self, *, output: IO[str], transcript: 'TranscriptBuffer') -> None:
        """Retain the original destination without taking ownership of it."""
        super().__init__()
        self.output, self.transcript = output, transcript

    def write(self, text: str) -> int:
        """Record and forward a console chunk once; it is already in the terminal's own scrollback."""
        self.transcript.write(text)
        self.transcript.mark_printed()
        return self.output.write(text)

    def flush(self) -> None:
        """Leave console flushing behavior unchanged."""
        self.output.flush()

    def isatty(self) -> bool:
        """Preserve terminal detection during startup and plugin hooks."""
        return self.output.isatty()


def incomplete_escape_start(text: str) -> int:
    """Find trailing control data outside complete tokens, including OSC's ST."""
    end = 0
    for escape in ANSI_ESCAPE_RE.finditer(text):
        end = escape.end()
    return text.find('\x1b', end)


class TranscriptDecoder(AnsiDecoder):
    """Keep OSC 8 state independent of SGR resets, as terminals do."""

    def decode_line(self, line: str) -> Text:
        """Decode styles without letting a colour reset close an active hyperlink."""
        text = Text()
        for index, chunk in enumerate(re.split(r'(\x1b\[[0-9;]*m)', line.rsplit('\r', 1)[-1])):
            link = self.style.link
            text.append_text(super().decode_line(chunk))
            if index % 2:
                self.style = self.style.update_link(link)
        return text


class _Line:
    """One line and the theme that painted it, with rows cached for the last width and theme.

    `theme_name` is `None` for branding, which no theme recolours.
    """

    def __init__(self, text: Text, *, theme_name: str | None) -> None:
        self.text = text
        self.theme_name = theme_name
        self._rows: tuple[tuple[int, str], tuple[str, ...]] = ((0, ''), ())

    @classmethod
    def rebind(cls, line: '_Line') -> '_Line':
        line.__class__ = cls
        # Lines retained before themes were recorded were painted in the theme of the reload.
        vars(line).setdefault('theme_name', theme.name())
        return line

    def _themed(self) -> Text:
        """The line in the current theme, translated role by role from the theme that painted it."""
        current = theme.name()
        if self.theme_name is None or current == self.theme_name:
            return self.text
        return recolor(self.text, source=self.theme_name, target=current)

    def rows(self, *, width: int) -> tuple[str, ...]:
        key = (width, theme.name())
        if self._rows[0] != key:
            self._rows = (key, wrap(self._themed(), width=width))
        return self._rows[1]

    def continued(self, *, width: int) -> tuple[bool, ...]:
        """Per row at `width`, whether it wraps on from the row above rather than starting the line."""
        return tuple(index > 0 for index in range(len(self.rows(width=width))))

    def printed(self, *, width: int) -> str:
        return _encode(_clean(self._themed())) + '\n'


class _Lines:
    """An unbounded ANSI stream decoded into styled lines, keeping an unfinished one."""

    def __init__(self, *, max_chars: int = 4_000_000, max_lines: int = 10_000) -> None:
        self.max_chars, self.max_lines = max_chars, max_lines
        self.chars = 0
        self.revision = 0
        self.lines: list[_Line] = []
        self.pending = ''
        self.pending_theme = theme.painted_in()
        """The theme the unfinished line started in, which its completion keeps."""
        self._decoder = TranscriptDecoder()

    @classmethod
    def rebind(cls, stream: '_Lines') -> '_Lines':
        stream.__class__ = cls
        stream._decoder.__class__ = TranscriptDecoder
        vars(stream).setdefault('pending_theme', theme.name())
        for line in stream.lines:
            _Line.rebind(line)
        return stream

    def write(self, text: str) -> None:
        self.revision += 1
        started = self.pending_theme if self.pending else theme.painted_in()
        self.pending = _OSC.sub(replay_osc, self.pending + text)
        *done, self.pending = self.pending.split('\n')
        for line in done:
            decoded = self._decoder.decode_line(line.removesuffix('\r'))[-self.max_chars :]
            self.lines.append(_Line(decoded, theme_name=started))
            self.chars += len(decoded)
            started = theme.painted_in()
        self.pending_theme = started
        self.pending = self.pending[-self.max_chars :]
        while len(self.lines) > self.max_lines or self.chars + len(self.pending) > self.max_chars:
            self.chars -= len(self.lines.pop(0).text)

    def tail(self) -> Text:
        decoder = TranscriptDecoder()
        decoder.style = self._decoder.style
        complete = self.pending
        escape = incomplete_escape_start(complete)
        return decoder.decode_line(complete if escape < 0 else complete[:escape])

    def all(self) -> list[_Line]:
        return [*self.lines, _Line(self.tail(), theme_name=self.pending_theme)] if self.pending else self.lines


class Render(Protocol):
    """Render a whole Markdown part to ANSI at `width`, in the current theme."""

    def __call__(self, *, source: str, width: int) -> str: ...


class MarkdownBlock(io.StringIO):
    """One assistant text or thinking part, kept as Markdown so a new width or theme renders it again.

    The stream writes its incrementally rendered ANSI here, which is shown while the width and theme
    still match. Otherwise the whole source renders again, once per width, theme, and length.
    """

    def __init__(
        self, *, render: Render, width: int, changed: Callable[[], None], max_chars: int, max_lines: int
    ) -> None:
        """Start empty; `changed` asks the panel for a frame."""
        super().__init__()
        self.source: str | None = ''
        self._render = render
        self._key = (width, theme.name())
        self._stream = _Lines(max_chars=max_chars, max_lines=max_lines)
        self._changed = changed
        self._rows: tuple[object, tuple[str, ...]] = ((), ())
        self._continued: tuple[bool, ...] = ()
        """Per cached row, whether it wraps on from the row above."""

    @classmethod
    def rebind(cls, block: 'MarkdownBlock') -> 'MarkdownBlock':
        block.__class__ = cls
        block._stream = _Lines.rebind(block._stream)
        # Blocks from before a reload have no wrap flags, so render their rows again.
        block._rows, block._continued = ((), ()), ()
        return block

    def extend(self, markdown: str) -> None:
        """Record source as it arrives, ahead of the smoothed rendering."""
        if self.source is not None:
            self.source += markdown
            if self.chars > self._stream.max_chars:
                self.freeze()
            self._changed()

    @property
    def chars(self) -> int:
        """Retained source plus visible output, for the transcript memory limit."""
        return len(self.source or '') + self._stream.chars + len(self._stream.pending)

    def write(self, text: str) -> int:
        """Receive the stream's rendered ANSI."""
        self._stream.write(text)
        if self.chars > self._stream.max_chars:
            self.freeze()
        self._changed()
        return len(text)

    def flush(self) -> None:
        """Painting is the panel's job."""

    def freeze(self) -> None:
        """Keep only what was shown, as an aborted stream never rendered the rest of its source."""
        self.source = None
        self._changed()

    def _lines(self, *, width: int) -> list[_Line]:
        key = (width, theme.name())
        if self.source is None or key == self._key:
            return self._stream.all()
        rendered = _Lines(max_chars=self._stream.max_chars, max_lines=self._stream.max_lines)
        rendered.write(self._render(source=self.source, width=width))
        return rendered.all()

    def rows(self, *, width: int) -> tuple[str, ...]:
        """Rows at `width`; the cache follows theme and source, not just width."""
        source = None if self.source is None else len(self.source)
        key = (width, theme.name(), source, self._stream.revision)
        if self._rows[0] != key:
            lines = self._lines(width=width)
            self._rows = (key, tuple(row for line in lines for row in line.rows(width=width)))
            self._continued = tuple(flag for line in lines for flag in line.continued(width=width))
        return self._rows[1]

    def continued(self, *, width: int) -> tuple[bool, ...]:
        """Per row at `width`, whether it wraps on from the row above rather than starting a line."""
        self.rows(width=width)
        return self._continued

    def printed(self, *, width: int) -> str:
        return ''.join(line.printed(width=width) for line in self._lines(width=width))


_Item = _Line | MarkdownBlock


class TranscriptBuffer:
    """Retain recent output, never editor paint or terminal-control transactions.

    Items have stable ids, so the panel's scroll position survives new output. `clear` forgets
    every item, so neither the panel nor `printed` shows output from before it.
    """

    def __init__(self, *, max_lines: int = 10_000, max_chars: int = 4_000_000) -> None:
        """Bound both completed lines and an unterminated streaming line."""
        if max_lines < 1 or max_chars < 1:
            raise ValueError('Transcript limits must be positive.')
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._items: deque[_Item] = deque()
        self._first = 0
        """The id of `_items[0]`; ids grow by one per item and are never reused."""
        self._printed = 0
        """Items before this id are already in the terminal's own scrollback."""
        self._chars = 0
        self._pending = ''
        self._pending_theme = theme.painted_in()
        """The theme the unfinished line started in, which its completion keeps."""
        self._discard_until_newline = False
        self._decoder = TranscriptDecoder()

    @classmethod
    def rebind(cls, transcript: 'TranscriptBuffer') -> 'TranscriptBuffer':
        """Upgrade retained state in place when reload replaces its class definitions.

        The suspended `chat` coroutine and capture wrappers still reference this object.
        Pre-live transcripts stored styled Text lines, all already emitted to native
        scrollback. Rebinding live transcripts also refreshes nested class identities,
        so eviction does not mistake an old `_Line` for a Markdown block.
        """
        if '_items' not in vars(transcript):
            retained = cls(max_lines=transcript.max_lines, max_chars=transcript.max_chars)
            for line in cast(deque[Text], vars(transcript)['_lines']):
                retained._append(_Line(line, theme_name=theme.name()))
            retained._pending = transcript._pending
            retained._discard_until_newline = transcript._discard_until_newline
            retained._decoder.style = transcript._decoder.style
            retained.mark_printed()
            vars(transcript).clear()
            vars(transcript).update(vars(retained))
        transcript.__class__ = cls
        transcript._decoder.__class__ = TranscriptDecoder
        vars(transcript).setdefault('_pending_theme', theme.name())
        for item in transcript._items:
            # StringIO is a stable dependency type across CLAI reloads.
            if isinstance(item, io.StringIO):
                MarkdownBlock.rebind(item)
            else:
                _Line.rebind(item)
        return transcript

    @property
    def end(self) -> int:
        """The id of the unfinished line, after every completed item."""
        return self._first + len(self._items)

    def _append(self, item: _Item) -> None:
        self._items.append(item)
        self._chars += len(item.text) if isinstance(item, _Line) else item.chars
        self._prune()

    def _prune(self) -> None:
        while len(self._items) > self.max_lines or self._chars > self.max_chars:
            dropped = self._items.popleft()
            self._first += 1
            self._chars -= len(dropped.text) if isinstance(dropped, _Line) else dropped.chars

    def write(self, text: str) -> None:
        """Decode completed ANSI lines, retaining partial sequences between writes."""
        if self._discard_until_newline:
            _, separator, text = text.partition('\n')
            if not separator:
                return
            text = '\n' + text
            self._discard_until_newline = False
        started = self._pending_theme if self._pending else theme.painted_in()
        self._pending = _OSC.sub(replay_osc, self._pending + text)
        lines = self._pending.split('\n')
        self._pending = lines.pop()
        for line in lines:
            # CRLF is a line ending, not a progress-line overwrite.
            decoded = self._decoder.decode_line(line.removesuffix('\r'))
            self._append(_Line(decoded[-self.max_chars :], theme_name=started))
            started = theme.painted_in()
        self._pending_theme = started
        if len(self._pending) > self.max_chars:
            cutoff = len(self._pending) - self.max_chars
            for escape in ANSI_ESCAPE_RE.finditer(self._pending):
                if escape.start() < cutoff < escape.end():
                    cutoff = escape.end()
                    break
            # Keep an unfinished escape intact until the next write completes it.
            start = incomplete_escape_start(self._pending)
            if 0 <= start < cutoff:
                if len(self._pending) - start > 4096:
                    # A malformed unclosed control must not defeat the replay
                    # memory bound. Keep the visible prefix, omit through EOL.
                    self._pending = self._pending[:start]
                    self._discard_until_newline = True
                    cutoff = max(0, len(self._pending) - self.max_chars)
                else:
                    cutoff = start
            prefix, self._pending = self._pending[:cutoff], self._pending[cutoff:]
            self._decoder.decode_line(prefix)

    def markdown(self, *, render: Render, width: int, changed: Callable[[], None]) -> MarkdownBlock:
        """Start a Markdown part on its own row, after any unfinished line."""
        if self._pending:
            self.write('\n')
        item = self.end
        size = 0

        def updated() -> None:
            nonlocal size
            if item >= self._first:
                self._chars += block.chars - size
                size = block.chars
                self._prune()
            changed()

        block = MarkdownBlock(
            render=render, width=width, changed=updated, max_chars=self.max_chars, max_lines=self.max_lines
        )
        self._append(block)
        return block

    def clear(self, *, keep_current: bool = False) -> None:
        """Forget all retained output, as a fresh terminal has none; ids keep growing.

        `keep_current` keeps what a running turn is still writing: the unfinished line, or else a
        trailing Markdown part, so the rest of a streaming response still shows.
        """
        current = self._items[-1] if keep_current and self._items and not self._pending else None
        kept = current if isinstance(current, MarkdownBlock) else None
        self._first = self.end - (kept is not None)
        self._items.clear()
        self._chars = 0
        if kept is not None:
            self._append(kept)
        if not keep_current:
            self._pending = ''
            self._discard_until_newline = False
            self._decoder = TranscriptDecoder()

    def mark_printed(self) -> None:
        """Everything completed so far reached the terminal directly, so `printed` skips it."""
        self._printed = self.end

    @contextmanager
    def capture(self, console: Console) -> Generator[None]:
        """Record startup/lifecycle output outside the live editor without duplication."""
        original = console.file
        console.file = TranscriptOutput(output=original, transcript=self)
        try:
            yield
        finally:
            console.file = original

    def ids(self) -> range:
        """Ids the panel may show, ending with the unfinished line."""
        return range(self._first, self.end + 1)

    def rows(self, item: int, *, width: int) -> tuple[str, ...]:
        """One item's rows; the unfinished line always has one, the writer's position."""
        if item == self.end:
            return _Line(self._tail(), theme_name=self._pending_theme).rows(width=width)
        return self._items[item - self._first].rows(width=width)

    def continued(self, item: int, *, width: int) -> tuple[bool, ...]:
        """Per row of one item, whether it wraps on from the row above rather than starting a line."""
        if item == self.end:
            return _Line(self._tail(), theme_name=self._pending_theme).continued(width=width)
        return self._items[item - self._first].continued(width=width)

    def _tail(self) -> Text:
        decoder = TranscriptDecoder()
        decoder.style = self._decoder.style
        complete = self._pending
        escape = incomplete_escape_start(complete)
        return decoder.decode_line(complete if escape < 0 else complete[:escape])

    def printed(self, *, width: int) -> str:
        """Completed output not yet in the terminal's scrollback, unwrapped, with links; marks it printed."""
        start = max(self._first, self._printed)
        text = ''.join(item.printed(width=width) for item in list(self._items)[start - self._first :])
        self.mark_printed()
        return text

    def frame(self, *, width: int, height: int) -> TranscriptFrame:
        """The last `height` rows, wrapped without performing any terminal IO."""
        rows: deque[str] = deque(maxlen=max(1, height))
        for item in self.ids():
            rows.extend(self.rows(item, width=width))
        decoder = TranscriptDecoder()
        decoder.style = self._decoder.style
        complete = self._pending
        escape = incomplete_escape_start(complete)
        decoder.decode_line(complete if escape < 0 else complete[:escape])
        return TranscriptFrame(rows=tuple(rows), continuation_style=style_prefix(decoder.style))
