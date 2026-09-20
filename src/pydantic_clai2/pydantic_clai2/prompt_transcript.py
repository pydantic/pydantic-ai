"""Bounded, styled transcript tail for repainting the viewport after resize."""

import io
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

from rich.ansi import AnsiDecoder
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from termflow.ansi.utils import ANSI_ESCAPE_RE, visible_length  # pyright: ignore[reportMissingTypeStubs]


@dataclass(frozen=True, kw_only=True)
class TranscriptFrame:
    """Visible rows and the styling needed by the next streaming write."""

    rows: tuple[str, ...]
    continuation_style: str


def style_prefix(style: Style) -> str:
    """Render an SGR prefix without replaying hyperlinks or visible text."""
    return render_ansi(text=' ', style=style).split(' ', 1)[0]


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
    )
    return fresh.render(text, color_system=ColorSystem.TRUECOLOR)


class TranscriptOutput(io.StringIO):
    """Capture non-editor console output while forwarding it immediately."""

    def __init__(self, *, output: IO[str], transcript: 'TranscriptBuffer') -> None:
        """Retain the original destination without taking ownership of it."""
        super().__init__()
        self.output, self.transcript = output, transcript

    def write(self, text: str) -> int:
        """Record and forward a console chunk once."""
        self.transcript.write(text)
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


class TranscriptBuffer:
    """Retain recent output, never editor paint or terminal-control transactions."""

    def __init__(self, *, max_lines: int = 2000, max_chars: int = 1_000_000) -> None:
        """Bound both completed lines and an unterminated streaming line."""
        if max_lines < 1 or max_chars < 1:
            raise ValueError('Transcript limits must be positive.')
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines: deque[Text] = deque()
        self._chars = 0
        self._pending = ''
        self._discard_until_newline = False
        self._decoder = AnsiDecoder()
        self._console = Console(file=io.StringIO(), force_terminal=True, color_system='truecolor')

    def write(self, text: str) -> None:
        """Decode completed ANSI lines, retaining partial sequences between writes."""
        if self._discard_until_newline:
            _, separator, text = text.partition('\n')
            if not separator:
                return
            text = '\n' + text
            self._discard_until_newline = False
        self._pending += text
        lines = self._pending.split('\n')
        self._pending = lines.pop()
        for line in lines:
            # CRLF is a line ending, not a progress-line overwrite.
            decoded = self._decoder.decode_line(line.removesuffix('\r'))
            decoded = decoded[-self.max_chars :]
            self._lines.append(decoded)
            self._chars += len(decoded)
            while len(self._lines) > self.max_lines or self._chars > self.max_chars:
                self._chars -= len(self._lines.popleft())
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

    @contextmanager
    def capture(self, console: Console) -> Generator[None]:
        """Record startup/lifecycle output outside the live editor without duplication."""
        original = console.file
        console.file = TranscriptOutput(output=original, transcript=self)
        try:
            yield
        finally:
            console.file = original

    def frame(self, *, width: int, height: int) -> TranscriptFrame:
        """Rewrap recent styled text without performing any terminal IO."""
        decoder = AnsiDecoder()
        decoder.style = self._decoder.style
        complete = self._pending
        escape = incomplete_escape_start(complete)
        if escape >= 0:
            complete = complete[:escape]
        pending = decoder.decode_line(complete)
        rows: deque[str] = deque(maxlen=max(1, height))
        for line in (*self._lines, pending):
            text = line.copy()
            text.plain = ''.join(char if char.isprintable() or char == '\t' else '?' for char in text.plain)
            text.expand_tabs(8)
            offsets: list[int] = []
            cells = 0
            for index, char in enumerate(text.plain):
                size = visible_length(char)
                if cells and cells + size > width:
                    offsets.append(index)
                    cells = 0
                cells += size
            for piece in text.divide(offsets):
                piece.truncate(width, overflow='crop')
                # Rich only decodes recorded ANSI styling here. It owns neither
                # the editor nor a live renderer. Non-SGR controls are not replayed.
                rows.append(
                    ''.join(
                        render_ansi(text=segment.text, style=segment.style) if segment.style else segment.text
                        for segment in piece.render(self._console)
                    )
                )
        return TranscriptFrame(rows=tuple(rows), continuation_style=style_prefix(decoder.style))
