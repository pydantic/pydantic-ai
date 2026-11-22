"""A bare launch paints the last launch's first frame before its imports, and the panel takes those rows over."""

import io
import os
import struct
import sys
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from pydantic_ai_pax.ui import first_frame
from pydantic_ai_pax.ui.first_frame import FirstFrame
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from pydantic_ai_pax.ui.prompt.prompt_transcript import TranscriptBuffer, TranscriptOutput
from tests.pax.surface_terminal import SurfaceTerminal

if sys.platform != 'win32':
    import fcntl
    import termios
    from tty import setraw

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='needs a POSIX pseudo-terminal')

ROWS = ('', 'RULE', '> ', 'RULE', 'FOOTER')


class TerminalErrors(io.StringIO):
    """Standard error shown in the terminal, as it is unless redirected."""

    def isatty(self) -> bool:
        return True


class Tty:
    """A pseudo-terminal standing in for the one `pax` runs in: `sys.stdin` and `sys.stdout` are its other end."""

    def __init__(self, *, width: int, height: int) -> None:
        self.screen, self.terminal = os.openpty()
        self.resize(width=width, height=height)

    def attach(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the test in this terminal; pytest swaps the standard streams back in after fixtures run."""
        monkeypatch.setattr(sys, 'stdin', open(self.terminal, encoding='utf-8', closefd=False))
        monkeypatch.setattr(sys, 'stdout', open(self.terminal, 'w', encoding='utf-8', closefd=False))
        monkeypatch.setattr(sys, 'stderr', TerminalErrors())
        monkeypatch.delenv('TERM', raising=False)

    def resize(self, *, width: int, height: int) -> None:
        fcntl.ioctl(self.terminal, termios.TIOCSWINSZ, struct.pack('HHHH', height, width, 0, 0))

    def shown(self) -> str:
        """What the program wrote since the last call, with the line discipline's added carriage returns removed."""
        os.set_blocking(self.screen, False)
        chunks: list[bytes] = []
        with suppress(BlockingIOError):
            while True:
                chunks.append(os.read(self.screen, 65536))
        return b''.join(chunks).decode().replace('\r\r\n', '\r\n')

    def typed(self) -> bytes:
        """What the editor reads, in the raw mode it sets."""
        modes = termios.tcgetattr(self.terminal)
        setraw(self.terminal, termios.TCSANOW)
        os.set_blocking(self.terminal, False)
        try:
            return os.read(self.terminal, 1024)
        finally:
            termios.tcsetattr(self.terminal, termios.TCSANOW, modes)

    def close(self) -> None:
        os.close(self.screen)
        os.close(self.terminal)


@pytest.fixture
def tty() -> Iterator[Tty]:
    tty = Tty(width=40, height=10)
    yield tty
    tty.close()


def launch(
    terminal: SurfaceTerminal, *, adopted: int = 0, size: tuple[int, int] = (40, 10), draft: str = ''
) -> PromptSurface:
    """The panel of one launch: the banner above the editor, painted inline."""
    surface = PromptSurface(output=terminal, size=lambda: (terminal.width, terminal.height))
    surface.full_screen = False
    surface.adopt(adopted, size=size)
    surface.write('banner\n')
    surface.paint(tuple(row.replace('> ', f'> {draft}') for row in ROWS))
    return surface


def test_the_next_launch_paints_the_first_frame_where_the_panel_will_and_keeps_typed_keys(
    tty: Tty, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tty.attach(monkeypatch)
    path = tmp_path / 'first-frame'
    first, before = FirstFrame(), SurfaceTerminal(width=40, height=10)
    first.paint(path)
    assert not first.holding and tty.shown() == '', 'nothing saved yet'
    before.write('$ pax\r\n')
    first.save(launch(before).startup_frame())
    first.save(['ignored'])

    second, after = FirstFrame(), SurfaceTerminal(width=40, height=10)
    after.write('$ pax\r\n')
    second.paint(path)
    shown = tty.shown()
    assert '\x1b[?2004h' in shown, 'a paste waits as one paste, not as lines that each press Enter'
    after.write(shown)
    assert after.lines() == before.lines()
    os.write(tty.screen, b'hi\r')
    assert tty.shown() == '', 'keys typed during imports are not echoed among the rows'
    print('Traceback: import-time warning', file=sys.stderr)

    rows, size, errors = second.adopt()
    assert sys.stderr is not None and errors == 'Traceback: import-time warning\n'
    second.restore_input()
    second.restore_input()
    assert tty.typed() == b'hi\r', 'the keys wait for the editor, Enter as raw mode reads it'
    os.write(tty.screen, b'x')
    assert tty.shown() == 'x', 'echo is back for the editor to save and turn off itself'
    launch(after, adopted=rows, size=size, draft='hi')
    assert after.lines() == [('> hi' if line == '>' else line) for line in before.lines()], 'no row painted twice'


def test_a_resize_before_the_panel_opens_moves_it_to_full_screen(
    tty: Tty, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal reflowed the early rows, so erasing them by count would leave some behind."""
    tty.attach(monkeypatch)
    path = tmp_path / 'first-frame'
    saved = FirstFrame()
    saved.paint(path)
    saved.save(['banner', *ROWS])
    frame, terminal = FirstFrame(), SurfaceTerminal(width=40, height=10)
    frame.paint(path)
    terminal.write(tty.shown())
    tty.resize(width=30, height=10)
    terminal.resize(width=30, height=10)
    rows, size, _ = frame.adopt()
    launch(terminal, adopted=rows, size=size)
    assert terminal.alternate


def test_a_frame_saved_for_another_terminal_is_not_painted(
    tty: Tty, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tty.attach(monkeypatch)
    path = tmp_path / 'first-frame'
    saved = FirstFrame()
    saved.paint(path)
    saved.save(['banner', *ROWS])
    tty.resize(width=30, height=10)
    frame = FirstFrame()
    frame.paint(path)
    assert not frame.holding and tty.shown() == ''
    frame.save(None)
    assert not path.exists(), 'a panel on the alternate screen leaves no frame to paint'


def test_a_frame_that_fills_the_terminal_is_not_saved() -> None:
    """Holding all of it unprinted, the next launch's panel would move to full screen, then forget the frame."""
    terminal = SurfaceTerminal(width=40, height=10)
    surface = PromptSurface(output=terminal, size=lambda: (terminal.width, terminal.height))
    surface.full_screen = False
    for line in range(4):
        terminal.write(f'printed {line}\r\n')
        surface.transcript.write(f'printed {line}\n')
    surface.transcript.mark_printed()
    surface.paint(ROWS)
    assert not terminal.alternate
    assert surface.startup_frame() is None


def test_closing_before_the_panel_opens_erases_the_rows_and_shows_what_went_to_stderr(
    tty: Tty, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tty.attach(monkeypatch)
    path = tmp_path / 'first-frame'
    saved = FirstFrame()
    saved.paint(path)
    saved.save(['banner', *ROWS])
    frame, terminal = FirstFrame(), SurfaceTerminal(width=40, height=10)
    terminal.write('$ pax\r\n')
    stderr = sys.stderr
    assert isinstance(stderr, TerminalErrors)
    frame.paint(path)
    print('Traceback: ImportError', file=sys.stderr)
    assert stderr.getvalue() == '', 'held while the rows show'
    frame.close()
    shown = tty.shown()
    assert shown.endswith('\x1b[?2004l'), 'the shell gets pastes back as typed text'
    terminal.write(shown)
    assert terminal.lines() == ['$ pax', *[''] * 9]
    assert stderr.getvalue() == 'Traceback: ImportError\n'
    frame.close()


def test_standard_error_redirected_away_from_the_terminal_is_not_held(
    tty: Tty, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tty.attach(monkeypatch)
    path = tmp_path / 'first-frame'
    saved = FirstFrame()
    saved.paint(path)
    saved.save(['banner', *ROWS])
    errors = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', errors)
    frame = FirstFrame()
    frame.paint(path)
    assert frame.holding
    print('Traceback: ImportError', file=sys.stderr)
    assert errors.getvalue() == 'Traceback: ImportError\n', 'it is in the file at once, whatever happens next'
    assert frame.adopt()[2] == ''


def test_nothing_is_painted_or_saved_outside_a_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    frame = FirstFrame()
    frame.paint(tmp_path / 'first-frame')
    frame.save(['banner'])
    assert not (tmp_path / 'first-frame').exists()


def test_console_output_waits_for_the_panel_while_the_early_rows_show(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal, transcript = io.StringIO(), TranscriptBuffer()
    output = TranscriptOutput(output=terminal, transcript=transcript)
    monkeypatch.setattr(first_frame.current, 'rows', 5)
    output.write('banner\n')
    assert terminal.getvalue() == '', 'printed among the early rows, it would be painted over'
    monkeypatch.setattr(first_frame.current, 'rows', 0)
    output.write('later\n')
    assert terminal.getvalue() == 'later\n'
    assert [row for item in transcript.ids() for row in transcript.rows(item, width=40)] == ['banner', 'later', '']
