"""Asking the terminal for its background, over a pseudo-terminal whose far end plays the terminal."""

import io
import os
import pty
import select
import termios
import tty
from collections.abc import Iterator
from typing import IO

import pytest

from pydantic_clai2.ui.prompt.terminal_background import detect_background

DA1 = b'\x1b[?62;22c'


class Display(io.StringIO):
    """The terminal's screen: it records what CLAI writes, and types `answer` back once asked."""

    def __init__(self, keyboard: int, answer: bytes) -> None:
        super().__init__()
        self.keyboard, self.answer = keyboard, answer

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        os.write(self.keyboard, self.answer)
        return super().write(text)


@pytest.fixture
def terminal() -> Iterator[tuple[int, IO[str]]]:
    """A pseudo-terminal: the keyboard end the test types into, and the end CLAI reads as its input."""
    keyboard, end = pty.openpty()
    with open(end, encoding='utf-8') as stdin:
        yield keyboard, stdin
    os.close(keyboard)


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.delenv('COLORFGBG', raising=False)


@pytest.mark.parametrize(
    ('answer', 'background'),
    [
        pytest.param(b'\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\' + DA1, '#1E1E1E', id='four-digit-st'),
        pytest.param(b'\x1b]11;rgb:ff/ff/ff\x07' + DA1, '#FFFFFF', id='two-digit-bel'),
        pytest.param(b'\x1b]11;rgb:f/8/0\x07' + DA1, '#FF8800', id='one-digit'),
        pytest.param(b'\x1b]11;rgba:0000/0000/0000/ffff\x1b\\' + DA1, '#000000', id='rgba'),
    ],
)
def test_the_terminal_answer_is_the_background(terminal: tuple[int, IO[str]], answer: bytes, background: str) -> None:
    keyboard, stdin = terminal
    mode = termios.tcgetattr(stdin)
    display = Display(keyboard, answer)
    assert detect_background(stdin=stdin, stdout=display) == background
    assert display.getvalue() == '\x1b]11;?\x1b\\\x1b[c'
    restored = termios.tcgetattr(stdin)
    # A BSD kernel sets its own local flag when canonical mode returns, so compare the local flags CLAI changes.
    changed = termios.ICANON | termios.ECHO
    assert restored[3] & changed == mode[3] & changed, 'canonical input and echo are back'
    assert [*restored[:3], *restored[4:]] == [*mode[:3], *mode[4:]]


@pytest.mark.parametrize(
    ('colorfgbg', 'background'),
    [
        pytest.param('0;15', '#FFFFFF', id='light'),
        pytest.param('15;0', '#000000', id='dark'),
        pytest.param('7;default;8', '#000000', id='rxvt-dark'),
        pytest.param('default;default', None, id='unknown'),
        pytest.param(None, None, id='unset'),
    ],
)
@pytest.mark.parametrize(
    'answer',
    [
        pytest.param(DA1, id='no-osc-11'),
        pytest.param(b'\x1b]11;rgb:zz/00/00\x07' + DA1, id='malformed'),
        pytest.param(b'', id='silent'),
    ],
)
def test_without_an_answer_colorfgbg_tells_light_from_dark(
    terminal: tuple[int, IO[str]],
    monkeypatch: pytest.MonkeyPatch,
    answer: bytes,
    colorfgbg: str | None,
    background: str | None,
) -> None:
    keyboard, stdin = terminal
    if colorfgbg is not None:
        monkeypatch.setenv('COLORFGBG', colorfgbg)
    assert detect_background(stdin=stdin, stdout=Display(keyboard, answer), timeout=0.01) == background


def test_keys_typed_ahead_are_left_for_the_editor(
    terminal: tuple[int, IO[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer would arrive behind those keys, so CLAI does not ask and reads nothing."""
    keyboard, stdin = terminal
    monkeypatch.setenv('COLORFGBG', '0;15')
    os.write(keyboard, b'typed\n')
    display = Display(keyboard, b'\x1b]11;rgb:00/00/00\x07' + DA1)
    assert detect_background(stdin=stdin, stdout=display) == '#FFFFFF'
    assert display.getvalue() == ''
    assert select.select([stdin], [], [], 0)[0]
    assert os.read(stdin.fileno(), 64) == b'typed\n'


def test_keys_typed_after_the_answer_are_left_for_the_editor(terminal: tuple[int, IO[str]]) -> None:
    keyboard, stdin = terminal
    display = Display(keyboard, b'\x1b]11;rgb:00/00/00\x07' + DA1 + b'typed')
    assert detect_background(stdin=stdin, stdout=display) == '#000000'
    # As the editor does: read without waiting for a line, and without flushing what is queued.
    tty.setcbreak(stdin, termios.TCSANOW)
    assert select.select([stdin], [], [], 1)[0]
    assert os.read(stdin.fileno(), 64) == b'typed'


@pytest.mark.parametrize('piped', ['stdin', 'stdout'])
def test_a_pipe_on_either_side_is_never_asked(
    terminal: tuple[int, IO[str]], monkeypatch: pytest.MonkeyPatch, piped: str
) -> None:
    keyboard, stdin = terminal
    monkeypatch.setenv('COLORFGBG', '0;15')
    display = Display(keyboard, DA1)
    if piped == 'stdin':
        assert detect_background(stdin=io.StringIO(), stdout=display) is None
    else:
        assert detect_background(stdin=stdin, stdout=io.StringIO()) is None
    assert display.getvalue() == ''


def test_a_dumb_terminal_is_never_asked(terminal: tuple[int, IO[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    keyboard, stdin = terminal
    monkeypatch.setenv('TERM', 'dumb')
    display = Display(keyboard, DA1)
    assert detect_background(stdin=stdin, stdout=display) is None
    assert display.getvalue() == ''


def test_a_stream_that_only_claims_to_be_a_terminal_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """`termios` refuses a pipe even when its stream says it is a terminal."""
    monkeypatch.setenv('COLORFGBG', '15;0')
    read, write = os.pipe()

    class Claimed(io.StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return read

    display = Display(write, DA1)
    try:
        assert detect_background(stdin=Claimed(), stdout=display) == '#000000'
    finally:
        os.close(read)
        os.close(write)
    assert display.getvalue() == ''
