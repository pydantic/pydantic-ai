"""The terminal background query, answered by a fake terminal on a pseudo-terminal.

A unit test, since no model is involved: the claim is what goes over the terminal and what comes back.
"""

import os
import select
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager

import pytest

from pydantic_ai_pax.ui import terminal_background as background_module
from pydantic_ai_pax.ui.terminal_background import terminal_background
from tests._timeouts import READINESS_WAIT_TIMEOUT

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='needs a POSIX pseudo-terminal')

DA1 = b'\x1b[?1;2;4c'
QUERY = b'\x1b]11;?\x07\x1b[c'


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('COLORFGBG', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')


@contextmanager
def on_a_terminal() -> Generator[int]:
    """A pseudo-terminal on stdin and stdout; yields the terminal's side, which reads queries and writes replies.

    Entered in the test body, not a fixture, since pytest's output capture takes stdout back between phases.
    """
    leader, follower = os.openpty()
    saved = os.dup(0), os.dup(1)
    os.dup2(follower, 0)
    os.dup2(follower, 1)
    try:
        yield leader
    finally:
        os.dup2(saved[0], 0)
        os.dup2(saved[1], 1)
        for fd in (*saved, follower, leader):
            os.close(fd)


def answer(leader: int, reply: bytes) -> tuple[threading.Thread, list[bytes]]:
    """Answer the query with `reply` once it arrives; returns the thread and what it read before answering."""

    def respond() -> None:
        received = b''
        while not received.endswith(QUERY[-3:]) and select.select([leader], [], [], READINESS_WAIT_TIMEOUT)[0]:
            received += os.read(leader, 1024)
        read.append(received)
        os.write(leader, reply)

    read: list[bytes] = []
    thread = threading.Thread(target=respond, daemon=True)
    thread.start()
    return thread, read


@pytest.mark.parametrize(
    ('reply', 'expected'),
    [
        (b'\x1b]11;rgb:ffff/ffff/ffff\x07' + DA1, '#FFFFFF'),
        (b'\x1b]11;rgb:fd/f6/e3\x1b\\' + DA1, '#FDF6E3'),
        (b'\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\' + DA1, '#1E1E2E'),
        (DA1, None),
    ],
    ids=['bel', 'st-two-digit', 'dark', 'no-background-reply'],
)
def test_reads_the_reported_background(reply: bytes, expected: str | None) -> None:
    import termios

    with on_a_terminal() as leader:
        responder, read = answer(leader, reply)
        assert terminal_background() == expected
        responder.join()
        assert read == [QUERY]
        restored = termios.ICANON | termios.ECHO
        assert termios.tcgetattr(0)[3] & restored == restored  # Echo and line editing are back.


@pytest.mark.parametrize(('value', 'expected'), [('0;15', '#FFFFFF'), ('0;default;7', '#FFFFFF'), ('15;0', None)])
def test_falls_back_to_colorfgbg(value: str, expected: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORFGBG', value)
    with on_a_terminal() as leader:
        responder, _ = answer(leader, DA1)
        assert terminal_background() == expected
        responder.join()


def test_a_silent_terminal_is_waited_on_briefly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background_module, '_TIMEOUT', 0.01)
    with on_a_terminal():
        assert terminal_background() is None


def test_keys_typed_ahead_are_left_for_the_prompt() -> None:
    """Keys typed before Enter, which line mode holds back from `select`, are not read, and no query is sent."""
    with on_a_terminal() as leader:
        os.write(leader, b'hi')
        assert terminal_background() is None
        os.write(leader, b'\n')
        assert os.read(0, 1024) == b'hi\n'
        echoed = os.read(leader, 1024) if select.select([leader], [], [], 0)[0] else b''
        assert b'\x1b]11' not in echoed


def test_not_asked_without_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORFGBG', '0;15')
    reader, writer = os.pipe()
    saved = os.dup(0)
    os.dup2(reader, 0)
    try:
        assert terminal_background() == '#FFFFFF'
    finally:
        os.dup2(saved, 0)
        for fd in (saved, reader, writer):
            os.close(fd)
