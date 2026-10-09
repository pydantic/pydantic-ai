"""macOS's `MallocStackLogging` fork notices stay off the terminal; everything else written to stderr reaches it."""

import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from pydantic_clai2.ui import stderr_relay
from pydantic_clai2.ui.stderr_relay import drop_fork_notices, relay_stderr, terminal_stderr

_NOTICE = b"python(4242) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.\n"

posix_only = pytest.mark.skipif(sys.platform == 'win32', reason='needs a POSIX pseudo-terminal')


@pytest.mark.parametrize(
    ('data', 'expected'),
    [
        (_NOTICE, b''),
        (b'error: boom\n' + _NOTICE + b'warning: careful\n', b'error: boom\nwarning: careful\n'),
        (_NOTICE.replace(b'\n', b'\r\n') + b'Password: ', b'Password: '),
        (b'Python(12) ' + _NOTICE.split(b' ', 1)[1], b''),
        # Other allocator diagnostics are the user's to see, such as those `MallocStackLogging=1` asks for.
        (b'python(1) MallocStackLogging: recording malloc (and VM allocation) stacks using lite mode\n',) * 2,
        (_NOTICE.rstrip(b'\n'),) * 2,
        (b'quoting ' + _NOTICE,) * 2,
    ],
)
def test_drop_fork_notices(data: bytes, expected: bytes) -> None:
    assert drop_fork_notices(data) == expected


def test_a_line_left_open_is_not_a_notice() -> None:
    """Text completing a line an earlier write started is kept, even when it reads like a notice."""
    assert drop_fork_notices(_NOTICE + _NOTICE, at_line_start=False) == _NOTICE
    assert drop_fork_notices(b'\r' + _NOTICE, at_line_start=False) == b'\r'


@contextmanager
def _stderr_to(fd: int) -> Generator[None]:
    """Point file descriptor 2 at `fd` for the block.

    Not a fixture: pytest points it at its capture file again when the test body starts.
    """
    saved = os.dup(2)
    os.dup2(fd, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _shown(screen: int) -> bytes:
    """What the terminal has shown so far; the relay has finished writing before it hands the terminal back."""
    os.set_blocking(screen, False)
    chunks: list[bytes] = []
    with suppress(BlockingIOError):
        while True:
            chunks.append(os.read(screen, 65536))
    return b''.join(chunks).replace(b'\r\n', b'\n')


@posix_only
@pytest.mark.subprocess(reason='a child process must inherit the relayed stderr, which is the behavior under test')
def test_relay_keeps_fork_notices_off_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stderr_relay.sys, 'platform', 'darwin')
    screen, terminal = os.openpty()
    with _stderr_to(terminal):
        os.close(terminal)
        with relay_stderr():
            # Children inherit the pipe rather than the terminal, so what a child writes, and what the allocator
            # writes in a fork's child before `exec`, is filtered the same way.
            assert not os.isatty(2)
            os.write(2, b'parent error\n' + _NOTICE)
            subprocess.run(['sh', '-c', 'cat >&2'], input=b'child error\n' + _NOTICE + b'child done\n', check=True)
            os.write(2, b'Password: ')
            original = terminal_stderr()
            assert original is not None and os.isatty(original)
        assert terminal_stderr() is None
        assert os.isatty(2)
        os.write(2, b'after\n')
        try:
            assert _shown(screen) == b'parent error\nchild error\nchild done\nPassword: after\n'
        finally:
            os.close(screen)


@posix_only
def test_relay_leaves_stderr_alone_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stderr_relay.sys, 'platform', 'linux')
    screen, terminal = os.openpty()
    try:
        with _stderr_to(terminal), relay_stderr():
            assert os.isatty(2)
            assert terminal_stderr() is None
    finally:
        os.close(terminal)
        os.close(screen)


def test_relay_leaves_stderr_alone_when_it_is_not_a_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stderr_relay.sys, 'platform', 'darwin')
    log = os.open(tmp_path / 'stderr.log', os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        with _stderr_to(log), relay_stderr():
            assert terminal_stderr() is None
            os.write(2, _NOTICE)
    finally:
        os.close(log)
    assert (tmp_path / 'stderr.log').read_bytes() == _NOTICE
