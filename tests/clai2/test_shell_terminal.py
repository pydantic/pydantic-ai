"""`!command` on CLAI's terminal runs on a pseudo-terminal and hands the terminal to full-screen programs."""

import asyncio
import io
import os
import signal
import struct
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_clai2.cli.shell_passthrough import run_shell_command
from pydantic_clai2.cli.shell_terminal import Terminal, TerminalText, attached_terminal
from pydantic_clai2.ui.menus.menu_worker import holding_output
from pydantic_clai2.ui.prompt.interrupts import Interrupts

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Windows has no pseudo-terminal')


def _set_size(fd: int, rows: int, columns: int) -> None:
    import fcntl
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, int]]:
    """A terminal for CLAI to run on: the user's end, which the test types into and reads, and CLAI's end."""
    user, clai = os.openpty()
    os.set_blocking(user, False)
    _set_size(clai, 30, 100)
    monkeypatch.setattr(
        'pydantic_clai2.cli.shell_passthrough.attached_terminal', lambda: Terminal(input=clai, output=clai)
    )
    yield user, clai
    os.close(user)
    os.close(clai)


@pytest.fixture
def user(terminal: tuple[int, int]) -> int:
    return terminal[0]


async def _screen_shows(user: int, needle: bytes, screen: bytearray) -> None:
    """Read what CLAI's terminal shows into `screen` until it contains `needle`."""
    with anyio.fail_after(5):
        while needle not in screen:
            try:
                screen.extend(os.read(user, 65536))
            except BlockingIOError:
                await anyio.sleep(0.01)


async def _console_shows(console: io.StringIO, needle: str) -> None:
    with anyio.fail_after(5):
        while needle not in console.getvalue():
            await anyio.sleep(0.01)


def _captured(command: str, status: str, stdout: str, stderr: str) -> str:
    return (
        f'The user ran a local shell command (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}'
    )


def _interactive(command: str, status: str) -> str:
    return (
        'The user ran an interactive local shell command in the terminal (not an agent tool call); '
        f'its output was not captured:\n$ {command}\n{status}'
    )


@contextmanager
def _recorded_hold(events: list[str]) -> Generator[None]:
    @contextmanager
    def hold(*, leave_screen: bool) -> Generator[None]:
        events.append(f'hold leave_screen={leave_screen}')
        yield
        events.append('release')

    with holding_output(hold):
        yield


async def test_output_is_captured_from_a_terminal(terminal: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """The command sees a terminal it controls; colours and terminal line ends do not reach the transcript."""
    import termios

    _, clai = terminal
    cooked = termios.ICANON | termios.ECHO
    assert termios.tcgetattr(clai)[3] & cooked == cooked

    monkeypatch.setenv('PAGER', 'less')
    command = (
        '[ -t 0 ] && [ -t 1 ] && echo tty; : < /dev/tty && echo controlling; printf \'pager=%s\\n\' "$PAGER"; '
        "printf '\\033[31mred\\033[0m\\r\\nline\\n'; printf to-stderr >&2"
    )
    console = io.StringIO()
    context = await run_shell_command(command, console=Console(file=console), interrupts=Interrupts(), cwd=Path.cwd())
    stdout = 'tty\ncontrolling\npager=cat\nred\nline\n'
    assert context == _captured(command, 'Exit code 0', stdout, 'to-stderr')
    # stderr comes through a pipe of its own, not the terminal, so it is shown as it arrives: anywhere in stdout.
    assert stdout in console.getvalue().replace('to-stderr', '')
    assert '\x1b' not in console.getvalue()
    # CLAI's terminal was raw while keys went to the command, and has its own modes back.
    assert termios.tcgetattr(clai)[3] & cooked == cooked


async def test_keys_reach_the_command(user: int) -> None:
    command = "printf 'name? '; IFS= read -r name; printf 'hi %s\\n' \"$name\""
    console = io.StringIO()
    task = asyncio.create_task(
        run_shell_command(command, console=Console(file=console), interrupts=Interrupts(), cwd=Path.cwd())
    )
    await _console_shows(console, 'prompt\nname? ')
    os.write(user, b'bob\r')
    context = await task
    # The pseudo-terminal echoes what was typed, as a terminal does.
    assert context == _captured(command, 'Exit code 0', 'name? bob\nhi bob\n', '')


async def test_ctrl_c_interrupts_the_command(
    terminal: tuple[int, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import termios

    user, clai = terminal
    monkeypatch.chdir(tmp_path)
    interrupts = Interrupts()
    console = io.StringIO()
    task = asyncio.create_task(
        run_shell_command(
            ': > ready; exec sleep 30', console=Console(file=console), interrupts=interrupts, cwd=Path.cwd()
        )
    )
    with anyio.fail_after(5):
        while not (tmp_path / 'ready').exists() or termios.tcgetattr(clai)[3] & termios.ICANON:
            await anyio.sleep(0.01)  # pragma: lax no cover -- the shell may already be ready.
    os.write(user, b'\x03')
    with anyio.fail_after(5):
        context = await task
    assert '\nInterrupted\n' in context
    assert 'Interrupted (' in console.getvalue()
    assert not interrupts.exit_requested


async def test_full_screen_program_gets_the_terminal(user: int) -> None:
    """Once the command switches to the alternate screen, it owns the terminal and nothing more is saved."""
    command = (
        "printf 'before\\n'; printf '\\033[?1049hFULL'; stty raw -echo; printf READY; IFS= read -r line; stty sane; printf oops >&2; "
        "printf 'got:%s' \"$line\"; printf '\\033[?1049l'; exit 3"
    )
    events: list[str] = []
    console = io.StringIO()
    screen = bytearray()
    with _recorded_hold(events):
        task = asyncio.create_task(
            run_shell_command(command, console=Console(file=console), interrupts=Interrupts(), cwd=Path.cwd())
        )
        await _screen_shows(user, b'\x1b[?1049hFULLREADY', screen)
        assert events == ['hold leave_screen=True']
        os.write(user, b'typed\x03\n')  # Ctrl-C is the program's own key now.
        await _screen_shows(user, b'\x1b[?1049l', screen)
        context = await task
    assert events == ['hold leave_screen=True', 'release']
    assert b'oops' in screen
    assert b'got:typed\x03' in screen
    assert context == _interactive(command, 'Exit code 3')
    shown = console.getvalue().split('saved for the next prompt\n', 1)[1]
    assert shown.startswith('before\n')
    assert 'Full-screen command: only its exit status is saved' in shown
    assert 'Exit code 3 (' in shown
    for hidden in ('FULL', 'got:', 'oops'):
        assert hidden not in shown


async def test_cancelling_a_full_screen_program_restores_the_terminal(
    user: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    screen = bytearray()
    with _recorded_hold(events):
        task = asyncio.create_task(
            run_shell_command(
                "printf '\\033[?1049h'; exec sleep 30",
                console=Console(file=io.StringIO()),
                interrupts=Interrupts(),
                cwd=Path.cwd(),
            )
        )
        await _screen_shows(user, b'\x1b[?1049h', screen)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
            await task
    assert events == ['hold leave_screen=True', 'release']


async def test_resizing_reaches_the_command(user: int, tmp_path: Path) -> None:
    """The command runs in the session's directory and sees the terminal's size change."""
    command = "trap 'stty size > resized; exit 0' WINCH; stty size > started; : > ready; while :; do sleep 0.05; done"
    task = asyncio.create_task(
        run_shell_command(command, console=Console(file=io.StringIO()), interrupts=Interrupts(), cwd=tmp_path)
    )
    with anyio.fail_after(5):
        while not (tmp_path / 'ready').exists():
            await anyio.sleep(0.01)  # pragma: lax no cover -- the shell may already be ready.
    assert (tmp_path / 'started').read_text() == '30 100\n'
    # Resize CLAI's terminal as a terminal emulator would, and tell CLAI as the kernel would.
    _set_size(user, 40, 120)
    os.kill(os.getpid(), signal.SIGWINCH)
    with anyio.fail_after(5):
        context = await task
    assert '\nExit code 0\n' in context
    assert (tmp_path / 'resized').read_text() == '40 120\n'


async def test_background_job_cannot_hold_the_command_open(user: int) -> None:
    """A job left running keeps the pseudo-terminal open; CLAI stops waiting for its output shortly after."""
    with anyio.fail_after(1.5):
        context = await run_shell_command(
            'sleep 2 & echo started', console=Console(file=io.StringIO()), interrupts=Interrupts(), cwd=Path.cwd()
        )
    assert context.endswith('stdout:\nstarted\n\n\nstderr:\n')


async def test_spawn_failure_releases_the_pseudo_terminal(user: int) -> None:
    console = io.StringIO()
    context = await run_shell_command(
        'echo a\x00b', console=Console(file=console), interrupts=Interrupts(), cwd=Path.cwd()
    )
    assert '\nShell error: embedded null byte\n' in context


def test_terminal_text_holds_split_sequences() -> None:
    text = TerminalText()
    assert text.feed(b'a\x1b[3') == b'a'
    assert text.feed(b'1mred\r') == b'red'
    assert text.feed(b'\r\nb\x1b]0;title\x07c\x1bPdata\x1b\\d\x1b=e') == b'\nbcde'
    assert text.feed(b'f\x1b') == b'f'
    assert text.feed(b'', final=True) == b''
    assert text.feed(b'\x1b[' + b'1' * 5000) == b'[' + b'1' * 5000  # Too long to be a sequence.


@pytest.mark.parametrize(
    ('platform', 'tty', 'expected'), [('linux', True, Terminal()), ('linux', False, None), ('win32', True, None)]
)
def test_attached_terminal(
    monkeypatch: pytest.MonkeyPatch, platform: str, tty: bool, expected: Terminal | None
) -> None:
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(os, 'isatty', lambda fd: tty)  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    assert attached_terminal() == expected
