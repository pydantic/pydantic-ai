"""Run `!command` input in the system shell instead of starting an agent turn."""

import asyncio
import codecs
import contextlib
import io
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from types import FrameType

from rich.console import Console
from rich.text import Text

from pydantic_clai2.cli.shell_terminal import COMMAND_ENVIRONMENT, PtyRun, attached_terminal
from pydantic_clai2.runtime._processes import kill_process_tree, signal_process_group
from pydantic_clai2.ui.menus.menu_worker import full_terminal
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.stderr_relay import terminal_stderr

HELP = (
    '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); save command and output for the next prompt.'
    ' A full-screen program such as `!vim` gets the terminal, and only its exit status is saved\n'
    '!!COMMAND: Give COMMAND the terminal from the start, for prompts and REPLs; save only its exit status'
)

# Matches `subprocess.run`: a Ctrl-C'd child gets this long to exit on its own SIGINT before it is killed.
_INTERRUPT_GRACE = 0.25
_OUTPUT_DRAIN_GRACE = 0.1
_MAX_OUTPUT_CHARS = 100_000


class _ShellOutput(asyncio.SubprocessProtocol):
    """Observe shell exit independently of descendants keeping its output pipes open."""

    def __init__(self, *, console: Console) -> None:
        self.console = console
        self.pty: PtyRun | None = None
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.exited, self.closed = asyncio.Event(), asyncio.Event()
        self.needs_newline = False
        self._decoders = {fd: codecs.getincrementaldecoder('utf-8')(errors='replace') for fd in (1, 2)}

    def _write(self, fd: int, data: bytes, *, final: bool = False) -> None:
        text = self._decoders[fd].decode(data, final=final)
        captured = self.stdout if fd == 1 else self.stderr
        remaining = _MAX_OUTPUT_CHARS - captured.tell()
        if remaining >= 0:
            captured.write(text[:remaining])
            if len(text) > remaining:
                captured.write(f'\n[Output truncated after {_MAX_OUTPUT_CHARS} characters]\n')
        if text:
            self.console.print(text, end='', markup=False, highlight=False, soft_wrap=True)
            self.needs_newline = not text.endswith('\n')

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.pty is not None and self.pty.handed_over:
            # A full-screen program's errors belong on its screen, uncaptured like the rest of it.
            self.pty.pass_through(data)
        else:
            self._write(fd, data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self._write(fd, b'', final=True)

    def process_exited(self) -> None:
        self.exited.set()

    def connection_lost(self, exc: Exception | None) -> None:
        self.closed.set()

    async def drain(self) -> None:
        """Drain buffered output without waiting indefinitely for detached descendants."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.closed.wait(), _OUTPUT_DRAIN_GRACE)


def _interrupt(process: asyncio.SubprocessTransport) -> None:
    """Forward Ctrl-C, which the terminal delivers to CLAI but not to a command in its own session."""
    # The Windows console delivers Ctrl-C to every attached process.
    if sys.platform != 'win32':  # pragma: no branch
        signal_process_group(process.get_pid(), signal.SIGINT)


def shell_command(text: str) -> str | None:
    """Return the command for `!command` input, or `None` when the input is a prompt.

    A bare `!` or `!!`, or one followed only by whitespace, stays a prompt. `!!command` returns
    `!command`, which `run_shell_command` gives the terminal from the start.
    """
    stripped = text.strip()
    if not stripped.startswith('!'):
        return None
    command = stripped[1:].strip()
    return command if command.lstrip('!').strip() else None


async def run_shell_command(command: str, *, console: Console, interrupts: Interrupts) -> str:
    """Run locally and return context for the next model request.

    Output streams to the transcript and is saved. On CLAI's terminal the command runs on a
    pseudo-terminal (`PtyRun`), and a full-screen program gets the terminal itself, saving only its
    exit status. A leading `!` (typed as `!!command`) gives the command the terminal from the start
    (`run_in_terminal`). Ctrl-C cancels only this command, not CLAI; cancellation terminates the
    shell's process tree.
    """
    if command.startswith('!'):
        return await run_in_terminal(command[1:].strip(), console=console)
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print('Shell command and output saved for the next prompt', style=theme.color(theme.MUTED))
    output = _ShellOutput(console=console)
    terminal = attached_terminal()
    pty = (
        None
        if terminal is None
        else PtyRun(
            terminal=terminal,
            capture=lambda data: output.pipe_data_received(1, data),
            interrupt=interrupts.cancel,
        )
    )
    output.pty = pty
    exit_code: int | None = None

    async def execute() -> None:
        nonlocal exit_code
        loop = asyncio.get_running_loop()
        # A new POSIX session gives the command a process group to kill; `start_new_session` is
        # ignored on Windows, where `taskkill /T` follows parent PIDs and the console's Ctrl-C
        # still reaches the command.
        # A child can signal us before asyncio returns its process handle. Keep spawning shielded so we can reap it.
        spawn_task = asyncio.create_task(
            loop.subprocess_shell(
                lambda: output,
                command,
                start_new_session=True,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if pty is None
            else loop.subprocess_shell(
                lambda: output,
                pty.command(command),
                start_new_session=True,
                stdin=pty.follower,
                stdout=pty.follower,
                stderr=subprocess.PIPE,
                env={**os.environ, **COMMAND_ENVIRONMENT},
            )
        )
        try:
            process, _ = await asyncio.shield(spawn_task)
            if pty is not None:
                pty.start()
            await output.exited.wait()
            await output.drain()
            if pty is not None:
                await pty.drain()
        except asyncio.CancelledError:
            process, _ = await spawn_task
            _interrupt(process)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(output.exited.wait(), _INTERRUPT_GRACE)
            await kill_process_tree(process)
            await output.exited.wait()
            await output.drain()
            if pty is not None:
                await pty.drain()
            raise
        finally:
            process, _ = await spawn_task
            process.close()
            await output.closed.wait()
            exit_code = process.get_returncode()

    started = time.monotonic()
    error: str | None = None
    completed = True
    try:
        completed = await interrupts.run(execute())
    except (OSError, ValueError) as exc:  # `ValueError`: the command text contains a NUL byte.
        error = str(exc)
    finally:
        if pty is not None:
            pty.close()
            output.pipe_connection_lost(1, None)
    if output.needs_newline:
        console.print()
    if pty is not None and pty.handed_over:
        console.print('Full-screen command: only its exit status is saved', style=theme.color(theme.MUTED))
        status = _report(console, error=error, completed=completed, exit_code=exit_code, started=started)
        return _interactive_context(command, status)
    status = _report(console, error=error, completed=completed, exit_code=exit_code, started=started)
    return (
        f'The user ran a local shell command (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{output.stdout.getvalue()}\n\nstderr:\n{output.stderr.getvalue()}'
    )


async def run_in_terminal(command: str, *, console: Console) -> str:
    """Hand the terminal to a command from the start and return context naming it and its exit status.

    The live panel leaves the screen, the command inherits CLAI's standard streams, and the panel
    repaints once it exits. The command stays in CLAI's process group, in the terminal's foreground,
    so it gets Ctrl-C and window size changes itself. Its output is never captured.
    """
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print(
        'Interactive command: it has the terminal, and its output is not saved', style=theme.color(theme.MUTED)
    )
    started = time.monotonic()
    error: str | None = None
    exit_code = 0
    with full_terminal(), terminal_modes_kept(), _ctrl_c_ignored():
        try:
            exit_code = await _wait_in_terminal(command)
        except (OSError, ValueError) as exc:  # `ValueError`: the command text contains a NUL byte.
            error = str(exc)
    status = _report(console, error=error, completed=True, exit_code=exit_code, started=started)
    return _interactive_context(command, status)


def _report(console: Console, *, error: str | None, completed: bool, exit_code: int | None, started: float) -> str:
    """Print how the command ended and return the status saved for the model."""
    elapsed = f' ({time.monotonic() - started:.1f}s)'
    if error is not None:
        status = f'Shell error: {error}'
        console.print(status, style=theme.color(theme.ERROR), markup=False)
    elif not completed:
        status = 'Interrupted'
        console.print(f'Interrupted{elapsed}', style=theme.color(theme.WARNING), highlight=False)
    elif exit_code:
        status = f'Exit code {exit_code}'
        console.print(f'{status}{elapsed}', style=theme.color(theme.ERROR), highlight=False)
    else:
        status = 'Exit code 0'
        console.print(f'Done{elapsed}', style=theme.color(theme.SUCCESS), highlight=False)
    console.print()
    return status


def _interactive_context(command: str, status: str) -> str:
    return (
        'The user ran an interactive local shell command in the terminal (not an agent tool call); '
        f'its output was not captured:\n$ {command}\n{status}'
    )


async def _wait_in_terminal(command: str) -> int:
    """Run `command` with CLAI's standard streams and return its exit status; cancellation kills it."""
    # Spawning is shielded, so a cancellation that arrives meanwhile still finds the process to kill.
    spawn = asyncio.create_task(asyncio.create_subprocess_shell(command, stderr=terminal_stderr()))
    try:
        process = await asyncio.shield(spawn)
        return await process.wait()
    except asyncio.CancelledError:
        # Only the shell is killed, not a process tree: the command shares CLAI's process group, which
        # keeps it in the terminal's foreground for Ctrl-C, Ctrl-Z, and resizes, so `killpg` would kill
        # CLAI too. Ctrl-C is the command's, so this runs only when CLAI's own task is cancelled; a
        # terminal hangup still signals the whole group. Nothing is left to kill when spawning failed
        # or the process was already reaped.
        with contextlib.suppress(OSError, ValueError):
            process = await spawn
            process.kill()
            await process.wait()
        raise


@contextlib.contextmanager
def terminal_modes_kept(fd: int = 0) -> Generator[None]:
    """Restore the terminal's modes on `fd` afterwards, in case a command exits leaving it in raw mode."""
    if sys.platform == 'win32' or not os.isatty(fd):
        yield
        return
    import termios

    modes = termios.tcgetattr(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, modes)


def _ignore(signum: int, frame: FrameType | None) -> None:
    """Ctrl-C belongs to the command that has the terminal."""


@contextlib.contextmanager
def _ctrl_c_ignored() -> Generator[None]:
    """Leave Ctrl-C to the command, which the terminal signals along with CLAI.

    A handler that does nothing, not `SIG_IGN`, which the command would inherit.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, _ignore)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
