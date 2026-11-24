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
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from pydantic_ai.workspaces import CommandResult, Workspace
from pydantic_ai_pax.cli.shell_terminal import COMMAND_ENVIRONMENT, PtyRun, attached_terminal, ctrl_c_as_key
from pydantic_ai_pax.errors import error_message
from pydantic_ai_pax.runtime.processes import kill_process_tree, signal_process_group
from pydantic_ai_pax.ui.menus.menu_worker import full_terminal
from pydantic_ai_pax.ui.prompt.interrupts import Interrupts
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text
from pydantic_ai_pax.ui.stderr_relay import terminal_stderr

if TYPE_CHECKING:
    from pydantic_ai_pax.runtime.session import ShellLineWorkspace

HELP = (
    '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); save command and output for the next prompt.'
    ' In a /sandbox conversation it runs in the sandbox.'
    ' On this machine, a full-screen program such as `!vim` gets the terminal, and only its exit status is saved\n'
    '!!COMMAND: Give COMMAND the terminal from the start, for prompts and REPLs; save only its exit status.'
    ' Only on this machine, not in a sandbox'
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
        self.exit_code: int | None = None
        self._decoders = {fd: codecs.getincrementaldecoder('utf-8')(errors='replace') for fd in (1, 2)}

    def _write(self, fd: int, data: bytes, *, final: bool = False) -> None:
        text = self._decoders[fd].decode(data, final=final)
        captured = self.stdout if fd == 1 else self.stderr
        remaining = _MAX_OUTPUT_CHARS - captured.tell()
        if remaining >= 0:
            captured.write(text[:remaining])
            if len(text) > remaining:
                captured.write(_truncated())
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
    """Forward Ctrl-C, which the terminal delivers to Pax but not to a command in its own session."""
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


async def run_shell_command(command: str, *, console: Console, interrupts: Interrupts, cwd: Path) -> str:
    """Run locally in `cwd`, the session's directory, and return context for the next model request.

    Output streams to the transcript and is saved. On Pax's terminal the command runs on a
    pseudo-terminal (`PtyRun`), and a full-screen program gets the terminal itself, saving only its
    exit status. A leading `!` (typed as `!!command`) gives the command the terminal from the start
    (`run_in_terminal`). Ctrl-C cancels only this command, not Pax; cancellation terminates the
    shell's process tree.
    """
    if command.startswith('!'):
        return await run_in_terminal(command[1:].strip(), console=console, cwd=cwd)
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

    started = time.monotonic()
    error: str | None = None
    completed = True
    try:
        completed = await interrupts.run(_execute(command, output=output, cwd=cwd))
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
        status = _report(console, error=error, completed=completed, exit_code=output.exit_code, started=started)
        return _interactive_context(command, status)
    status = _report(console, error=error, completed=completed, exit_code=output.exit_code, started=started)
    return (
        f'The user ran a local shell command (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{output.stdout.getvalue()}\n\nstderr:\n{output.stderr.getvalue()}'
    )


async def run_workspace_command(
    command: str, *, place: 'ShellLineWorkspace', console: Console, interrupts: Interrupts
) -> str | None:
    """Run in `place`, such as the conversation's sandbox, and return context for the next model request.

    The workspace returns the output once the command exits, so it shows then, not as it arrives.
    Ctrl-C cancels the command, read as a key (`ctrl_c_as_key`) as for a local one. A `!!command` needs
    this terminal, so it is refused and `None` returned.
    """
    if command.startswith('!'):
        console.print(
            f'!! needs this terminal, which {place.name} does not have. Nothing ran.',
            style=theme.color(theme.WARNING),
            markup=False,
        )
        return None
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print(f'Runs in {place.name}; command and output saved for the next prompt', style=theme.color(theme.MUTED))
    started = time.monotonic()
    error: str | None = None
    result: CommandResult | None = None

    async def execute() -> None:
        nonlocal result
        result = await Workspace(place.workspace).run(command, shell=True)

    completed = True
    terminal = attached_terminal()
    try:
        with contextlib.nullcontext() if terminal is None else ctrl_c_as_key(terminal, interrupts.cancel):
            completed = await interrupts.run(execute())
    except Exception as exc:  # noqa: BLE001 -- a failed sandbox command must not end the shell.
        error = error_message(exc)
    stdout = stderr = ''
    exit_code: int | None = None
    if result is not None:
        stdout, stderr, exit_code = _capped(result.stdout), _capped(result.stderr), result.exit_code
        for text in (stdout, stderr):
            if text:
                # The sandbox's output, which a file it prints may control, gets no terminal escapes; the model sees it as is.
                console.print(
                    terminal_text(text),
                    end='' if text.endswith('\n') else '\n',
                    markup=False,
                    highlight=False,
                    soft_wrap=True,
                )
    status = _report(console, error=error, completed=completed, exit_code=exit_code, started=started)
    return (
        f'The user ran a shell command in {place.name} (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}'
    )


def _truncated() -> str:
    """Marks a stream cut to `_MAX_OUTPUT_CHARS` in the context a `!` line saves, locally or in a sandbox."""
    return f'\n[Output truncated after {_MAX_OUTPUT_CHARS} characters]\n'


def _capped(text: str) -> str:
    """`text` cut to the characters a `!` command saves for the model, marked when cut."""
    return text if len(text) <= _MAX_OUTPUT_CHARS else text[:_MAX_OUTPUT_CHARS] + _truncated()


async def _execute(command: str, *, output: _ShellOutput, cwd: Path) -> None:
    """Run `command` in `cwd` through `output`, on its pseudo-terminal if it has one, and record its exit status."""
    pty = output.pty
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
            cwd=cwd,
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
            cwd=cwd,
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
        try:
            _interrupt(process)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(output.exited.wait(), _INTERRUPT_GRACE)
            await kill_process_tree(process)
        except PermissionError as denial:
            # macOS refuses a signal to a group left with only zombies, a sandbox can refuse any, and Windows can refuse
            # the kill. Closing the transport below still tries the direct child, but nothing shows the rest stopped.
            raise PermissionError(
                'Ctrl-C could not stop the command: the system refused to signal it, so it may still be running'
            ) from denial
        await output.exited.wait()
        await output.drain()
        if pty is not None:
            await pty.drain()
        raise
    finally:
        process, _ = await spawn_task
        process.close()
        await output.closed.wait()
        output.exit_code = process.get_returncode()


async def run_in_terminal(command: str, *, console: Console, cwd: Path) -> str:
    """Hand the terminal to a command in `cwd` from the start; return context naming it and its exit status.

    The live panel leaves the screen, the command inherits Pax's standard streams, and the panel
    repaints once it exits. The command stays in Pax's process group, in the terminal's foreground,
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
            exit_code = await _wait_in_terminal(command, cwd=cwd)
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


async def _wait_in_terminal(command: str, *, cwd: Path) -> int:
    """Run `command` with Pax's standard streams and return its exit status; cancellation kills it."""
    # Spawning is shielded, so a cancellation that arrives meanwhile still finds the process to kill.
    spawn = asyncio.create_task(asyncio.create_subprocess_shell(command, stderr=terminal_stderr(), cwd=cwd))
    try:
        process = await asyncio.shield(spawn)
        return await process.wait()
    except asyncio.CancelledError:
        # Only the shell is killed, not a process tree: the command shares Pax's process group, which
        # keeps it in the terminal's foreground for Ctrl-C, Ctrl-Z, and resizes, so `killpg` would kill
        # Pax too. Ctrl-C is the command's, so this runs only when Pax's own task is cancelled; a
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
    """Leave Ctrl-C to the command, which the terminal signals along with Pax.

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
