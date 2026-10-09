"""Run `!command` input in the system shell instead of starting an agent turn."""

import asyncio
import codecs
import contextlib
import io
import signal
import subprocess
import sys
import time

from rich.console import Console
from rich.text import Text

from pydantic_clai2.runtime._processes import kill_process_tree, signal_process_group
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.rendering import theme

HELP = '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); save command and output for the next prompt'

# Matches `subprocess.run`: a Ctrl-C'd child gets this long to exit on its own SIGINT before it is killed.
_INTERRUPT_GRACE = 0.25
_OUTPUT_DRAIN_GRACE = 0.1
_MAX_OUTPUT_CHARS = 100_000


class _ShellOutput(asyncio.SubprocessProtocol):
    """Observe shell exit independently of descendants keeping its output pipes open."""

    def __init__(self, *, console: Console) -> None:
        self.console = console
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

    A bare `!`, or `!` followed only by whitespace, stays a prompt.
    """
    stripped = text.strip()
    if not stripped.startswith('!'):
        return None
    return stripped[1:].strip() or None


async def run_shell_command(command: str, *, console: Console, interrupts: Interrupts) -> str:
    """Run locally, streaming output and returning context for the next model request.

    Ctrl-C cancels only this command, not CLAI; cancellation terminates the shell's process tree.
    """
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print('Shell command and output saved for the next prompt', style=theme.color(theme.MUTED))
    output = _ShellOutput(console=console)
    exit_code: int | None = None

    async def execute() -> None:
        nonlocal exit_code
        # A new POSIX session gives the command a process group to kill; `start_new_session` is
        # ignored on Windows, where `taskkill /T` follows parent PIDs and the console's Ctrl-C
        # still reaches the command.
        # A child can signal us before asyncio returns its process handle. Keep spawning shielded so we can reap it.
        spawn_task = asyncio.create_task(
            asyncio.get_running_loop().subprocess_shell(
                lambda: output,
                command,
                start_new_session=True,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        try:
            process, _ = await asyncio.shield(spawn_task)
            await output.exited.wait()
            await output.drain()
        except asyncio.CancelledError:
            process, _ = await spawn_task
            _interrupt(process)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(output.exited.wait(), _INTERRUPT_GRACE)
            await kill_process_tree(process)
            await output.exited.wait()
            await output.drain()
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
    elapsed = f' ({time.monotonic() - started:.1f}s)'
    if output.needs_newline:
        console.print()
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
    return (
        f'The user ran a local shell command (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{output.stdout.getvalue()}\n\nstderr:\n{output.stderr.getvalue()}'
    )
