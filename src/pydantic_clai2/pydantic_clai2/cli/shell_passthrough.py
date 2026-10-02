"""Run `!command` input in the system shell instead of starting an agent turn."""

import asyncio
import codecs
import contextlib
import io
import ntpath
import os
import signal
import subprocess
import sys
import time

from rich.console import Console
from rich.text import Text

from pydantic_ai._utils import gather
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.rendering import theme

HELP = '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); save command and output for the next prompt'

# Matches `subprocess.run`: a Ctrl-C'd child gets this long to exit on its own SIGINT before it is killed.
_INTERRUPT_GRACE = 0.25


def _taskkill_path() -> str:
    """Resolve `taskkill.exe` in the system directory, never through the working directory."""
    return ntpath.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'taskkill.exe')


def _signal_process_group(process: asyncio.subprocess.Process, signum: int) -> None:
    """Signal the shell's process group, which holds every descendant that has not left it."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signum)


def _interrupt(process: asyncio.subprocess.Process) -> None:
    """Forward Ctrl-C, which the terminal delivers to CLAI but not to a command in its own session."""
    if sys.platform != 'win32':  # The Windows console delivers Ctrl-C to every attached process.
        _signal_process_group(process, signal.SIGINT)


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Kill the shell and its descendants."""
    if sys.platform == 'win32':
        try:
            killer = await asyncio.create_subprocess_exec(
                _taskkill_path(),
                '/PID',
                str(process.pid),
                '/T',
                '/F',
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await killer.wait()
        except OSError:
            killer_succeeded = False
        else:
            killer_succeeded = killer.returncode == 0
        if not killer_succeeded:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
    else:
        # Descendants that left the session with `setsid()` detached on purpose, as under any shell.
        _signal_process_group(process, signal.SIGKILL)


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
    stdout, stderr = io.StringIO(), io.StringIO()
    exit_code: int | None = None
    needs_newline = False

    async def read_output(stream: asyncio.StreamReader, output: io.StringIO) -> None:
        nonlocal needs_newline
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        while True:
            chunk = await stream.read(8192)
            text = decoder.decode(chunk, final=not chunk)
            output.write(text)
            if text:
                console.print(text, end='', markup=False, highlight=False, soft_wrap=True)
                needs_newline = not text.endswith('\n')
            if not chunk:
                break

    async def execute() -> None:
        # A new POSIX session gives the command a process group to kill; `start_new_session` is
        # ignored on Windows, where `taskkill /T` follows parent PIDs and the console's Ctrl-C
        # still reaches the command.
        # A child can signal us before asyncio returns its process handle. Keep spawning shielded so we can reap it.
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_shell(
                command, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        )

        async def capture_output() -> None:
            nonlocal exit_code
            process = await spawn_task
            assert process.stdout is not None and process.stderr is not None
            await gather(read_output(process.stdout, stdout), read_output(process.stderr, stderr))
            exit_code = await process.wait()

        output_task = asyncio.create_task(capture_output())
        try:
            await asyncio.shield(output_task)
        except asyncio.CancelledError:
            process = await spawn_task
            _interrupt(process)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), _INTERRUPT_GRACE)
            await _kill_process_tree(process)
            await process.wait()
            raise
        finally:
            await output_task

    started = time.monotonic()
    error: str | None = None
    completed = True
    try:
        completed = await interrupts.run(execute())
    except (OSError, ValueError) as exc:  # `ValueError`: the command text contains a NUL byte.
        error = str(exc)
    elapsed = f' ({time.monotonic() - started:.1f}s)'
    if needs_newline:
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
        f'$ {command}\n{status}\n\nstdout:\n{stdout.getvalue()}\n\nstderr:\n{stderr.getvalue()}'
    )
