"""Run `!command` input in the system shell instead of starting an agent turn."""

import asyncio
import contextlib
import ntpath
import os
import signal
import subprocess
import sys
import time

from rich.console import Console
from rich.text import Text

from . import theme
from .interrupts import Interrupts

HELP = '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); it is not sent to the agent'

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


async def run_shell_command(command: str, *, console: Console, interrupts: Interrupts) -> None:
    """Run with inherited stdio so interactive programs own the terminal until they exit.

    Ctrl-C cancels only this command, not CLAI; cancellation terminates the shell's process tree.
    """
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print('Shell passthrough, not sent to the agent', style=theme.color(theme.MUTED))
    exit_code = 0

    async def execute() -> None:
        nonlocal exit_code
        # A new POSIX session gives the command a process group to kill; `start_new_session` is
        # ignored on Windows, where `taskkill /T` follows parent PIDs and the console's Ctrl-C
        # still reaches the command.
        process = await asyncio.create_subprocess_shell(command, start_new_session=True)
        try:
            exit_code = await process.wait()
        except asyncio.CancelledError:
            _interrupt(process)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), _INTERRUPT_GRACE)
            await _kill_process_tree(process)
            await process.wait()
            raise

    started = time.monotonic()
    try:
        completed = await interrupts.run(execute())
    except (OSError, ValueError) as exc:  # `ValueError`: the command text contains a NUL byte.
        console.print(f'Shell error: {exc}', style=theme.color(theme.ERROR), markup=False)
        console.print()
        return
    elapsed = f' ({time.monotonic() - started:.1f}s)'
    if not completed:
        console.print(f'Interrupted{elapsed}', style=theme.color(theme.WARNING), highlight=False)
    elif exit_code:
        console.print(f'Exit code {exit_code}{elapsed}', style=theme.color(theme.ERROR), highlight=False)
    else:
        console.print(f'Done{elapsed}', style=theme.color(theme.SUCCESS), highlight=False)
    console.print()
