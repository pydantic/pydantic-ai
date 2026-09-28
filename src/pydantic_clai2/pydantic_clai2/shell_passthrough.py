"""Run `!command` input in the system shell instead of starting an agent turn."""

import asyncio
import contextlib
import time

from rich.console import Console
from rich.text import Text

from . import theme
from .interrupts import Interrupts

HELP = '!COMMAND: Run COMMAND with the system shell (/bin/sh, or cmd.exe on Windows); it is not sent to the agent'

# Matches `subprocess.run`: a Ctrl-C'd child gets this long to exit on its own SIGINT before it is killed.
_INTERRUPT_GRACE = 0.25


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

    Ctrl-C reaches the child through the terminal and cancels only this command, not CLAI.
    """
    console.print(Text.assemble(('$ ', theme.color(theme.ACCENT)), command))
    console.print('Shell passthrough, not sent to the agent', style=theme.color(theme.MUTED))
    exit_code = 0

    async def execute() -> None:
        nonlocal exit_code
        process = await asyncio.create_subprocess_shell(command)
        try:
            exit_code = await process.wait()
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), _INTERRUPT_GRACE)
            with contextlib.suppress(ProcessLookupError):
                process.kill()
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
