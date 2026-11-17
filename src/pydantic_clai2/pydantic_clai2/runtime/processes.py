"""Process-tree cleanup shared by shell commands and plugin installation."""

import asyncio
import ntpath
import os
import signal
import subprocess
import sys
from contextlib import suppress


def taskkill_path() -> str:
    """Resolve `taskkill.exe` in the system directory, never through the working directory."""
    return ntpath.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'taskkill.exe')


def signal_process_group(pid: int, signum: int) -> None:
    """Signal the process group, including descendants that have not left it."""
    with suppress(ProcessLookupError):
        os.killpg(pid, signum)


async def kill_process_tree(process: asyncio.subprocess.Process | asyncio.SubprocessTransport) -> None:
    """Kill the process and its descendants; the caller must reap it afterwards."""
    pid = process.pid if isinstance(process, asyncio.subprocess.Process) else process.get_pid()
    if sys.platform == 'win32':
        try:
            killer = await asyncio.create_subprocess_exec(
                taskkill_path(),
                '/PID',
                str(pid),
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
            with suppress(ProcessLookupError):
                process.kill()
    else:
        # Descendants that left the session with `setsid()` detached on purpose, as under any shell.
        signal_process_group(pid, signal.SIGKILL)
