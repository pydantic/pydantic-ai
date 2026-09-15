"""Persistent local commands with bounded foreground waiting."""

from __future__ import annotations

import fnmatch
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Literal

import anyio
from anyio.to_thread import run_sync
from pydantic_ai import ModelRetry

from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS


async def shell(
    workspace: Path,
    command: str,
    *,
    mode: Literal['foreground', 'background'] = 'foreground',
    timeout: float = 270,
) -> str:
    """Run a shell command, returning durable PID, output and exit-status paths."""
    if '\0' in command:
        raise ModelRetry('command must not contain NUL characters.')
    if not 0 < timeout <= 270:
        raise ModelRetry('timeout must be greater than zero and at most 270 seconds.')
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
    }
    directory = Path(tempfile.mkdtemp(prefix='coder-shell-'))
    supervisor = Path(__file__).with_name('_supervisor.py')
    try:
        process = subprocess.Popen(
            [sys.executable, str(supervisor), str(directory), command],
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        directory.rmdir()
        raise ModelRetry(f'Cannot start command: {exc}') from exc
    # Reap the supervisor without tying command lifetime to an async run or loop.
    threading.Thread(target=process.wait, daemon=True).start()
    status_path = directory / 'status.json'
    try:
        if mode == 'foreground':
            with anyio.move_on_after(timeout):
                while not status_path.exists() or json.loads(status_path.read_text())['exit_code'] is None:
                    if process.returncode is not None and not status_path.exists():
                        raise ModelRetry(f'Shell supervisor exited with {process.returncode}; logs: {directory}')
                    await anyio.sleep(0.05)
    except BaseException:
        # A cancelled call cannot return handles. Terminate its process group
        # instead of leaving an unreachable command behind.
        if os.name == 'nt':  # pragma: no cover
            if process.poll() is None:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=True, capture_output=True)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        with anyio.CancelScope(shield=True):
            await run_sync(process.wait)
        raise
    handles = (
        f'PID: {process.pid} (supervisor/session leader; use kill -- -{process.pid} to stop the process group)\n'
        f'Output: {directory / "output.log"}\nStatus: {status_path}\n'
    )
    if status_path.exists():
        handles += f'{status_path.read_text()}\n'
    output = directory / 'output.log'
    if mode == 'foreground' and output.exists():
        with output.open('rb') as source:
            source.seek(max(0, output.stat().st_size - 16000))
            handles += source.read(16000).decode('utf-8', errors='replace')
    return handles
