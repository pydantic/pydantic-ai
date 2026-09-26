"""Commands that outlive the agent run, with a bounded foreground wait.

A command is started by a detached supervisor process (`_supervisor.py`) that
runs it under its own session, appends its combined stdout and stderr to an
output log, and publishes the exit status as JSON. The tool call returns the
PID and the log/status paths; the process is not tied to the run, the event
loop, or the calling interpreter. There is no completion notification: the
agent inspects the status file itself.
"""

from __future__ import annotations

import codecs
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Generic, Literal

import anyio
from anyio.to_thread import run_sync
from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.shell._events import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent

MAX_FOREGROUND_WAIT: float = 270.0
"""Longest a foreground `shell` call waits before returning handles to the still-running command.

Bounded so a tool call returns before typical provider request timeouts, and so
the conversation keeps its request/response cadence instead of stalling on one
long command.
"""

_OUTPUT_TAIL_BYTES = 16_000
"""Bytes of the output log returned by a foreground call, and emitted as events per call."""

_EVENT_CHUNK_BYTES = 4096
_POLL_INTERVAL = 0.05
_LINE_COUNT_LIMIT = 1_048_576

CommandMode = Literal['foreground', 'background']


class CommandStatus(BaseModel):
    """The supervisor's published status: the command's PID and exit code (`None` while running)."""

    pid: int
    exit_code: int | None


def _count_lines(path: Path) -> int | None:
    """Logical lines in the log, or `None` when it is over `_LINE_COUNT_LIMIT` and not scanned."""
    if not path.exists():
        return 0
    remaining = path.stat().st_size
    if remaining > _LINE_COUNT_LIMIT:
        return None
    count = 0
    last = b''
    with path.open('rb') as source:
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk:  # pragma: no cover -- log externally truncated during the snapshot scan.
                break
            remaining -= len(chunk)
            count += chunk.count(b'\n')
            last = chunk[-1:]
    return count + int(bool(last) and last != b'\n')


class _CommandOutput(Generic[AgentDepsT]):
    """Emit at most `_OUTPUT_TAIL_BYTES` of the log as events, decoded incrementally."""

    def __init__(self, path: Path, ctx: RunContext[AgentDepsT]) -> None:
        self.path = path
        self.ctx = ctx
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    async def emit(self) -> bool:
        if self.offset >= _OUTPUT_TAIL_BYTES or not self.path.exists():
            return False
        with self.path.open('rb') as source:
            source.seek(self.offset)
            chunk = source.read(min(_EVENT_CHUNK_BYTES, _OUTPUT_TAIL_BYTES - self.offset))
        self.offset += len(chunk)
        if chunk:
            await self.ctx.emit(CommandOutputEvent(text=self.decoder.decode(chunk)))
        return bool(chunk)

    async def drain(self) -> None:
        while await self.emit():
            pass

    async def finish(self, *, pid: int, status_path: Path) -> None:
        tail = self.decoder.decode(b'', final=True)
        if tail:
            await self.ctx.emit(CommandOutputEvent(text=tail))
        status = CommandStatus.model_validate_json(status_path.read_text()) if status_path.exists() else None
        await self.ctx.emit(
            CommandFinishedEvent(
                pid=pid,
                output_path=str(self.path),
                status_path=str(status_path),
                exit_code=status.exit_code if status else None,
                truncated=self.path.exists() and self.path.stat().st_size > self.offset,
                total_lines=await run_sync(_count_lines, self.path),
            )
        )


def _finished(status_path: Path) -> bool:
    return status_path.exists() and json.loads(status_path.read_text())['exit_code'] is not None


def _kill_session(process: subprocess.Popen[bytes]) -> None:
    """Terminate the supervisor's whole session; a cancelled call cannot hand back its handles."""
    if os.name == 'nt':  # pragma: no cover
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=True, capture_output=True)
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def run_persistent_command(
    ctx: RunContext[AgentDepsT],
    command: str,
    *,
    cwd: Path,
    env: dict[str, str] | None,
    mode: CommandMode,
    timeout: float,
) -> str:
    """Start `command` under a supervisor and return its PID, output log, and status file.

    Foreground waits up to `timeout` for the exit status, then returns the
    handles either way; background returns them immediately.
    """
    if not 0 < timeout <= MAX_FOREGROUND_WAIT:
        raise ModelRetry(f'timeout must be greater than zero and at most {MAX_FOREGROUND_WAIT:g} seconds.')
    directory = Path(tempfile.mkdtemp(prefix='harness-shell-'))
    supervisor = Path(__file__).with_name('_supervisor.py')
    try:
        process = subprocess.Popen(
            [sys.executable, str(supervisor), str(directory), command],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except BaseException:
        directory.rmdir()
        raise
    # Reap the supervisor without tying the command's lifetime to a run or an event loop.
    threading.Thread(target=process.wait, daemon=True).start()
    status_path = directory / 'status.json'
    output_path = directory / 'output.log'
    output = _CommandOutput(output_path, ctx)

    try:
        await ctx.emit(CommandStartedEvent(tool_call_id=ctx.tool_call_id, command=command, pid=process.pid))
        if mode == 'foreground':
            with anyio.move_on_after(timeout):
                while not _finished(status_path):
                    if process.returncode is not None:
                        if not status_path.exists():
                            raise ModelRetry(f'Shell supervisor exited with {process.returncode}; logs: {directory}')
                        # The supervisor is gone, so the status will never be completed;
                        # the handles still name the command it may have left running.
                        break
                    await output.emit()
                    await anyio.sleep(_POLL_INTERVAL)
            await output.drain()
        await output.finish(pid=process.pid, status_path=status_path)
    except ModelRetry:
        # The supervisor already exited; its directory is what the message points at.
        raise
    except BaseException:
        _kill_session(process)
        with anyio.CancelScope(shield=True):
            await run_sync(process.wait)
            shutil.rmtree(directory, ignore_errors=True)
        raise

    # Handles last: `ShellToolset.call_tool` keeps the tail of an over-long
    # result, and the PID and paths are what the model must not lose.
    result = ''
    if mode == 'foreground' and output_path.exists():
        with output_path.open('rb') as source:
            source.seek(max(0, output_path.stat().st_size - _OUTPUT_TAIL_BYTES))
            result = source.read(_OUTPUT_TAIL_BYTES).decode('utf-8', errors='replace')
            if result and not result.endswith('\n'):
                result += '\n'
    stop = f'taskkill /PID {process.pid} /T /F' if os.name == 'nt' else f'kill -- -{process.pid}'
    result += (
        f'PID: {process.pid} (supervisor/session leader; use `{stop}` to stop the whole process tree)\n'
        f'Output: {output_path}\nStatus: {status_path}'
    )
    if status_path.exists():
        result += f'\n{status_path.read_text()}'
    return result
