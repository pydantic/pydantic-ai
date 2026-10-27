"""Commands that outlive the agent run, with a bounded foreground wait.

A command runs as a detached job inside the run's workspace (see `_jobs.py`): a wrapper shell in
its own session appends the command's combined stdout and stderr to an output log and publishes
the exit status as JSON. The tool call returns the PID and the log and status paths, which name
files inside the same workspace the model's other tools act on; the process is not tied to the
run or the event loop. There is no completion notification: the agent inspects the status file
itself.
"""

from __future__ import annotations

import codecs
from collections.abc import Mapping
from typing import Generic, Literal

import anyio

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._events import event_ctx
from pydantic_ai_harness.shell._events import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_ai_harness.shell._jobs import CONTROL_TIMEOUT, POLL_MAX, POLL_MIN, Job

MAX_FOREGROUND_WAIT: float = 270.0
"""Longest a foreground `shell` call waits before returning handles to the still-running command.

Bounded so a tool call returns before typical provider request timeouts, and so
the conversation keeps its request/response cadence instead of stalling on one
long command.
"""

_OUTPUT_TAIL_BYTES = 16_000
"""Bytes of the output log returned by a foreground call, and emitted as events per call."""

_EVENT_CHUNK_BYTES = 4096
_LINE_COUNT_LIMIT = 1_048_576

CommandMode = Literal['foreground', 'background']


async def _count_lines(job: Job) -> int | None:
    """Logical lines in the log, or `None` when it is over `_LINE_COUNT_LIMIT` and not scanned."""
    size = await job.size(job.output_path)
    if size > _LINE_COUNT_LIMIT:
        return None
    data = await job.read(job.output_path, 0, size)
    return data.count(b'\n') + int(bool(data) and not data.endswith(b'\n'))


async def _output_tail(job: Job) -> str:
    """The last `_OUTPUT_TAIL_BYTES` of the log, from a line start and marked when earlier output is left out."""
    size = await job.size(job.output_path)
    start = max(0, size - _OUTPUT_TAIL_BYTES)
    data = await job.read(job.output_path, start, size - start)
    if not start:
        return data.decode('utf-8', errors='replace')
    # Drop the cut first line, unless it is the only one.
    if 0 <= (newline := data.find(b'\n')) < len(data) - 1:
        start += newline + 1
        data = data[newline + 1 :]
    return f'[... output truncated, {start} earlier bytes omitted]\n' + data.decode('utf-8', errors='replace')


class _CommandOutput(Generic[AgentDepsT]):
    """Emit at most `_OUTPUT_TAIL_BYTES` of the log as events, decoded incrementally."""

    def __init__(self, job: Job, ctx: RunContext[AgentDepsT] | None, *, log_limited: bool) -> None:
        self.job = job
        # `None` where events cannot reach the run's event stream: then nothing is read or emitted.
        self.ctx = ctx
        self.log_limited = log_limited
        """Whether `max_log_bytes` can empty the log, so its clears are checked."""
        self.offset = 0
        """Position in the log, which goes back to 0 when `max_log_bytes` empties it."""
        self.emitted = 0
        self.clears = 0
        """The `log_clears` count the last emitted chunk was read under: two per clear."""
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    async def _clears(self) -> int:
        return await self.job.size(self.job.clears_path) if self.log_limited else 0

    async def _read(self) -> bytes:
        return await self.job.read(
            self.job.output_path, self.offset, min(_EVENT_CHUNK_BYTES, _OUTPUT_TAIL_BYTES - self.emitted)
        )

    async def emit(self) -> bool:
        """Emit the next chunk of the log, returning whether there was one."""
        if self.ctx is None or self.emitted >= _OUTPUT_TAIL_BYTES:
            return False
        chunk = await self._read()
        # Checked after the read: a chunk read since a clear began may come from the emptied log, at the
        # wrong position, so it is dropped. The watcher counts each clear before and after it, so an odd count
        # means it is under way: the log is read again at the next poll. Once it is done, the log is read again
        # from its start, in this call so that every chunk emitted counts against `_OUTPUT_TAIL_BYTES`.
        if (clears := await self._clears()) != self.clears:
            if clears % 2:
                return False
            self.clears = clears
            self.offset = 0
            self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            chunk = await self._read()
            if await self._clears() != clears:
                # Another clear began during that read too: start again at the next poll.
                return False
        self.offset += len(chunk)
        self.emitted += len(chunk)
        if chunk:
            await self.ctx.emit(CommandOutputEvent(text=self.decoder.decode(chunk)))
        return bool(chunk)

    async def drain(self) -> None:
        while await self.emit():
            pass

    async def finish(self) -> None:
        if self.ctx is None:
            return
        tail = self.decoder.decode(b'', final=True)
        if tail:
            await self.ctx.emit(CommandOutputEvent(text=tail))
        clears = await self._clears()
        size = await self.job.size(self.job.output_path)
        total_lines = None if clears else await _count_lines(self.job)
        # Counted again afterwards: a clear that began meanwhile leaves a size and count from two different logs.
        cleared = clears > 0 or await self._clears() != clears
        await self.ctx.emit(
            CommandFinishedEvent(
                pid=self.job.pid,
                output_path=self.job.output_path,
                status_path=self.job.status_path,
                exit_code=(await self.job.status())[1],
                truncated=cleared or size > self.offset,
                # An emptied log no longer holds the earlier lines, so their count is unknown.
                total_lines=None if cleared else total_lines,
            )
        )


async def run_persistent_command(
    ctx: RunContext[AgentDepsT],
    command: str,
    *,
    base: str,
    job_id: str | None,
    cwd: str,
    env: Mapping[str, str] | None,
    mode: CommandMode,
    timeout: float,
    log_limit: int | None = None,
) -> str:
    """Start `command` as a detached job and return its PID, output log, and status file.

    Foreground waits up to `timeout` for the exit status, polling with a backoff from 50 ms to
    1 s, then returns the handles either way; background returns them immediately.
    """
    if not 0 < timeout <= MAX_FOREGROUND_WAIT:
        raise ModelRetry(f'timeout must be greater than zero and at most {MAX_FOREGROUND_WAIT:g} seconds.')
    # Before launching: a toolset used outside its capability fails here, and must not leave a job running.
    events = event_ctx(ctx, 'Shell')
    job = await Job.launch(
        ctx.workspace, command, base=base, cwd=cwd, env=env, combined=True, log_limit=log_limit, job_id=job_id
    )
    output = _CommandOutput(job, events, log_limited=log_limit is not None)

    try:
        if output.ctx is not None:
            await output.ctx.emit(CommandStartedEvent(tool_call_id=ctx.tool_call_id, command=command, pid=job.pid))
        if mode == 'foreground':
            interval = POLL_MIN
            with anyio.move_on_after(timeout):
                while (await job.status())[0]:
                    emitted = await output.emit()
                    await anyio.sleep(interval)
                    interval = POLL_MIN if emitted else min(interval * 2, POLL_MAX)
            await output.drain()
        await output.finish()

        # Handles last: `ShellToolset.call_tool` keeps the tail of an over-long
        # result, and the PID and paths are what the model must not lose.
        result = ''
        if mode == 'foreground':
            result = await _output_tail(job)
            if result and not result.endswith('\n'):
                result += '\n'
        result += (
            f'PID: {job.pid} (supervisor; use `{job.stop_command}` to stop the whole process tree)\n'
            f'Output: {job.output_path}\nStatus: {job.status_path}'
        )
        if (status := await job.status_text()) is not None:
            result += f'\n{status}'
    except BaseException:
        # A cancelled or failed call cannot hand back its handles, so nothing is left running.
        with anyio.move_on_after(CONTROL_TIMEOUT, shield=True):
            await job.kill()
            await job.cleanup()
        raise
    return result
