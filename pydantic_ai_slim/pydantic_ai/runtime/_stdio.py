"""Host-side ABC for stdio-pipe-based code runtimes.

Provides ``StdioSandboxRuntime``, an intermediate abstract base class that
handles the NDJSON protocol, tool dispatch, interrupt/checkpoint logic, and
resume-via-re-execution. Concrete subclasses (Docker, E2B, Modal, etc.)
implement a single method: ``_start_driver``.
"""

from __future__ import annotations

import asyncio
import base64
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import pydantic

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.messages import ToolReturnContent, tool_return_ta
from pydantic_ai.runtime.abstract import (
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
)

# TypeAdapter for deserializing checkpoint values with proper type reconstruction.
_checkpoint_result_ta: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    ToolReturnContent,
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64'),
)


class DriverProcess:
    """Interface for communicating with a driver subprocess.

    Concrete implementations wrap platform-specific process types
    (``asyncio.subprocess.Process``, SDK handles, etc.).
    """

    async def read_line(self) -> bytes:
        """Read a single newline-terminated line from the driver's stdout."""
        raise NotImplementedError

    async def write_line(self, data: bytes) -> None:
        """Write a line to the driver's stdin (must include trailing newline)."""
        raise NotImplementedError

    async def read_stderr(self) -> bytes:
        """Read all available stderr output from the driver."""
        raise NotImplementedError

    async def kill(self) -> None:
        """Terminate the driver process."""
        raise NotImplementedError


def _serialize_checkpoint(
    completed_results: dict[int, Any],
    interrupted_calls: list[InterruptedToolCall],
) -> bytes:
    """Serialize a stdio checkpoint into an opaque bytes blob.

    Each completed result is individually serialized to JSON bytes via Pydantic's
    ``tool_return_ta``, then base64-encoded for embedding in the outer JSON payload.
    """
    raw_results = {
        str(k): base64.b64encode(tool_return_ta.dump_json(v)).decode('ascii') for k, v in completed_results.items()
    }
    pending_calls = {
        ic.call.call_id: {
            'function_name': ic.call.function_name,
            'args': list(ic.call.args),
            'kwargs': ic.call.kwargs,
        }
        for ic in interrupted_calls
    }
    payload = {
        'completed_results': raw_results,
        'pending_calls': pending_calls,
    }
    return json.dumps(payload).encode('utf-8')


@dataclass
class _DeserializedCheckpoint:
    completed_results: dict[str, Any]
    pending_calls: dict[str, dict[str, Any]]


def _deserialize_checkpoint(checkpoint: bytes) -> _DeserializedCheckpoint:
    """Deserialize a stdio checkpoint back into its components."""
    payload = json.loads(checkpoint)
    return _DeserializedCheckpoint(
        completed_results=payload.get('completed_results', {}),
        pending_calls=payload.get('pending_calls', {}),
    )


@dataclass
class StdioSandboxRuntime(CodeRuntime):
    """Abstract base for all stdio-pipe-based code runtimes.

    Subclasses implement ``_start_driver`` to launch the driver script inside
    their specific sandbox environment. Everything else — protocol handling,
    tool dispatch, interrupt/checkpoint, and resume — is handled here.
    """

    @abstractmethod
    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverProcess:
        """Launch the driver process and send the init message.

        The implementation should:
        1. Start a process running ``_driver.py``
        2. Write the JSON-encoded init message as the first line to stdin
        3. Return a ``DriverProcess`` wrapping the subprocess handles

        Args:
            init_msg: The init message dict to send to the driver.

        Returns:
            A DriverProcess for communicating with the driver.
        """
        ...

    async def run(
        self,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
        *,
        signatures: list[str],
        checkpoint: bytes | None = None,
    ) -> Any:
        if checkpoint is not None:
            return await self._resume_from_checkpoint(checkpoint, code, functions, call_tool)

        init_msg: dict[str, Any] = {
            'type': 'init',
            'code': code,
            'functions': functions,
        }
        process = await self._start_driver(init_msg)
        try:
            return await self._execution_loop(process, call_tool)
        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError, ModelRetry):
            raise
        except Exception as e:
            raise CodeRuntimeError(f'Driver communication error: {e}') from e

    async def _resume_from_checkpoint(
        self,
        checkpoint: bytes,
        code: str,
        functions: list[str],
        call_tool: ToolCallback,
    ) -> Any:
        """Resume execution from a serialized checkpoint via re-execution with a result cache."""
        try:
            ckpt = _deserialize_checkpoint(checkpoint)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise CodeRuntimeError(f'Invalid checkpoint data: {e}')

        # Build result cache: decode base64-encoded results for the driver
        result_cache: dict[str, Any] = {}
        for k, v in ckpt.completed_results.items():
            result_cache[k] = _checkpoint_result_ta.validate_json(base64.b64decode(v))

        init_msg: dict[str, Any] = {
            'type': 'init',
            'code': code,
            'functions': functions,
            'result_cache': result_cache,
        }
        process = await self._start_driver(init_msg)
        try:
            return await self._execution_loop(process, call_tool)
        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError, ModelRetry):
            raise
        except Exception as e:
            raise CodeRuntimeError(f'Driver communication error: {e}') from e

    async def _execution_loop(self, process: DriverProcess, call_tool: ToolCallback) -> Any:
        """Run the dual-wait event loop: read driver stdout + dispatch tool tasks.

        Simultaneously reads protocol messages from the driver and manages
        asyncio tasks for tool call execution. Handles interrupts (ApprovalRequired,
        CallDeferred) by building a checkpoint when all pending tasks are settled.
        """
        completed_results: dict[int, Any] = {}
        interrupted_calls: list[InterruptedToolCall] = []
        tool_tasks: dict[int, asyncio.Task[Any]] = {}
        call_id_to_fc: dict[int, FunctionCall] = {}

        stdout_task: asyncio.Task[bytes] = asyncio.ensure_future(process.read_line())

        try:
            while True:
                waitables: list[asyncio.Task[Any]] = [stdout_task, *tool_tasks.values()]
                done, _ = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    if task is stdout_task:
                        result = await self._handle_stdout(task, process, call_tool, tool_tasks, call_id_to_fc)
                        if result is not _SENTINEL:
                            return result
                        stdout_task = asyncio.ensure_future(process.read_line())
                    else:
                        await self._handle_tool_done(
                            task, process, tool_tasks, call_id_to_fc, completed_results, interrupted_calls, stdout_task
                        )

                if interrupted_calls and not tool_tasks:
                    # Before interrupting, drain any pending call messages from
                    # stdout. With network-based runtimes (e.g. Modal) stdout
                    # messages may arrive with inter-line latency, so later call
                    # messages might not have been read yet.
                    try:
                        drain_done, _ = await asyncio.wait([stdout_task], timeout=0.5)
                    except Exception:
                        drain_done = set()
                    if drain_done:
                        # More data available — process it before interrupting
                        continue
                    await _cancel_task(stdout_task)
                    await process.kill()
                    checkpoint = _serialize_checkpoint(completed_results, interrupted_calls)
                    raise CodeInterruptedError(interrupted_calls=interrupted_calls, checkpoint=checkpoint)

        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError, ModelRetry):
            raise
        except Exception as e:
            await self._cancel_all(tool_tasks, stdout_task, process)
            raise CodeRuntimeError(f'Execution loop error: {e}') from e

    @staticmethod
    async def _handle_stdout(
        task: asyncio.Task[bytes],
        process: DriverProcess,
        call_tool: ToolCallback,
        tool_tasks: dict[int, asyncio.Task[Any]],
        call_id_to_fc: dict[int, FunctionCall],
    ) -> Any:
        """Handle a completed stdout read task. Returns _SENTINEL to continue, or the final result."""
        raw = task.result()
        if not raw:
            stderr = b''
            try:
                stderr = await asyncio.wait_for(process.read_stderr(), timeout=1.0)
            except Exception:
                pass
            err_msg = stderr.decode(errors='replace').strip() if stderr else 'Driver process exited unexpectedly'
            raise CodeRuntimeError(err_msg)

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return _SENTINEL

        msg_type = msg.get('type')

        if msg_type == 'call':
            cid = msg['id']
            fc = FunctionCall(
                call_id=str(cid),
                function_name=msg['function'],
                args=tuple(msg.get('args', ())),
                kwargs=msg.get('kwargs', {}),
            )
            call_id_to_fc[cid] = fc
            tool_tasks[cid] = asyncio.ensure_future(call_tool(fc))
        elif msg_type == 'complete':
            await process.kill()
            return msg.get('result')
        elif msg_type == 'error':
            await process.kill()
            error_type = msg.get('error_type', 'runtime')
            error_msg = msg.get('error', 'Unknown driver error')
            if error_type == 'syntax':
                raise CodeSyntaxError(error_msg)
            raise CodeRuntimeError(error_msg)

        return _SENTINEL

    @staticmethod
    async def _handle_tool_done(
        task: asyncio.Task[Any],
        process: DriverProcess,
        tool_tasks: dict[int, asyncio.Task[Any]],
        call_id_to_fc: dict[int, FunctionCall],
        completed_results: dict[int, Any],
        interrupted_calls: list[InterruptedToolCall],
        stdout_task: asyncio.Task[Any],
    ) -> None:
        """Handle a completed tool task: send result, accumulate interrupt, or propagate error."""
        cid = _task_to_cid(task, tool_tasks)
        if cid is None:
            return
        del tool_tasks[cid]

        try:
            result = task.result()
            completed_results[cid] = result
            result_msg = json.dumps({'type': 'result', 'id': cid, 'result': result}, default=str) + '\n'
            await process.write_line(result_msg.encode())
        except (ApprovalRequired, CallDeferred) as e:
            fc = call_id_to_fc[cid]
            interrupted_calls.append(InterruptedToolCall(type=e, call=fc))
        except ModelRetry:
            await _cancel_all(tool_tasks, stdout_task, process)
            raise
        except Exception as e:
            await _cancel_all(tool_tasks, stdout_task, process)
            raise ModelRetry(str(e))

    @staticmethod
    async def _cancel_all(
        tool_tasks: dict[int, asyncio.Task[Any]],
        stdout_task: asyncio.Task[Any],
        process: DriverProcess,
    ) -> None:
        """Cancel all pending tasks and kill the driver process."""
        for t in tool_tasks.values():
            t.cancel()
        stdout_task.cancel()
        for t in [*tool_tasks.values(), stdout_task]:
            await _cancel_task(t)
        try:
            await process.kill()
        except Exception:
            pass


# Sentinel object for _handle_stdout to signal "continue loop"
_SENTINEL = object()


def _task_to_cid(task: asyncio.Task[Any], tool_tasks: dict[int, asyncio.Task[Any]]) -> int | None:
    """Find the call ID for a completed tool task."""
    for cid, t in tool_tasks.items():
        if t is task:
            return cid
    return None


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel a task and suppress the CancelledError."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _cancel_all(
    tool_tasks: dict[int, asyncio.Task[Any]],
    stdout_task: asyncio.Task[Any],
    process: DriverProcess,
) -> None:
    """Cancel all pending tasks and kill the driver process."""
    for t in tool_tasks.values():
        t.cancel()
    stdout_task.cancel()
    for t in [*tool_tasks.values(), stdout_task]:
        await _cancel_task(t)
    try:
        await process.kill()
    except Exception:
        pass
