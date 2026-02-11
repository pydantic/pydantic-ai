"""Host-side ABC for driver-based code runtimes.

Provides ``DriverBasedRuntime``, an intermediate abstract base class that
handles the NDJSON protocol, tool dispatch, interrupt/checkpoint logic, and
resume-via-re-execution. Concrete subclasses (Docker, E2B, Modal, etc.)
implement a single method: ``_start_driver``.
"""

from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from pydantic import ValidationError

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import tool_return_ta
from pydantic_ai.runtime.abstract import (
    CodeInterruptedError,
    CodeRuntime,
    CodeRuntimeError,
    CodeSyntaxError,
    FunctionCall,
    InterruptedToolCall,
    ToolCallback,
    checkpoint_result_ta,
    deserialize_checkpoint,
    serialize_checkpoint_results,
)


class DriverTransport(ABC):
    """Interface for communicating with a driver process.

    Concrete implementations wrap platform-specific transport types
    (``asyncio.subprocess.Process``, SDK handles, WebSocket connections, etc.).
    """

    @abstractmethod
    async def read_line(self) -> bytes:
        """Read a single newline-terminated line from the driver's stdout."""
        ...

    @abstractmethod
    async def write_line(self, data: bytes) -> None:
        """Write a line to the driver's stdin (must include trailing newline)."""
        ...

    @abstractmethod
    async def read_stderr(self) -> bytes:
        """Read all available stderr output from the driver."""
        ...

    @abstractmethod
    async def kill(self) -> None:
        """Terminate the driver process."""
        ...


class _StdoutSignal(Enum):
    """Typed signals from _handle_stdout indicating what happened."""

    CONTINUE = auto()
    CALLS_READY = auto()
    NEW_CALL = auto()


@dataclass(frozen=True)
class _FinalResult:
    """Wraps the final result value from a completed driver execution."""

    value: Any


@dataclass
class DriverBasedRuntime(CodeRuntime):
    """Abstract base for all driver-based code runtimes.

    Subclasses implement ``_start_driver`` to launch the driver script inside
    their specific sandbox environment. Everything else — protocol handling,
    tool dispatch, interrupt/checkpoint, and resume — is handled here.
    """

    @abstractmethod
    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        """Launch the driver process and send the init message.

        The implementation should:
        1. Start a process running ``_driver.py``
        2. Write the JSON-encoded init message as the first line to stdin
        3. Return a ``DriverTransport`` wrapping the subprocess handles

        Args:
            init_msg: The init message dict to send to the driver.

        Returns:
            A DriverTransport for communicating with the driver.
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
        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError):
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
            ckpt = deserialize_checkpoint(checkpoint)

            # Build result cache: decode base64-encoded results for the driver.
            # Normalize to JSON-compatible form so the init_msg can be serialized
            # with json.dumps — validate_json may reconstruct Pydantic model instances
            # that aren't JSON-serializable without an explicit dump step.
            result_cache: dict[str, Any] = {}
            for k, v in ckpt.completed_results.items():
                reconstructed = checkpoint_result_ta.validate_json(base64.b64decode(v))
                result_cache[k] = tool_return_ta.dump_python(reconstructed, mode='json')
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as e:
            raise CodeRuntimeError(f'Invalid checkpoint data: {e}') from e

        init_msg: dict[str, Any] = {
            'type': 'init',
            'code': code,
            'functions': functions,
            'result_cache': result_cache,
        }
        process = await self._start_driver(init_msg)
        try:
            return await self._execution_loop(process, call_tool)
        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError):
            raise
        except Exception as e:
            raise CodeRuntimeError(f'Driver communication error: {e}') from e

    async def _execution_loop(self, process: DriverTransport, call_tool: ToolCallback) -> Any:
        """Run the dual-wait event loop: read driver stdout + dispatch tool tasks.

        Simultaneously reads protocol messages from the driver and manages
        asyncio tasks for tool call execution. Handles interrupts (ApprovalRequired,
        CallDeferred) by building a checkpoint when all pending tasks are settled.
        """
        completed_results: dict[int, Any] = {}
        interrupted_calls: list[InterruptedToolCall] = []
        tool_tasks: dict[int, asyncio.Task[Any]] = {}
        task_id_to_cid: dict[int, int] = {}
        call_id_to_fc: dict[int, FunctionCall] = {}
        calls_ready_seen = False

        stdout_task: asyncio.Task[bytes] = asyncio.create_task(process.read_line())

        try:
            while True:
                waitables: list[asyncio.Task[Any]] = [stdout_task, *tool_tasks.values()]
                done, _ = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    if task is stdout_task:
                        result = await self._handle_stdout(
                            task, process, call_tool, tool_tasks, task_id_to_cid, call_id_to_fc
                        )
                        if isinstance(result, _FinalResult):
                            return result.value
                        elif result is _StdoutSignal.CALLS_READY:
                            calls_ready_seen = True
                        elif result is _StdoutSignal.NEW_CALL:
                            calls_ready_seen = False
                        stdout_task = asyncio.create_task(process.read_line())
                    else:
                        await self._handle_tool_done(
                            task,
                            process,
                            tool_tasks,
                            task_id_to_cid,
                            call_id_to_fc,
                            completed_results,
                            interrupted_calls,
                            stdout_task,
                        )

                # Only interrupt once we've received the calls_ready fence
                # from the driver, confirming all call messages for this batch
                # have been sent. Without this, network-based runtimes (Modal,
                # E2B, etc.) could lose in-transit call messages.
                #
                # Known limitation: if a successful tool result sent back to the
                # driver causes the code to fire new calls (a subsequent "wave"),
                # those calls may not yet be on stdout when this condition is
                # checked, resulting in a premature interrupt. This is not a
                # correctness bug — re-execution with the result cache on resume
                # will replay the missed calls — but it adds an extra
                # interrupt+resume round-trip. The Monty runtime does not have
                # this issue because its snapshot captures full interpreter state.
                if interrupted_calls and not tool_tasks and calls_ready_seen:
                    await _cancel_task(stdout_task)
                    await process.kill()
                    checkpoint = serialize_checkpoint_results(completed_results, interrupted_calls)
                    raise CodeInterruptedError(interrupted_calls=interrupted_calls, checkpoint=checkpoint)

        except (CodeInterruptedError, CodeSyntaxError, CodeRuntimeError):
            raise
        except Exception as e:
            await _cancel_all(tool_tasks, stdout_task, process)
            raise CodeRuntimeError(f'Execution loop error: {e}') from e

    @staticmethod
    async def _handle_stdout(
        task: asyncio.Task[bytes],
        process: DriverTransport,
        call_tool: ToolCallback,
        tool_tasks: dict[int, asyncio.Task[Any]],
        task_id_to_cid: dict[int, int],
        call_id_to_fc: dict[int, FunctionCall],
    ) -> _StdoutSignal | _FinalResult:
        """Handle a completed stdout read task. Returns a signal or the final result."""
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
            raise CodeRuntimeError(f'Malformed protocol message from driver: {raw[:200]!r}')

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
            t = asyncio.create_task(call_tool(fc))
            tool_tasks[cid] = t
            task_id_to_cid[id(t)] = cid
            return _StdoutSignal.NEW_CALL
        elif msg_type == 'calls_ready':
            return _StdoutSignal.CALLS_READY
        elif msg_type == 'complete':
            await process.kill()
            return _FinalResult(value=msg.get('result'))
        elif msg_type == 'error':
            await process.kill()
            error_type = msg.get('error_type', 'runtime')
            error_msg = msg.get('error', 'Unknown driver error')
            if error_type == 'syntax':
                raise CodeSyntaxError(error_msg)
            raise CodeRuntimeError(error_msg)

        return _StdoutSignal.CONTINUE

    @staticmethod
    async def _handle_tool_done(
        task: asyncio.Task[Any],
        process: DriverTransport,
        tool_tasks: dict[int, asyncio.Task[Any]],
        task_id_to_cid: dict[int, int],
        call_id_to_fc: dict[int, FunctionCall],
        completed_results: dict[int, Any],
        interrupted_calls: list[InterruptedToolCall],
        stdout_task: asyncio.Task[Any],
    ) -> None:
        """Handle a completed tool task: send result, accumulate interrupt, or propagate error."""
        cid = task_id_to_cid.pop(id(task))
        del tool_tasks[cid]

        try:
            result = task.result()
            completed_results[cid] = result
            # Serialize via tool_return_ta to preserve type fidelity (bytes, Pydantic models, etc.)
            # before embedding in the JSON protocol message.
            json_result = tool_return_ta.dump_python(result, mode='json')
            result_msg = json.dumps({'type': 'result', 'id': cid, 'result': json_result}) + '\n'
            await process.write_line(result_msg.encode())
        except (ApprovalRequired, CallDeferred) as e:
            fc = call_id_to_fc[cid]
            interrupted_calls.append(InterruptedToolCall(type=e, call=fc))
        except Exception as e:
            # Intentional broad catch: this is a defensive boundary between the runtime
            # protocol and the tool execution layer. Tool implementation bugs get wrapped
            # as CodeRuntimeError (→ ModelRetry) rather than crashing the runtime protocol.
            # The original exception message is preserved so the LLM sees what went wrong.
            await _cancel_all(tool_tasks, stdout_task, process)
            raise CodeRuntimeError(f'Tool execution error: {e}') from e


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel a task and suppress CancelledError."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _cancel_all(
    tool_tasks: dict[int, asyncio.Task[Any]],
    stdout_task: asyncio.Task[Any],
    process: DriverTransport,
) -> None:
    """Cancel all pending tasks and kill the driver process."""
    all_tasks = [*tool_tasks.values(), stdout_task]
    for t in all_tasks:
        t.cancel()
    for t in all_tasks:
        await _cancel_task(t)
    try:
        await process.kill()
    except Exception:
        pass
