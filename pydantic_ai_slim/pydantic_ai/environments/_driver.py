"""Host-side ABC for driver-based execution environments.

Provides `DriverBasedEnvironment`, an intermediate abstract base class that
extends `ExecutionEnvironment` with code execution via the NDJSON driver protocol.
Concrete subclasses (Docker, Local) implement `_start_driver` and `_copy_driver`.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from pydantic_ai.toolsets.code_execution._abstract import (
    CodeExecutionTimeout,
    CodeRuntimeError,
    CodeSyntaxError,
    FunctionCall,
)

from ._base import ExecutionEnvironment

if TYPE_CHECKING:
    from pydantic_ai._python_signature import FunctionSignature, TypeSignature
    from pydantic_ai.toolsets.code_execution._abstract import FunctionCallback

    from ._base import ExecutionProcess


class DriverTransport(ABC):
    """Interface for communicating with a driver process.

    Concrete implementations wrap platform-specific transport types
    (`asyncio.subprocess.Process`, SDK handles, WebSocket connections, etc.).
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


class ExecutionProcessTransport(DriverTransport):
    """Adapts an `ExecutionProcess` to the `DriverTransport` interface.

    Provides line-buffered reads on top of the raw `recv()` interface,
    which is what the NDJSON protocol requires.
    """

    def __init__(self, process: ExecutionProcess) -> None:
        self._proc = process
        self._buffer = b''

    async def read_line(self) -> bytes:
        while b'\n' not in self._buffer:
            chunk = await self._proc.recv()
            if not chunk:
                remaining = self._buffer
                self._buffer = b''
                return remaining
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b'\n', 1)
        return line + b'\n'

    async def write_line(self, data: bytes) -> None:
        await self._proc.send(data)

    async def read_stderr(self) -> bytes:
        try:
            return await self._proc.recv_stderr(timeout=1.0)
        except Exception:
            return b''

    async def kill(self) -> None:
        await self._proc.kill()


class _ToolError(Exception):
    """Wrapper to distinguish tool execution errors from transport/protocol errors."""


class _StdoutSignal(Enum):
    """Typed signals from _handle_stdout indicating what happened."""

    CONTINUE = auto()


@dataclass(frozen=True)
class _FinalResult:
    """Wraps the final result value from a completed driver execution."""

    value: Any


class DriverBasedEnvironment(ExecutionEnvironment, ABC):
    """Environment with code execution via the NDJSON driver protocol.

    Extends `ExecutionEnvironment` with `run_python` that launches a
    driver script inside the environment and communicates via NDJSON over
    stdin/stdout. The driver handles code compilation, execution, and
    proxying of external function calls.

    Subclasses must implement `_copy_driver` (install the driver script into
    the environment). The default `_start_driver` uses `create_process`
    with an `ExecutionProcessTransport` adapter; override for custom transport.
    """

    execution_timeout: float | None = None
    """Optional timeout in seconds for code execution. None means no timeout."""

    driver_python_path: str = 'python'
    """Path to the Python interpreter inside the environment."""

    driver_script_path: str = '/tmp/pydantic_ai_driver.py'
    """Path where the driver script is installed inside the environment."""

    _driver_copied: bool = False

    # --- Driver protocol ---

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        """Launch the driver process and send the init message.

        The default implementation uses `create_process` with an
        `ExecutionProcessTransport` adapter. Override for custom transport
        (e.g. asyncio subprocess with the Docker CLI).

        Args:
            init_msg: The init message dict to send to the driver.

        Returns:
            A DriverTransport for communicating with the driver.
        """
        proc = await self.create_process(f'{self.driver_python_path} -u {self.driver_script_path}')
        await proc.__aenter__()
        transport = ExecutionProcessTransport(proc)
        init_line = json.dumps(init_msg).encode() + b'\n'
        await transport.write_line(init_line)
        return transport

    @abstractmethod
    async def _copy_driver(self) -> None:
        """Install the driver script into the environment.

        Called once before the first `run_python` invocation. Implementations
        should copy the driver script from the host to the environment
        (e.g. via `docker exec tee`, file API, or local file reference).
        """
        ...

    async def run_python_with_functions(
        self,
        code: str,
        *,
        function_callback: FunctionCallback,
        functions: dict[str, FunctionSignature] | None = None,
        referenced_types: list[TypeSignature] | None = None,
    ) -> Any:
        """Execute Python code with external functions via the NDJSON driver protocol."""
        if not self._driver_copied:  # pragma: no branch
            await self._copy_driver()
            self._driver_copied = True

        init_msg: dict[str, Any] = {
            'type': 'init',
            'code': code,
            'functions': list(functions) if functions else [],
        }
        process = await self._start_driver(init_msg)
        try:
            return await self._run_with_timeout(process, function_callback)
        except (CodeSyntaxError, CodeRuntimeError):
            raise
        except _ToolError as e:
            if e.__cause__ is None:  # pragma: no cover
                raise
            raise e.__cause__
        except Exception as e:
            raise CodeRuntimeError(f'Driver communication error: {e}') from e

    # --- Protocol implementation ---

    async def _run_with_timeout(self, process: DriverTransport, function_callback: FunctionCallback) -> Any:
        """Run the execution loop, applying `execution_timeout` if configured."""
        coro = self._execution_loop(process, function_callback)
        if self.execution_timeout is not None:
            try:
                return await asyncio.wait_for(coro, timeout=self.execution_timeout)
            except asyncio.TimeoutError:
                await process.kill()
                raise CodeExecutionTimeout(f'Code execution timed out after {self.execution_timeout} seconds')
        return await coro

    async def _execution_loop(self, process: DriverTransport, function_callback: FunctionCallback) -> Any:
        """Run the dual-wait event loop: read driver stdout + dispatch tool tasks."""
        tool_tasks: dict[int, asyncio.Task[Any]] = {}
        task_id_to_cid: dict[int, int] = {}

        stdout_task: asyncio.Task[bytes] = asyncio.create_task(process.read_line())

        try:
            while True:
                waitables: list[asyncio.Task[Any]] = [stdout_task, *tool_tasks.values()]
                done, _ = await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    if task is stdout_task:
                        result = await self._handle_stdout(task, process, function_callback, tool_tasks, task_id_to_cid)
                        if isinstance(result, _FinalResult):
                            return result.value
                        stdout_task = asyncio.create_task(process.read_line())
                    else:
                        try:
                            await self._handle_tool_done(
                                task,
                                process,
                                tool_tasks,
                                task_id_to_cid,
                            )
                        except Exception as e:
                            raise _ToolError() from e
        finally:
            await _cancel_all(tool_tasks, stdout_task, process)

    @staticmethod
    async def _handle_stdout(
        task: asyncio.Task[bytes],
        process: DriverTransport,
        function_callback: FunctionCallback,
        tool_tasks: dict[int, asyncio.Task[Any]],
        task_id_to_cid: dict[int, int],
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
            t = asyncio.ensure_future(function_callback(fc))
            tool_tasks[cid] = t
            task_id_to_cid[id(t)] = cid
            return _StdoutSignal.CONTINUE
        elif msg_type == 'calls_ready':
            return _StdoutSignal.CONTINUE
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
    ) -> None:
        """Handle a completed tool task: send result back to the driver."""
        cid = task_id_to_cid.pop(id(task))
        del tool_tasks[cid]

        result = task.result()
        result_msg = json.dumps({'type': 'result', 'id': cid, 'result': result}) + '\n'
        await process.write_line(result_msg.encode())


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
        await _cancel_task(t)
    try:
        await process.kill()
    except Exception:
        pass
