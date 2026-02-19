"""Tests for the driver transport protocol handler."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pydantic_ai.environments._driver import (
    DriverBasedEnvironment,
    DriverTransport,
    ExecutionProcessTransport,
    _StdoutSignal,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.toolsets.code_execution._abstract import (
    CodeExecutionTimeout,
    CodeRuntimeError,
    CodeSyntaxError,
)

pytestmark = pytest.mark.anyio


# --- Helpers ---


class _MockTransport(DriverTransport):
    """Mock transport implementing the public DriverTransport interface."""

    def __init__(self, *, stderr: bytes = b''):
        self._stderr = stderr

    async def read_line(self) -> bytes:
        return b''  # pragma: no cover

    async def write_line(self, data: bytes) -> None:
        pass

    async def read_stderr(self) -> bytes:
        return self._stderr

    async def kill(self) -> None:
        pass


async def _handle(data: bytes, *, stderr: bytes = b'') -> Any:
    """Create a resolved task and feed it to _handle_stdout."""

    async def _return() -> bytes:
        return data

    task: asyncio.Task[bytes] = asyncio.ensure_future(_return())
    await task
    return await DriverBasedEnvironment._handle_stdout(  # pyright: ignore[reportPrivateUsage]
        task, _MockTransport(stderr=stderr), AsyncMock(), {}, {}
    )


class _ScriptedTransport(DriverTransport):
    """Transport that feeds pre-programmed NDJSON messages.

    Messages are consumed in order from `lines`. When a message is a `call`,
    the transport waits for a result to be written back via `write_line`
    before yielding the next message.
    """

    def __init__(self, lines: list[dict[str, Any]], *, stderr: bytes = b''):
        self._lines = [json.dumps(m).encode() + b'\n' for m in lines]
        self._index = 0
        self._stderr = stderr
        self._written: list[bytes] = []
        self._killed = False
        self._result_event: asyncio.Event = asyncio.Event()
        # Pre-set so non-call messages proceed immediately
        self._result_event.set()

    async def read_line(self) -> bytes:
        # Wait for any pending result write before yielding the next line
        await self._result_event.wait()
        if self._index >= len(self._lines):
            return b''  # pragma: no cover
        line = self._lines[self._index]
        self._index += 1
        # If this was a call message, block until the host writes a result
        msg = json.loads(line)
        if msg.get('type') == 'call':
            self._result_event.clear()
        return line

    async def write_line(self, data: bytes) -> None:
        self._written.append(data)
        self._result_event.set()

    async def read_stderr(self) -> bytes:
        return self._stderr  # pragma: no cover

    async def kill(self) -> None:
        self._killed = True


class _ScriptedEnvironment(DriverBasedEnvironment):
    """Environment that uses a _ScriptedTransport."""

    def __init__(self, lines: list[dict[str, Any]], **kwargs: Any):
        self._transport = _ScriptedTransport(lines, **kwargs)

    @property
    def capabilities(self) -> frozenset[Any]:
        return frozenset({'run_python', 'run_python_with_functions'})  # pragma: no cover

    async def _copy_driver(self) -> None:
        pass

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        return self._transport


class _ErrorTransport(DriverTransport):
    """Transport that raises TypeError on read_line."""

    async def read_line(self) -> bytes:
        raise TypeError('bad read')

    async def write_line(self, data: bytes) -> None:
        pass  # pragma: no cover

    async def read_stderr(self) -> bytes:
        return b''  # pragma: no cover

    async def kill(self) -> None:
        pass


class _ErrorEnvironment(DriverBasedEnvironment):
    """Environment that uses _ErrorTransport."""

    @property
    def capabilities(self) -> frozenset[Any]:
        return frozenset({'run_python'})  # pragma: no cover

    async def _copy_driver(self) -> None:
        pass

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        return _ErrorTransport()


class _BlockingTransport(DriverTransport):
    """Transport whose read_line blocks forever (for timeout tests)."""

    async def read_line(self) -> bytes:
        await asyncio.sleep(999)
        return b''  # pragma: no cover

    async def write_line(self, data: bytes) -> None:
        pass  # pragma: no cover

    async def read_stderr(self) -> bytes:
        return b''  # pragma: no cover

    async def kill(self) -> None:
        pass


class _TimeoutEnvironment(DriverBasedEnvironment):
    """Environment with a very short execution timeout."""

    execution_timeout: float | None = 0.01

    @property
    def capabilities(self) -> frozenset[Any]:
        return frozenset({'run_python', 'run_python_with_functions'})  # pragma: no cover

    async def _copy_driver(self) -> None:
        pass

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        return _BlockingTransport()


# --- _handle_stdout unit tests ---


async def test_handle_stdout_eof_with_stderr():
    """EOF on stdout reads stderr for error message."""
    with pytest.raises(CodeRuntimeError, match='segfault'):
        await _handle(b'', stderr=b'segfault\n')


async def test_handle_stdout_eof_no_stderr():
    """EOF with no stderr gives default message."""
    with pytest.raises(CodeRuntimeError, match='exited unexpectedly'):
        await _handle(b'')


async def test_handle_stdout_malformed_json():
    """Non-JSON output raises CodeRuntimeError."""
    with pytest.raises(CodeRuntimeError, match='Malformed'):
        await _handle(b'not json\n')


async def test_handle_stdout_unknown_msg_type():
    """Unknown message type returns CONTINUE."""
    result = await _handle(json.dumps({'type': 'unknown'}).encode() + b'\n')
    assert result == _StdoutSignal.CONTINUE


# --- Full execution loop tests ---


async def test_complete_message_returns_result():
    """A 'complete' message returns the final result."""
    env = _ScriptedEnvironment([{'type': 'complete', 'result': 'hello'}])
    result = await env.run_python_with_functions(
        'x = 1',
        function_callback=AsyncMock(),
        functions={},
        referenced_types=[],
    )
    assert result == 'hello'
    assert env._transport._killed  # pyright: ignore[reportPrivateUsage]


async def test_error_message_raises_runtime_error():
    """An 'error' message raises CodeRuntimeError."""
    env = _ScriptedEnvironment([{'type': 'error', 'error': 'boom'}])
    with pytest.raises(CodeRuntimeError, match='boom'):
        await env.run_python_with_functions(
            'x = 1',
            function_callback=AsyncMock(),
            functions={},
            referenced_types=[],
        )


async def test_syntax_error_message_raises_syntax_error():
    """An 'error' message with error_type='syntax' raises CodeSyntaxError."""
    env = _ScriptedEnvironment([{'type': 'error', 'error_type': 'syntax', 'error': 'bad syntax'}])
    with pytest.raises(CodeSyntaxError, match='bad syntax'):
        await env.run_python_with_functions(
            'x = 1',
            function_callback=AsyncMock(),
            functions={},
            referenced_types=[],
        )


async def test_function_call_and_result():
    """A 'call' message dispatches a function callback, then sends the result back."""

    async def callback(call: Any) -> str:
        assert call.function_name == 'my_func'
        assert call.kwargs == {'x': 1}
        return 'callback_result'

    env = _ScriptedEnvironment(
        [
            {'type': 'call', 'id': 1, 'function': 'my_func', 'kwargs': {'x': 1}},
            {'type': 'complete', 'result': 'done'},
        ]
    )
    result = await env.run_python_with_functions(
        'my_func(x=1)',
        function_callback=callback,
        functions={},
        referenced_types=[],
    )
    assert result == 'done'
    # Verify a result was written back
    assert len(env._transport._written) > 0  # pyright: ignore[reportPrivateUsage]
    written_msg = json.loads(env._transport._written[0])  # pyright: ignore[reportPrivateUsage]
    assert written_msg == {'type': 'result', 'id': 1, 'result': 'callback_result'}


async def test_calls_ready_continues():
    """A 'calls_ready' message continues the loop."""
    env = _ScriptedEnvironment(
        [
            {'type': 'calls_ready'},
            {'type': 'complete', 'result': 42},
        ]
    )
    result = await env.run_python_with_functions(
        'x = 1',
        function_callback=AsyncMock(),
        functions={},
        referenced_types=[],
    )
    assert result == 42


async def test_execution_timeout():
    """Timeout raises CodeExecutionTimeout."""
    env = _TimeoutEnvironment()
    with pytest.raises(CodeExecutionTimeout, match='timed out'):
        await env.run_python_with_functions(
            'x = 1',
            function_callback=AsyncMock(),
            functions={},
            referenced_types=[],
        )


async def test_run_catches_non_standard_exception():
    """Non-standard exceptions from transport are wrapped in CodeRuntimeError."""
    env = _ErrorEnvironment()
    with pytest.raises(CodeRuntimeError, match='Driver communication error.*bad read'):
        await env.run_python_with_functions(
            'x = 1',
            function_callback=AsyncMock(),
            functions={},
            referenced_types=[],
        )


async def test_error_reraised_from_run_python_with_functions():
    """CodeRuntimeError and CodeSyntaxError are re-raised directly (not wrapped)."""
    env = _ScriptedEnvironment([{'type': 'error', 'error_type': 'syntax', 'error': 'parse fail'}])
    with pytest.raises(CodeSyntaxError, match='parse fail'):
        await env.run_python_with_functions(
            'x = 1',
            function_callback=AsyncMock(),
            functions={},
            referenced_types=[],
        )


async def test_tool_error_unwrapped():
    """When a tool callback raises, the cause is unwrapped from _ToolError."""

    async def bad_callback(call: Any) -> str:
        raise ValueError('tool exploded')

    env = _ScriptedEnvironment(
        [
            {'type': 'call', 'id': 1, 'function': 'exploder', 'kwargs': {}},
            {'type': 'complete', 'result': 'never reached'},
        ]
    )
    with pytest.raises(ValueError, match='tool exploded'):
        await env.run_python_with_functions(
            'exploder()',
            function_callback=bad_callback,
            functions={},
            referenced_types=[],
        )


# --- ExecutionProcessTransport tests ---


class _MockProcess:
    """Minimal mock of ExecutionProcess for testing ExecutionProcessTransport."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._sent: list[bytes] = []
        self._killed = False

    async def recv(self, timeout: float | None = None) -> bytes:
        if not self._chunks:
            return b''
        return self._chunks.pop(0)

    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        raise OSError('stderr error')

    async def send(self, data: bytes) -> None:
        self._sent.append(data)

    async def kill(self) -> None:
        self._killed = True


async def test_execution_process_transport_read_line():
    """read_line accumulates chunks until newline."""
    proc = _MockProcess([b'hel', b'lo\nwo', b'rld\n'])
    transport = ExecutionProcessTransport(proc)  # pyright: ignore[reportArgumentType]
    line = await transport.read_line()
    assert line == b'hello\n'
    # Second read should use the buffered data
    line2 = await transport.read_line()
    assert line2 == b'world\n'


async def test_execution_process_transport_read_line_eof():
    """read_line returns remaining buffer on EOF (empty recv)."""
    proc = _MockProcess([b'partial'])
    transport = ExecutionProcessTransport(proc)  # pyright: ignore[reportArgumentType]
    line = await transport.read_line()
    assert line == b'partial'


async def test_execution_process_transport_write_line():
    """write_line delegates to process.send."""
    proc = _MockProcess([])
    transport = ExecutionProcessTransport(proc)  # pyright: ignore[reportArgumentType]
    await transport.write_line(b'data\n')
    assert proc._sent == [b'data\n']  # pyright: ignore[reportPrivateUsage]


async def test_execution_process_transport_read_stderr():
    """read_stderr returns empty bytes when the process raises."""
    proc = _MockProcess([])
    transport = ExecutionProcessTransport(proc)  # pyright: ignore[reportArgumentType]
    result = await transport.read_stderr()
    assert result == b''


async def test_execution_process_transport_kill():
    """kill delegates to process.kill."""
    proc = _MockProcess([])
    transport = ExecutionProcessTransport(proc)  # pyright: ignore[reportArgumentType]
    await transport.kill()
    assert proc._killed  # pyright: ignore[reportPrivateUsage]


# --- Default _start_driver test ---


async def test_default_start_driver():
    """The default _start_driver creates a process and sends the init message."""
    sent_data: list[bytes] = []

    class _FakeProcess:
        async def __aenter__(self) -> _FakeProcess:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass  # pragma: no cover

        async def recv(self, timeout: float | None = None) -> bytes:
            return b''  # pragma: no cover

        async def send(self, data: bytes) -> None:
            sent_data.append(data)

        async def recv_stderr(self, timeout: float | None = None) -> bytes:
            return b''  # pragma: no cover

        async def kill(self) -> None:
            pass  # pragma: no cover

    class _FakeDriverEnvironment(DriverBasedEnvironment):
        @property
        def capabilities(self) -> frozenset[Any]:
            return frozenset({'run_python', 'run_python_with_functions'})  # pragma: no cover

        async def _copy_driver(self) -> None:
            pass

        async def create_process(self, command: str, **kwargs: Any) -> Any:
            return _FakeProcess()

    env = _FakeDriverEnvironment()
    init_msg: dict[str, Any] = {'type': 'init', 'code': 'x = 1', 'functions': []}
    transport = await env._start_driver(init_msg)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(transport, ExecutionProcessTransport)
    assert len(sent_data) == 1
    assert json.loads(sent_data[0]) == init_msg
