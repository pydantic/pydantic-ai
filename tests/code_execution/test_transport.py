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
    _StdoutSignal,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError

pytestmark = pytest.mark.anyio


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
        return frozenset({'run_code'})

    async def _copy_driver(self) -> None:
        pass

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        return _ErrorTransport()


async def test_run_catches_non_standard_exception():
    """Non-standard exceptions from transport are wrapped in CodeRuntimeError."""
    env = _ErrorEnvironment()
    with pytest.raises(CodeRuntimeError, match='Driver communication error.*bad read'):
        await env.run_python(
            'x = 1',
            AsyncMock(),
            functions={},
            referenced_types=[],
        )
