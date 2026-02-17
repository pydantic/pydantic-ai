"""Tests for the driver transport protocol handler."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError
from pydantic_ai.toolsets.code_execution._transport import (
    DriverBasedRuntime,
    DriverTransport,
    _StdoutSignal,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.anyio


class _MockTransport(DriverTransport):
    """Mock transport implementing the public DriverTransport interface."""

    def __init__(self, *, stderr: bytes = b''):
        self._stderr = stderr

    async def read_line(self) -> bytes:
        return b''

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
    return await DriverBasedRuntime._handle_stdout(  # pyright: ignore[reportPrivateUsage]
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
