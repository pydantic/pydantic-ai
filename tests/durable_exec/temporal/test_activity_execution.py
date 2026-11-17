from __future__ import annotations

import asyncio
from typing import Any

import pytest

try:
    from pydantic_ai.durable_exec.temporal._activity_execution import _await_with_single_cancel
except ImportError:  # pragma: no cover
    pytest.skip('temporal not installed', allow_module_level=True)


class _ExceptionTrackingFuture(asyncio.Future[Any]):
    def __init__(self) -> None:
        super().__init__()
        self.exception_calls = 0

    def exception(self) -> BaseException | None:
        self.exception_calls += 1
        return super().exception()


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_await_with_single_cancel_retrieves_done_handle_exception(
    monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
    handle = _ExceptionTrackingFuture()
    handle.set_exception(RuntimeError('activity failed'))

    async def cancelled_shield(_: asyncio.Future[Any]) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, 'shield', cancelled_shield)

    with pytest.raises(asyncio.CancelledError):
        await _await_with_single_cancel(handle)

    assert handle.exception_calls == 1
