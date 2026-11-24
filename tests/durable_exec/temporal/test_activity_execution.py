"""The Temporal handle await in `_activity_execution`, driven with plain asyncio futures.

No Temporal server, so unlike the Temporal suite this module also runs on Python 3.14+, where
`asyncio.shield` logs the error of a shielded future that fails after the outer one was cancelled.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

import pytest

try:
    from pydantic_ai.durable_exec.temporal._activity_execution import (
        _await_with_single_cancel,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = pytest.mark.parametrize('anyio_backend', ['asyncio'])


class _ActivityHandle(asyncio.Future[Any]):
    """Like Temporal's activity handle, cancelling it requests a cancel that later resolves it with an error."""

    def cancel(self, msg: Any | None = None) -> bool:
        self.get_loop().call_soon(self.set_exception, RuntimeError('Activity cancelled'))
        return False


async def _cancel_while_awaiting(handle: asyncio.Future[Any]) -> None:
    task = asyncio.create_task(_await_with_single_cancel(handle))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handle.done()


async def test_cancelled_handle_error_is_not_logged(anyio_backend: str, caplog: pytest.LogCaptureFixture) -> None:
    """The handle's error after a cancel is retrieved, not left for asyncio to log (#13459)."""
    caplog.set_level(logging.ERROR)
    await _cancel_while_awaiting(_ActivityHandle())
    gc.collect()
    assert [record.getMessage() for record in caplog.records] == []


async def test_handle_that_ends_cancelled_still_reraises(anyio_backend: str) -> None:
    await _cancel_while_awaiting(asyncio.get_running_loop().create_future())


async def test_result_and_error_pass_through(anyio_backend: str) -> None:
    loop = asyncio.get_running_loop()
    succeeded: asyncio.Future[int] = loop.create_future()
    succeeded.set_result(42)
    assert await _await_with_single_cancel(succeeded) == 42

    failed: asyncio.Future[int] = loop.create_future()
    failed.set_exception(ValueError('activity failed'))
    with pytest.raises(ValueError, match='activity failed'):
        await _await_with_single_cancel(failed)
