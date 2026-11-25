"""The Temporal handle await in `_activity_execution`, driven with plain asyncio futures.

It needs no Temporal server, so it lives outside `tests/durable_exec/temporal/`, whose fixtures skip on
Python 3.14+. That keeps it running on 3.14+, where `asyncio.shield` logs the error of a shielded future
that fails after the outer one was cancelled.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import anyio
import pytest

from pydantic_ai.exceptions import ModelRequestTimeout
from pydantic_ai.models._request_timeout import RequestDeadline

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.durable_exec.temporal._activity_execution import (
        _await_with_single_cancel,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='temporal not installed'),
    pytest.mark.parametrize('anyio_backend', ['asyncio']),
]


class _ActivityHandle(asyncio.Future[Any]):
    """Like Temporal's activity handle, cancelling it requests a cancel that later resolves it with an error."""

    def cancel(self, msg: Any | None = None) -> bool:
        self.get_loop().call_soon(self.set_exception, RuntimeError('Activity cancelled'))
        return False


@contextmanager
def _loop_errors() -> Generator[list[dict[str, Any]]]:
    """Collect what asyncio reports to the loop's exception handler, such as a never-retrieved future error."""
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    try:
        yield errors
        # A handle's never-retrieved error is reported when it is garbage collected.
        gc.collect()
    finally:
        loop.set_exception_handler(previous_handler)


async def _cancel_while_awaiting(handle: asyncio.Future[Any], *, fail_in_the_same_tick: bool = False) -> None:
    """Cancel a task awaiting `handle`.

    Run it in a task of its own when the test needs the handle garbage collected: the frames that hold
    the handle are then freed by the time that task is awaited.
    """
    task = asyncio.create_task(_await_with_single_cancel(handle))
    await asyncio.sleep(0)
    if fail_in_the_same_tick:
        handle.set_exception(RuntimeError('activity failed'))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handle.done()


async def test_cancelled_handle_error_is_not_reported(anyio_backend: str) -> None:
    """The error a handle resolves with after a cancel is retrieved, not left for asyncio to report (#13459)."""
    with _loop_errors() as errors:
        await asyncio.create_task(_cancel_while_awaiting(_ActivityHandle()))
    assert errors == []


async def test_cancel_in_the_tick_the_handle_fails(anyio_backend: str) -> None:
    """A cancel that lands in the same tick the handle fails re-raises the cancel and retrieves the error."""
    with _loop_errors() as errors:
        handle: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await asyncio.create_task(_cancel_while_awaiting(handle, fail_in_the_same_tick=True))
        del handle
    assert errors == []


async def test_handle_that_ends_cancelled(anyio_backend: str) -> None:
    handle: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    await _cancel_while_awaiting(handle)
    assert handle.cancelled()


class _CompletingActivityHandle(asyncio.Future[Any]):
    """Like an activity handle under `WAIT_CANCELLATION_COMPLETED` whose activity returns a result for the cancel."""

    def cancel(self, msg: Any | None = None) -> bool:
        self.get_loop().call_soon(self.set_result, 'partial')
        return False


async def test_result_of_activity_completed_after_cancel_is_handed_over(anyio_backend: str) -> None:
    """An activity that completes after the cancel hands its result over with the cancellation, which still raises."""
    received: list[tuple[asyncio.CancelledError, Any]] = []
    task = asyncio.create_task(
        _await_with_single_cancel(
            _CompletingActivityHandle(), on_cancelled_result=lambda error, result: received.append((error, result))
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [(type(error), result) for error, result in received] == [(asyncio.CancelledError, 'partial')]


def _passed_deadline() -> RequestDeadline:
    return RequestDeadline(model_name='model', timeout=0.0, deadline=anyio.current_time())


async def _time_out_while_awaiting(handle: asyncio.Future[Any]) -> None:
    """Await `handle` under a request deadline that has already passed, as a stream's model activity is."""
    with pytest.raises(ModelRequestTimeout):
        with _passed_deadline().enforce():
            await _await_with_single_cancel(handle, waits_on_request_deadline=False)
    # The handle resolves after the timeout was raised, and then has its error retrieved.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert handle.done()


@pytest.mark.parametrize('ends_cancelled', [False, True])
async def test_request_deadline_does_not_wait_for_the_cancelled_activity(
    anyio_backend: str, ends_cancelled: bool
) -> None:
    """A request deadline raises its timeout without waiting for the activity it cancelled to resolve.

    The timeout doesn't keep the activity's result, so it's not held up waiting for it, and the error the activity
    resolves with later isn't reported as never retrieved.
    """
    with _loop_errors() as errors:
        handle: asyncio.Future[Any] = (
            asyncio.get_running_loop().create_future() if ends_cancelled else _ActivityHandle()
        )
        await asyncio.create_task(_time_out_while_awaiting(handle))
        assert handle.cancelled() is ends_cancelled
        del handle
    assert errors == []


async def test_request_deadline_waits_for_the_cancelled_activity_by_default(anyio_backend: str) -> None:
    """Unless told otherwise, a request deadline waits for the activity it cancelled, as the configured cancellation
    type says, such as an explicit `WAIT_CANCELLATION_COMPLETED`."""
    handle = _ActivityHandle()
    with pytest.raises(ModelRequestTimeout):
        with _passed_deadline().enforce():
            await _await_with_single_cancel(handle)
    assert handle.done()
    with pytest.raises(RuntimeError, match='Activity cancelled'):
        handle.result()


async def test_result_and_error_pass_through(anyio_backend: str) -> None:
    loop = asyncio.get_running_loop()
    succeeded: asyncio.Future[int] = loop.create_future()
    succeeded.set_result(42)
    assert await _await_with_single_cancel(succeeded) == 42

    failed: asyncio.Future[int] = loop.create_future()
    failed.set_exception(ValueError('activity failed'))
    with pytest.raises(ValueError, match='activity failed'):
        await _await_with_single_cancel(failed)
