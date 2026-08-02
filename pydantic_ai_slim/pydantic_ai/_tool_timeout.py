from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio

from .exceptions import ModelRetry

T = TypeVar('T')


async def run_with_tool_timeout(call: Callable[[], Awaitable[T]], timeout: float | None) -> T:
    """Run a tool call and turn only the configured deadline into a retry."""
    if timeout is None:
        return await call()

    scope: anyio.CancelScope | None = None
    try:
        with anyio.fail_after(timeout) as scope:
            return await call()
    except TimeoutError:
        if scope is None or not scope.cancel_called:
            raise
        raise ModelRetry(f'Timed out after {timeout} seconds.') from None
