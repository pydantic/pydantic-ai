from __future__ import annotations as _annotations

import sys
from collections.abc import Awaitable
from typing import TypeVar, cast

import anyio

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup

T = TypeVar('T')


async def gather(*awaitables: Awaitable[T]) -> list[T]:
    """Run multiple awaitables concurrently using an AnyIO task group.

    Unlike `asyncio.gather`, AnyIO task groups wrap exceptions in `ExceptionGroup`.
    To preserve `asyncio.gather`-like behavior, if the group contains a single
    exception, it is unwrapped and re-raised directly.
    """
    # We initialize the list, so we can insert the results in the correct order.
    results: list[T] = cast(list[T], [None] * len(awaitables))

    async def run_and_store(coro: Awaitable[T], index: int) -> None:
        results[index] = await coro

    try:
        async with anyio.create_task_group() as tg:
            for i, c in enumerate(awaitables):
                tg.start_soon(run_and_store, c, i)
    except BaseExceptionGroup as eg:
        if len(eg.exceptions) == 1:
            raise eg.exceptions[0] from None
        raise

    return results
