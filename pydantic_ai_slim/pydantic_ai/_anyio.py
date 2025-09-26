from __future__ import annotations as _annotations

from collections.abc import Awaitable
from typing import TypeVar, cast

import anyio

T = TypeVar('T')


async def gather(*awaitables: Awaitable[T]) -> list[T]:
    """Run multiple awaitables concurrently using an AnyIO task group."""
    # We initialize the list, so we can insert the results in the correct order.
    results: list[T] = cast(list[T], [None] * len(awaitables))

    async def run_and_store(coro: Awaitable[T], index: int) -> None:
        results[index] = await coro

    async with anyio.create_task_group() as tg:
        for i, c in enumerate(awaitables):
            tg.start_soon(run_and_store, c, i)

    return results
