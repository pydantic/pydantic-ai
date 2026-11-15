"""`MontyExecutor.reraise` errors end the feed instead of reaching the sandbox (#12539)."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from pydantic_monty import AsyncMonty

from pydantic_ai_harness._monty_exec import MontyExecutor


class _Diverged(Exception):
    """Stands in for an engine's replay-divergence error, such as DBOS's `DBOSUnexpectedStepError`."""


@pytest.mark.parametrize(
    'code,sequential',
    [
        ("try:\n    await lookup()\nexcept Exception:\n    'caught'", False),
        ("import asyncio\ntry:\n    await asyncio.gather(lookup())\nexcept Exception:\n    'caught'", False),
        ("try:\n    lookup()\nexcept Exception:\n    'caught'", True),
        ("future = lookup()\n'ignored'", False),
    ],
    ids=['awaited', 'gathered', 'sequential', 'never-awaited'],
)
async def test_reraise_error_ends_the_feed(code: str, sequential: bool) -> None:
    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        raise _Diverged

    async with AsyncMonty() as pool:
        async with pool.checkout() as session:
            executor = MontyExecutor(
                dispatch=dispatch,
                valid_names={'lookup'},
                sequential_names={'lookup'} if sequential else set[str](),
                reraise=(_Diverged,),
            )
            with pytest.raises(_Diverged):
                await executor.run(partial(session.feed_start, code))
