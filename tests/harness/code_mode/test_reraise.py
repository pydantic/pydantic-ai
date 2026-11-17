"""`MontyExecutor.reraise` errors end the feed instead of reaching the sandbox."""

from __future__ import annotations

import asyncio
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


async def test_reraise_error_does_not_replace_cancellation() -> None:
    """A never-awaited call's error must not turn a cancelled feed into a failed one."""
    waiting = asyncio.Event()

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        if name == 'lookup':
            raise _Diverged
        waiting.set()
        await asyncio.Event().wait()

    async with AsyncMonty() as pool:
        async with pool.checkout() as session:
            executor = MontyExecutor(dispatch=dispatch, valid_names={'lookup', 'wait'}, reraise=(_Diverged,))
            task = asyncio.ensure_future(executor.run(partial(session.feed_start, 'future = lookup()\nawait wait()')))
            await asyncio.wait_for(waiting.wait(), timeout=10)  # hang guard, not a timing assertion
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


async def test_reraise_error_replaces_an_ordinary_failure() -> None:
    """A diverged replay outranks the snippet's own error, which would otherwise become a model retry."""

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        raise _Diverged

    async with AsyncMonty() as pool:
        async with pool.checkout() as session:
            executor = MontyExecutor(dispatch=dispatch, valid_names={'lookup'}, reraise=(_Diverged,))
            with pytest.raises(_Diverged):
                await executor.run(partial(session.feed_start, "future = lookup()\nraise ValueError('snippet bug')"))
