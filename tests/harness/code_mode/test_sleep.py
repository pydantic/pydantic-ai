"""Sandbox sleeps: waited for by the executor, charged to the `max_duration_secs` allowance.

The executor tests swap in a recording sleep, so they check what was waited for and in what order
without depending on how long anything takes.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import pytest
from pydantic_monty import AsyncMonty

from pydantic_ai_harness._monty_exec import MontyExecutor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def _run(
    code: str, *, max_sleep_secs: float | None = None, global_sequential: bool = False, starts_before_waking: int = 1
) -> tuple[Any, list[str]]:
    """Run `code` through the executor with a sleep that records when each wait starts and ends.

    A wait ends only once `starts_before_waking` sleeps have started, so sleeps that overlap both
    start before either ends, while sleeps that were run one at a time never finish.
    """
    events: list[str] = []
    started = 0
    all_started = asyncio.Event()

    async def sleep(secs: float) -> None:
        nonlocal started
        events.append(f'start {secs:g}')
        started += 1
        if started >= starts_before_waking:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=30)  # hang guard, not a timing assertion
        events.append(f'end {secs:g}')

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        raise AssertionError('no tools in these snippets')  # pragma: no cover

    async with AsyncMonty() as pool:
        async with pool.checkout(os_policy={'sleep': 'call_host'}) as session:
            executor = MontyExecutor(
                dispatch=dispatch,
                valid_names=set[str](),
                global_sequential=global_sequential,
                max_sleep_secs=max_sleep_secs,
                sleep=sleep,
            )
            completed = await executor.run(partial(session.feed_start, code))
    return completed.output, events


async def test_sleeps_are_waited_for_in_order() -> None:
    code = 'import asyncio, time\ntime.sleep(2)\nr = await asyncio.sleep(3, result="awake")\nr'
    assert await _run(code) == ('awake', ['start 2', 'end 2', 'start 3', 'end 3'])


_GATHER = 'import asyncio\nawait asyncio.gather(asyncio.sleep(1), asyncio.sleep(2))\n"awake"'


def _kinds(events: list[str]) -> list[str]:
    return [event.split()[0] for event in events]


async def test_gathered_sleeps_overlap() -> None:
    output, events = await _run(_GATHER, starts_before_waking=2)
    assert output == 'awake'
    assert sorted(events) == ['end 1', 'end 2', 'start 1', 'start 2']
    assert _kinds(events) == ['start', 'start', 'end', 'end']


async def test_gathered_sleeps_take_turns_when_calls_are_sequential() -> None:
    # Monty does not promise which pending call is resolved first, only that they take turns here.
    output, events = await _run(_GATHER, global_sequential=True)
    assert output == 'awake'
    assert sorted(events) == ['end 1', 'end 2', 'start 1', 'start 2']
    assert _kinds(events) == ['start', 'end', 'start', 'end']


@pytest.mark.parametrize(
    'second_sleep',
    [pytest.param('time.sleep(3)', id='time.sleep'), pytest.param('await asyncio.sleep(3)', id='asyncio.sleep')],
)
async def test_sleep_past_the_allowance_raises_without_waiting(second_sleep: str) -> None:
    code = (
        f'import asyncio, time\ntime.sleep(3)\ntry:\n    {second_sleep}\nexcept TimeoutError as e:\n    r = str(e)\nr'
    )
    output, events = await _run(code, max_sleep_secs=5)
    assert output == 'sleeping 3s would exceed the 5s this code may sleep (max_duration_secs); 2s left'
    assert events == ['start 3', 'end 3']


async def test_no_allowance_means_no_cap() -> None:
    assert await _run('import time\ntime.sleep(86400)\n"awake"') == ('awake', ['start 86400', 'end 86400'])


@pytest.mark.parametrize(
    ('code', 'sleeps'),
    [
        pytest.param('import time\ntime.sleep(3600)', 1, id='time.sleep'),
        pytest.param(
            'import asyncio\nawait asyncio.gather(asyncio.sleep(3600), asyncio.sleep(3600))', 2, id='asyncio.sleep'
        ),
    ],
)
async def test_cancelling_the_run_interrupts_a_sleep(code: str, sleeps: int) -> None:
    started = 0
    all_started = asyncio.Event()
    interrupted: list[float] = []

    async def sleep(secs: float) -> None:
        nonlocal started
        started += 1
        if started == sleeps:
            all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            interrupted.append(secs)
            raise

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        raise AssertionError('no tools in these snippets')  # pragma: no cover

    async with AsyncMonty() as pool:
        async with pool.checkout(os_policy={'sleep': 'call_host'}) as session:
            executor = MontyExecutor(dispatch=dispatch, valid_names=set[str](), sleep=sleep)
            run = asyncio.ensure_future(executor.run(partial(session.feed_start, code)))
            await asyncio.wait_for(all_started.wait(), timeout=30)  # hang guard, not a timing assertion
            run.cancel()
            await asyncio.wait([run])
    assert run.cancelled()
    assert interrupted == [3600] * sleeps
