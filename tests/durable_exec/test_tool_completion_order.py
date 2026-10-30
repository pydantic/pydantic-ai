from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from typing import TypeVar

import pytest

from pydantic_ai import Agent, EndStrategy, FunctionToolResultEvent
from pydantic_ai._run_context import EventStreamBuffer
from pydantic_ai._tool_execution import (
    _iter_completed_or_buffered,  # pyright: ignore[reportPrivateUsage]
    set_tool_task_wait,
)
from pydantic_ai._utils import aclose_if_supported
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models.test import TestModel

from ..conftest import try_import

with try_import() as imports_successful:
    from temporalio import workflow

pytestmark = pytest.mark.skipif(not imports_successful(), reason='temporal not installed')

T = TypeVar('T')


class _ReverseCompletionSet(set[asyncio.Task[T]]):
    """Make the identity-dependent set order reproducible without relying on an allocator."""

    def __iter__(self) -> Iterator[asyncio.Task[T]]:
        return iter(sorted(super().__iter__(), key=lambda task: task.get_name(), reverse=True))


@pytest.fixture
def reverse_completion_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    original_wait = asyncio.wait

    async def wait(
        fs: Iterable[asyncio.Task[T]],
        *,
        timeout: float | None = None,
        return_when: str = asyncio.ALL_COMPLETED,
    ) -> tuple[set[asyncio.Task[T]], set[asyncio.Task[T]]]:
        done, pending = await original_wait(fs, timeout=timeout, return_when=return_when)
        return _ReverseCompletionSet(done), pending

    monkeypatch.setattr(asyncio, 'wait', wait)


@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
async def test_durable_wait_preserves_simultaneous_tool_result_order(
    reverse_completion_sets: None, end_strategy: EndStrategy
) -> None:
    """Completion-set identity order must not determine workflow-visible result events.

    This exercises a real agent; provider recordings cannot control simultaneous task completion.
    The ordinary-run control demonstrates the exact ordering the engine override must avoid.
    """
    agent = Agent(TestModel(custom_output_text='done'), end_strategy=end_strategy)

    @agent.tool_plain
    async def first() -> str:
        return 'first'

    @agent.tool_plain
    async def second() -> str:
        return 'second'

    async def results() -> list[str]:
        async with agent.run_stream_events('go') as events:
            return [event.part.tool_name async for event in events if isinstance(event, FunctionToolResultEvent)]

    assert await results() == ['second', 'first']
    with set_tool_task_wait(workflow.wait):
        assert await results() == ['first', 'second']
    # An engine's override is scoped to the run, including after it exits.
    assert await results() == ['second', 'first']


@pytest.mark.parametrize('signalling_buffer', [False, True])
async def test_durable_wait_preserves_order_with_both_buffer_shapes(
    reverse_completion_sets: None, signalling_buffer: bool
) -> None:
    """Persistence can restore a plain list; both shapes must retain deterministic task order."""

    async def result(value: int) -> int:
        return value

    tasks = [asyncio.create_task(result(value), name=str(value)) for value in range(3)]
    buffer: list[AgentStreamEvent] = EventStreamBuffer() if signalling_buffer else []
    with set_tool_task_wait(workflow.wait):
        items = [item async for item in _iter_completed_or_buffered(tasks, buffer)]
    assert [item.result() for item in items if isinstance(item, asyncio.Task)] == [0, 1, 2]


@pytest.mark.parametrize('signalling_buffer', [False, True])
async def test_durable_wait_yields_earlier_completion_immediately(signalling_buffer: bool) -> None:
    """Deterministic ties must not make an earlier tool wait for its preceding sibling."""
    release_first = asyncio.Event()

    async def first() -> str:
        await release_first.wait()
        return 'first'

    async def second() -> str:
        return 'second'

    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    buffer: list[AgentStreamEvent] = EventStreamBuffer() if signalling_buffer else []
    with set_tool_task_wait(workflow.wait):
        completions = _iter_completed_or_buffered(tasks, buffer)
        try:
            async with asyncio.timeout(5):
                assert await anext(completions) is tasks[1]
                release_first.set()
                assert await anext(completions) is tasks[0]
        finally:
            release_first.set()
            await aclose_if_supported(completions)
            await asyncio.gather(*tasks)
