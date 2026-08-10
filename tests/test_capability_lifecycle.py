"""Definitory tests for the private capability lifecycle sequencing engine.

Provider recordings cannot precisely assert internal ordering, replacement-error
propagation, or stream ownership, so these contracts require focused unit tests.
Public capability behavior remains covered by the integration test suite.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable

import pytest
from inline_snapshot import snapshot
from typing_extensions import Self

from pydantic_ai.capabilities._lifecycle import SKIP_LIFECYCLE_ENTRY, LifecycleStack, SkipLifecycleEntry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


async def test_lifecycle_stack_ordering_and_recovery() -> None:
    stack = LifecycleStack(['outer', 'inner'])
    events: list[str] = []

    async def transform(entry: str, value: str) -> str:
        events.append(f'transform:{entry}')
        return f'{value}:{entry}'

    assert await stack.forward('start', transform) == 'start:outer:inner'
    assert await stack.reverse('start', transform) == 'start:inner:outer'

    async def notify(entry: str) -> None:
        events.append(f'notify:{entry}')

    await stack.notify(notify)

    async def handler(value: str) -> str:
        events.append(f'handler:{value}')
        return 'result'

    def wrap(entry: str, inner: Callable[[str], Awaitable[str]]) -> Callable[[str], Awaitable[str]]:
        async def wrapped(value: str) -> str:
            events.append(f'enter:{entry}')
            result = await inner(value)
            events.append(f'exit:{entry}')
            return result

        return wrapped

    assert await stack.wrap(handler, wrap)('value') == 'result'

    async def recover(entry: str, error: Exception) -> str:
        events.append(f'recover:{entry}:{error}')
        if entry == 'inner':
            raise RuntimeError('replacement')
        return 'recovered'

    assert await stack.recover(ValueError('original'), recover, Exception) == 'recovered'
    assert events == snapshot(
        [
            'transform:outer',
            'transform:inner',
            'transform:inner',
            'transform:outer',
            'notify:outer',
            'notify:inner',
            'enter:outer',
            'enter:inner',
            'handler:value',
            'exit:inner',
            'exit:outer',
            'recover:inner:original',
            'recover:outer:replacement',
        ]
    )


async def test_lifecycle_stack_recovery_can_skip_entries_and_run_forward() -> None:
    stack = LifecycleStack(['first', 'skipped', 'last'])
    errors: list[str] = []

    async def recover(entry: str, error: Exception) -> str | SkipLifecycleEntry:
        errors.append(f'{entry}:{error}')
        if entry == 'skipped':
            return SKIP_LIFECYCLE_ENTRY
        if entry == 'first':
            raise RuntimeError('replacement')
        return 'recovered'

    assert await stack.recover(ValueError('original'), recover, Exception, reverse=False) == 'recovered'
    assert errors == ['first:original', 'skipped:replacement', 'last:replacement']


async def test_lifecycle_stack_all_skipped_recovery_preserves_error_traceback() -> None:
    stack = LifecycleStack(['first', 'second', 'third'])
    error = ValueError('original')

    async def skip(_entry: str, _error: Exception) -> SkipLifecycleEntry:
        return SKIP_LIFECYCLE_ENTRY

    with pytest.raises(ValueError) as exc_info:
        await stack.recover(error, skip, Exception)

    assert exc_info.value is error
    assert [entry.name for entry in exc_info.traceback] == [
        'test_lifecycle_stack_all_skipped_recovery_preserves_error_traceback',
        'recover',
    ]


class _TrackingStream(AsyncIterator[int]):
    def __init__(
        self,
        name: str,
        closed: list[str],
        source: AsyncIterable[int] | None = None,
    ) -> None:
        self.name = name
        self.closed = closed
        self.source = source.__aiter__() if source is not None else None
        self.yielded = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> int:
        if self.source is not None:
            return await self.source.__anext__()
        if self.yielded:
            raise StopAsyncIteration
        self.yielded = True
        return 1

    async def aclose(self) -> None:
        self.closed.append(self.name)


async def test_lifecycle_stack_stream_skips_entries_and_closes_every_layer() -> None:
    closed: list[str] = []
    source = _TrackingStream('source', closed)
    stack = LifecycleStack(['outer', 'skipped', 'inner'])

    def wrap(entry: str, stream: AsyncIterable[int]) -> AsyncIterable[int] | None:
        if entry == 'skipped':
            return None
        return _TrackingStream(entry, closed, stream)

    async with stack.stream(source, wrap) as stream:
        assert [value async for value in stream] == [1]

    assert closed == ['outer', 'inner', 'source']


async def test_lifecycle_stack_deferred_settlement_passes_only_unresolved_calls() -> None:
    requests = DeferredToolRequests(
        approvals=[ToolCallPart('approve', {}, tool_call_id='approval')],
        calls=[ToolCallPart('external', {}, tool_call_id='call')],
    )
    seen: list[tuple[str, list[str], list[str]]] = []
    stack = LifecycleStack(['approval-handler', 'call-handler'])

    async def handle(entry: str, remaining: DeferredToolRequests) -> DeferredToolResults:
        seen.append(
            (
                entry,
                [call.tool_call_id for call in remaining.approvals],
                [call.tool_call_id for call in remaining.calls],
            )
        )
        if entry == 'approval-handler':
            return DeferredToolResults(approvals={'approval': True})
        return DeferredToolResults(calls={'call': 'result'})

    assert await stack.settle_deferred(requests, handle) == DeferredToolResults(
        approvals={'approval': True}, calls={'call': 'result'}
    )
    assert seen == [
        ('approval-handler', ['approval'], ['call']),
        ('call-handler', [], ['call']),
    ]
