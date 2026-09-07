from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.direct import StreamedResponseSync, model_request_stream_sync
from pydantic_ai.messages import ModelMessage, ModelRequest, PartDeltaEvent
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.mark.parametrize('surface', ['agent', 'direct'])
@pytest.mark.parametrize('exit_mode', ['complete', 'break', 'caller_error', 'source_error'])
def test_sync_stream_owns_children_until_cleanup(
    surface: Literal['agent', 'direct'], exit_mode: Literal['complete', 'break', 'caller_error', 'source_error']
) -> None:
    """Task identities and cleanup ordering require an in-process model rather than recorded HTTP traffic."""
    owners: list[anyio.TaskInfo] = []
    children: dict[int, anyio.TaskInfo] = {}
    tasks: list[asyncio.Task[object]] = []
    cleaned = False

    async def source(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal cleaned
        owners.append(anyio.get_current_task())
        try:
            yield 'first '
            for _ in range(10):
                current = anyio.get_current_task()
                children[current.id] = current
                task = asyncio.current_task()
                assert task is not None
                tasks.append(task)
                yield 'next '
            if exit_mode == 'source_error':
                raise ValueError('source failed')
        finally:
            await anyio.sleep(0)
            cleaned = True

    model = FunctionModel(stream_function=source)
    error = pytest.raises(ValueError, match='failed') if exit_mode.endswith('error') else nullcontext()
    with error:
        with (
            Agent(model).run_stream_sync('Hello')
            if surface == 'agent'
            else model_request_stream_sync(model, [ModelRequest.user_text_prompt('Hello')])
        ) as result:
            stream = iter(result) if isinstance(result, StreamedResponseSync) else result.stream_text(debounce_by=None)
            for _ in stream:
                if children and exit_mode == 'break':
                    break
                if children and exit_mode == 'caller_error':
                    raise ValueError('caller failed')

    assert cleaned
    assert children
    assert all(child.parent_id == owners[0].id for child in children.values())
    assert all(task.done() for task in tasks)


@pytest.mark.parametrize('wait_for_owner', [False, True])
def test_sync_stream_owner_cancellation_drains_active_source(wait_for_owner: bool) -> None:
    owner_tasks: list[asyncio.Task[object]] = []
    pump_tasks: list[asyncio.Task[object]] = []
    ready = asyncio.Event()
    cleanup_finished = False
    cleanup_before_exit: list[tuple[bool, bool]] = []

    async def source(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal cleanup_finished
        owner = asyncio.current_task()
        assert owner is not None
        owner_tasks.append(owner)
        yield 'first '
        pump = asyncio.current_task()
        assert pump is not None
        pump_tasks.append(pump)
        try:
            yield 'second '
            ready.set()
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0)
                cleanup_finished = True

    with pytest.raises(asyncio.CancelledError):
        with model_request_stream_sync(
            FunctionModel(stream_function=source), [ModelRequest.user_text_prompt('Hello')]
        ) as result:
            stream = iter(result)
            event = next(stream)
            while not isinstance(event, PartDeltaEvent):
                event = next(stream)
            loop = owner_tasks[0].get_loop()
            watchdog = loop.call_later(10, loop.stop)
            try:
                loop.run_until_complete(ready.wait())
                owner_tasks[0].cancel()
                if wait_for_owner:
                    with pytest.raises(asyncio.CancelledError):
                        loop.run_until_complete(owner_tasks[0])
                with pytest.raises(RuntimeError, match=r'already closed|task group is not active'):
                    result.response
                with pytest.raises(RuntimeError, match='already closed'):
                    next(iter(result))
                cleanup_before_exit.append((cleanup_finished, all(task.done() for task in pump_tasks)))
            finally:
                watchdog.cancel()

    assert cleanup_before_exit == [(True, True)]
