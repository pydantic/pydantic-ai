from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from threading import Barrier, Event
from typing import Literal

import pytest
from anyio.to_thread import current_default_thread_limiter

import pydantic_ai._utils as utils_module
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.mark.parametrize(
    'executor_kind',
    [
        pytest.param('anyio', id='anyio-worker'),
        pytest.param('custom', id='custom-executor'),
    ],
)
@pytest.mark.parametrize('entrypoint', ['sync', 'async'])
def test_nested_run_sync_uses_originating_event_loop(
    executor_kind: Literal['anyio', 'custom'], entrypoint: Literal['sync', 'async']
) -> None:
    """Exercise the public API because a unit test cannot reproduce the callback's cross-thread loop boundary."""
    event = asyncio.Event()
    loops: list[asyncio.AbstractEventLoop] = []
    originating_loop: asyncio.AbstractEventLoop | None = None

    def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('wait_event', {})])
        return ModelResponse(parts=[TextPart('done')])

    inner = Agent(FunctionModel(inner_model), output_type=str)

    @inner.tool_plain
    async def wait_event() -> str:
        loops.append(asyncio.get_running_loop())
        assert originating_loop is not None
        originating_loop.call_soon_threadsafe(event.set)
        await event.wait()
        return 'set'

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal originating_loop
        loop = asyncio.get_running_loop()
        originating_loop = loop
        loops.append(loop)

        waiter = asyncio.create_task(event.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with suppress(asyncio.CancelledError):
            await waiter

        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    if executor_kind == 'custom':
        with ThreadPoolExecutor(max_workers=1) as executor, Agent.using_thread_executor(executor):
            if entrypoint == 'sync':
                result = outer.run_sync('go')
            else:
                result = asyncio.run(outer.run('go'))
    else:
        if entrypoint == 'sync':
            result = outer.run_sync('go')
        else:
            result = asyncio.run(outer.run('go'))

    assert result.output == 'done'
    assert len(loops) == 2
    assert loops[0] is loops[1]


def test_nested_run_sync_from_sync_tool_uses_originating_event_loop() -> None:
    """Sync tools use the same worker boundary as sync output functions, so they need the same bridge."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        return ModelResponse(parts=[TextPart('inner')])

    inner = Agent(FunctionModel(inner_model))

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('call_inner', {'prompt': 'x'})])
        return ModelResponse(parts=[TextPart('done')])

    outer = Agent(FunctionModel(outer_model))

    @outer.tool_plain
    def call_inner(prompt: str) -> str:
        return inner.run_sync(prompt).output

    result = outer.run_sync('go')

    assert result.output == 'done'
    assert len(loops) == 3
    assert all(loop is loops[0] for loop in loops)


async def test_nested_run_sync_reenters_saturated_worker_pool() -> None:
    """Inner sync callbacks must reuse occupied workers instead of waiting for another pool token."""
    limiter = current_default_thread_limiter()
    previous_total_tokens = limiter.total_tokens
    limiter.total_tokens = 2
    barrier = Barrier(2)

    def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        barrier.wait()
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    try:
        first, second = await asyncio.gather(outer.run('first'), outer.run('second'))
    finally:
        limiter.total_tokens = previous_total_tokens

    assert (first.output, second.output) == ('done', 'done')


def test_recursive_nested_run_sync_uses_one_event_loop() -> None:
    """Each nested sync callback must return to the original loop instead of creating another worker loop."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        return ModelResponse(parts=[TextPart('done')])

    inner = Agent(FunctionModel(inner_model))

    def middle_output(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def middle_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'inner'})])

    middle = Agent(FunctionModel(middle_model), output_type=[middle_output])

    def outer_output(ctx: RunContext[object], instructions: str) -> str:
        return middle.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'middle'})])

    outer = Agent(FunctionModel(outer_model), output_type=[outer_output])

    result = outer.run_sync('go')

    assert result.output == 'done'
    assert len(loops) == 3
    assert all(loop is loops[0] for loop in loops)


def test_nested_run_sync_preserves_worker_context() -> None:
    """The inner coroutine must inherit context changes made by the synchronous callback that invokes it."""
    marker = contextvars.ContextVar('marker', default='outer')

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(marker.get())])

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        marker.set('worker')
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    assert outer.run_sync('go').output == 'worker'


def test_nested_run_sync_propagates_exception() -> None:
    """Exceptions from the inner coroutine must reach the synchronous caller unchanged."""
    error = RuntimeError('inner failed')

    def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise error

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    with pytest.raises(RuntimeError, match='inner failed') as exc_info:
        outer.run_sync('go')

    assert exc_info.value is error


def test_nested_run_sync_cancels_queued_worker_request() -> None:
    """A cancelled inner callback must be skipped if its parent worker has not started it yet."""
    first_started = Event()
    release_first = Event()
    second_ran = False

    def first_callback() -> None:
        first_started.set()
        assert release_first.wait(5)

    def second_callback() -> None:
        nonlocal second_ran
        second_ran = True

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        first_task = asyncio.create_task(utils_module.run_in_executor(first_callback))
        while not first_started.is_set():
            await asyncio.sleep(0)

        second_task = asyncio.create_task(utils_module.run_in_executor(second_callback))
        await asyncio.sleep(0)
        second_task.cancel()
        await asyncio.sleep(0)
        release_first.set()

        await first_task
        with suppress(asyncio.CancelledError):
            await second_task
        return ModelResponse(parts=[TextPart('done')])

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    assert outer.run_sync('go').output == 'done'
    assert not second_ran


def test_nested_run_sync_preserves_asyncio_cancellation() -> None:
    """Cross-thread future adaptation must retain `asyncio.CancelledError`, not expose its concurrent variant."""

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    with pytest.raises(asyncio.CancelledError):
        outer.run_sync('go')


def test_async_output_function_nested_run_uses_one_event_loop() -> None:
    """Pin the existing async delegation path as the single-loop control for the sync bridge."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        return ModelResponse(parts=[TextPart('done')])

    inner = Agent(FunctionModel(inner_model))

    async def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return (await inner.run(instructions)).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loops.append(asyncio.get_running_loop())
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    assert outer.run_sync('go').output == 'done'
    assert len(loops) == 2
    assert loops[0] is loops[1]


def test_run_sync_from_nested_async_code_still_requires_await() -> None:
    """The worker bridge must not bypass the active-loop error inside async code."""
    deepest = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])))

    async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return deepest.run_sync('deepest').response

    inner = Agent(FunctionModel(inner_model))

    def output_fn(ctx: RunContext[object], instructions: str) -> str:
        return inner.run_sync(instructions).output

    async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': 'x'})])

    outer = Agent(FunctionModel(outer_model), output_type=[output_fn])

    with pytest.raises(RuntimeError, match='This event loop is already running'):
        outer.run_sync('go')


def test_concurrent_nested_run_sync_keeps_originating_loops_isolated() -> None:
    """Concurrent top-level sync runs must bridge to their own loops rather than another caller's loop."""

    def run_nested(prompt: str) -> tuple[str, list[asyncio.AbstractEventLoop]]:
        loops: list[asyncio.AbstractEventLoop] = []

        async def inner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            loops.append(asyncio.get_running_loop())
            return ModelResponse(parts=[TextPart(prompt)])

        inner = Agent(FunctionModel(inner_model))

        def output_fn(ctx: RunContext[object], instructions: str) -> str:
            return inner.run_sync(instructions).output

        async def outer_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            loops.append(asyncio.get_running_loop())
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'instructions': prompt})])

        outer = Agent(FunctionModel(outer_model), output_type=[output_fn])
        try:
            return outer.run_sync(prompt).output, loops
        finally:
            asyncio.get_event_loop().close()
            asyncio.set_event_loop(None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(run_nested, 'first')
        second_future = executor.submit(run_nested, 'second')
        first_result, first_loops = first_future.result()
        second_result, second_loops = second_future.result()

    assert (first_result, second_result) == ('first', 'second')
    assert first_loops[0] is first_loops[1]
    assert second_loops[0] is second_loops[1]
    assert first_loops[0] is not second_loops[0]
