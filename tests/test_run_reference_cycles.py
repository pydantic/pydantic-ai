"""Run state should be released by reference counting once the application drops it.

In-process models isolate object lifetime; provider recordings add no ownership coverage.
"""

from __future__ import annotations

import asyncio
import datetime
import gc
import weakref
from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass
from types import AsyncGeneratorType
from typing import Any, Literal

import anyio
import pytest

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai._utils import ScopedTask
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsNow, IsStr

READINESS_WAIT_TIMEOUT = 5


@dataclass
class Deps:
    pass


@pytest.fixture
def no_gc() -> Generator[None]:
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


async def three_steps(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if sum(isinstance(message, ModelResponse) for message in messages) < 3:
        return ModelResponse(parts=[ToolCallPart('lookup', {})])
    return ModelResponse(parts=[TextPart('done')])


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('complete', [False, True])
async def test_stream_releases_tool_context_cycles(complete: bool) -> None:
    managers: list[ToolManager[Deps]] = []

    async def three_streamed_steps(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if sum(isinstance(message, ModelResponse) for message in messages) < 3:
            yield {0: DeltaToolCall(name='lookup', json_args='{}')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(stream_function=three_streamed_steps), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        assert ctx.tool_manager is not None
        managers.append(ctx.tool_manager)
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    # Before Python 3.13, name inference snapshots the caller's locals and retains `deps` there.
    async with agent.run_stream('hello', deps=deps, infer_name=False) as result:
        if complete:
            assert await result.get_output() == 'done'
    assert managers
    assert all(manager.ctx is not None and manager.ctx.tool_manager is None for manager in managers)
    managers.clear()
    del deps, result
    if complete:
        assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_stream_name_inference_releases_dependencies() -> None:
    async def stream_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    # The agent is passed in a dict so the caller's frame never names it and every call infers again.
    # Streaming from a nested frame keeps the pre-3.13 snapshot of the caller's locals out of this one.
    async def stream_output(agents: dict[str, Agent[Deps, str]], deps: Deps) -> str:
        async with agents['agent'].run_stream('hello', deps=deps) as result:
            return await result.get_output()

    agent = Agent(FunctionModel(stream_function=stream_text), deps_type=Deps)
    deps = Deps()
    ref = weakref.ref(deps)
    assert await stream_output({'agent': agent}, deps) == 'done'
    assert agent.name is None
    del deps
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_completed_run_releases_dependencies() -> None:
    agent = Agent(FunctionModel(three_steps), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    # Before Python 3.13, name inference snapshots the caller's locals and retains `deps` there.
    result = await agent.run('hello', deps=deps, infer_name=False)
    assert result.output == 'done'
    del deps, result
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_deferred_resume_releases_dependencies() -> None:
    agent = Agent(TestModel(custom_output_text='done'), deps_type=Deps, output_type=[str, DeferredToolRequests])

    @agent.tool(requires_approval=True)
    async def lookup(ctx: RunContext[Deps]) -> str:
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    deferred = await agent.run('hello', deps=deps, infer_name=False)
    assert isinstance(deferred.output, DeferredToolRequests)
    resumed = await agent.run(
        deps=deps,
        infer_name=False,
        message_history=deferred.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={deferred.output.approvals[0].tool_call_id: True}),
    )
    assert resumed.output == 'done'
    del deps, deferred, resumed
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_retained_run_keeps_result_and_messages() -> None:
    agent = Agent(FunctionModel(three_steps), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    async with agent.iter('hello', deps=deps, infer_name=False) as run:
        async for _ in run:
            pass
    del deps
    assert ref() is not None
    assert run.result is not None
    assert run.result.output == 'done'
    assert run.result.all_messages() == run.all_messages()
    assert run.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=datetime.UTC))],
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:three_steps:',
                model_id='function:function:three_steps:',
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup', content='result', timestamp=IsNow(tz=datetime.UTC), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=4),
                model_name='function:three_steps:',
                model_id='function:function:three_steps:',
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup', content='result', timestamp=IsNow(tz=datetime.UTC), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=53, output_tokens=6),
                model_name='function:three_steps:',
                model_id='function:function:three_steps:',
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup', content='result', timestamp=IsNow(tz=datetime.UTC), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=54, output_tokens=7),
                model_name='function:three_steps:',
                model_id='function:function:three_steps:',
                timestamp=IsNow(tz=datetime.UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    del run
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_retained_contexts_keep_tools() -> None:
    contexts: list[tuple[RunContext[Deps], dict[str, ToolDefinition]]] = []
    agent = Agent(FunctionModel(three_steps), deps_type=Deps)

    @agent.instructions
    async def instructions(ctx: RunContext[Deps]) -> str:
        contexts.append((ctx, ctx.tools))
        return 'Use lookup'

    async def prepare(ctx: RunContext[Deps], tool: ToolDefinition) -> ToolDefinition:
        contexts.append((ctx, ctx.tools))
        return tool

    @agent.tool(prepare=prepare)
    async def lookup(ctx: RunContext[Deps]) -> str:
        contexts.append((ctx, ctx.tools))
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    result = await agent.run('hello', deps=deps, infer_name=False)
    del deps, result
    assert ref() is not None
    assert contexts
    assert all(ctx.tools == tools for ctx, tools in contexts)
    assert any('lookup' in tools for _, tools in contexts)
    assert all(ctx.deps is ref() for ctx, _ in contexts)
    contexts.clear()
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('exit_kind', ['break', 'error', 'cancel'])
async def test_interrupted_iteration_releases_dependencies(exit_kind: Literal['break', 'error', 'cancel']) -> None:
    contexts: list[RunContext[Deps]] = []
    agent = Agent(FunctionModel(three_steps), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        contexts.append(ctx)
        return 'result'

    deps = Deps()
    ref = weakref.ref(deps)
    run = None
    try:
        async with agent.iter('hello', deps=deps, infer_name=False) as run:
            while not contexts:
                await anext(run)
            if exit_kind == 'error':
                raise ValueError('consumer failed')
            if exit_kind == 'cancel':
                run.cancel()
                try:
                    await anext(run)
                except asyncio.CancelledError:
                    pass
    except ValueError:
        pass
    assert run is not None
    assert run.result is None
    assert contexts[0].tools.keys() == {'lookup'}
    if exit_kind == 'error':
        # Consumer errors can retain run state independently of these edges on Python 3.11.
        assert run._graph_run._iterator_instance is None  # pyright: ignore[reportPrivateUsage]
        iterator = run._graph_run._iterator  # pyright: ignore[reportPrivateUsage]
        assert isinstance(iterator, AsyncGeneratorType)
        assert iterator.ag_frame is None
        manager = run.ctx.deps.tool_manager
        assert manager.ctx is not None
        assert manager.ctx.tool_manager is None
        del manager
    contexts.clear()
    del deps, run
    if exit_kind != 'error':
        assert ref() is None


class RunDeadline(AbstractCapability[Deps]):
    """A `wrap_run` whose frame refers to the task running it, through the `asyncio.timeout()` it holds."""

    async def wrap_run(self, ctx: RunContext[Deps], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        deadline = asyncio.timeout(READINESS_WAIT_TIMEOUT)
        async with deadline:
            return await handler()


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('exit_kind', ['break', 'cancel'])
async def test_interrupted_iteration_releases_dependencies_of_wrap_run_holding_its_task(
    exit_kind: Literal['break', 'cancel'],
) -> None:
    agent = Agent(TestModel(), deps_type=Deps, capabilities=[RunDeadline()])
    deps = Deps()
    ref = weakref.ref(deps)
    async with agent.iter('hello', deps=deps, infer_name=False) as run:
        await anext(run)
        if exit_kind == 'cancel':
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await anext(run)
    del deps, run
    assert ref() is None


class FrameMarker:
    """Held only by a hook's frame, so a weak reference to it shows whether the frame was released."""


class WrapRunPendingCleanup(AbstractCapability[Deps]):
    """`RunDeadline` with cleanup after `handler()` that waits until it's cancelled."""

    def __init__(self) -> None:
        self.cleanup_started = anyio.Event()
        self.frame: weakref.ref[FrameMarker] | None = None

    async def wrap_run(self, ctx: RunContext[Deps], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        marker = FrameMarker()
        self.frame = weakref.ref(marker)
        deadline = asyncio.timeout(READINESS_WAIT_TIMEOUT)
        async with deadline:
            try:
                return await handler()
            finally:
                self.cleanup_started.set()
                await anyio.Event().wait()


class WrapModelRequestPendingCleanup(AbstractCapability[Deps]):
    """`WrapRunPendingCleanup` for `wrap_model_request`."""

    def __init__(self) -> None:
        self.cleanup_started = anyio.Event()
        self.frame: weakref.ref[FrameMarker] | None = None

    async def wrap_model_request(
        self, ctx: RunContext[Deps], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        marker = FrameMarker()
        self.frame = weakref.ref(marker)
        deadline = asyncio.timeout(READINESS_WAIT_TIMEOUT)
        async with deadline:
            try:
                return await handler(request_context)
            finally:
                self.cleanup_started.set()
                await anyio.Event().wait()


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('cancel_until_done', [False, True])
@pytest.mark.parametrize('exit_kind', ['run', 'iter_break', 'run_stream'])
async def test_native_cancel_during_teardown_releases_hook_frame_holding_its_task(
    exit_kind: Literal['run', 'iter_break', 'run_stream'], cancel_until_done: bool
) -> None:
    # The cancellation propagates through the stdlib `AsyncExitStack.__aexit__`, whose frame cycles with it and keeps
    # `deps` alive until garbage collection on its own, so this checks that the hook's frame was released.
    capability = WrapModelRequestPendingCleanup() if exit_kind == 'run_stream' else WrapRunPendingCleanup()
    agent = Agent(TestModel(), deps_type=Deps, capabilities=[capability])

    async def drive() -> None:
        if exit_kind == 'run':
            await agent.run('hello', deps=Deps(), infer_name=False)
        elif exit_kind == 'iter_break':
            async with agent.iter('hello', deps=Deps(), infer_name=False) as run:
                await anext(run)
        else:
            async with agent.run_stream('hello', deps=Deps(), infer_name=False) as result:
                await result.get_output()

    task = asyncio.create_task(drive())
    await asyncio.wait_for(capability.cleanup_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    if cancel_until_done:
        # A supervisor that cancels until the task is done lands more cancels while the teardown waits for the hook's
        # task, including after that task has finished.
        async with asyncio.timeout(READINESS_WAIT_TIMEOUT):
            while not task.done():
                await asyncio.sleep(0)
                task.cancel()
    await asyncio.wait((task,), timeout=READINESS_WAIT_TIMEOUT)
    with pytest.raises(asyncio.CancelledError):
        task.result()
    del task
    assert capability.frame is not None
    assert capability.frame() is None


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize(
    ('method', 'cancellation', 'body_returns'),
    [
        ('wait', 'native', False),
        ('wait', 'anyio', False),
        ('result', 'native', False),
        ('result', 'anyio', False),
        ('result', 'native', True),
    ],
)
async def test_scoped_task_is_released_without_gc_after_its_awaiter_is_cancelled(
    method: Literal['wait', 'result'], cancellation: Literal['native', 'anyio'], body_returns: bool
) -> None:
    # Not through an agent run: there the cancellation already cycles with the stdlib `AsyncExitStack.__aexit__` frame,
    # which hides this one. `wait()` and `result()` delete their `cancellation` local before they raise or return; a
    # local left in that frame would cycle with the cancellation's traceback and keep the `ScopedTask` alive until GC.
    # The `body_returns` case fails without that `del` only on CPython versions whose traceback links the frame of
    # `wait_passing_on_cancellation()` back to `result()`'s, such as 3.12.15 and 3.13.3.
    refs: list[weakref.ref[ScopedTask[int]]] = []
    cancel_awaiter: Callable[[], object] | None = None

    async def body() -> int:
        assert cancel_awaiter is not None
        cancel_awaiter()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            if not body_returns:
                raise
        return 42

    async def await_scoped_task() -> int | None:
        scoped = ScopedTask(body())
        refs.append(weakref.ref(scoped))
        if method == 'wait':
            await scoped.wait()
        else:
            return await scoped.result()

    async with asyncio.timeout(READINESS_WAIT_TIMEOUT):
        if cancellation == 'native':
            awaiter = asyncio.create_task(await_scoped_task())
            cancel_awaiter = awaiter.cancel
            await asyncio.wait((awaiter,))
            if body_returns:
                assert awaiter.result() == 42
            else:
                with pytest.raises(asyncio.CancelledError):
                    awaiter.result()
            del awaiter
        else:
            with anyio.CancelScope() as scope:
                cancel_awaiter = scope.cancel
                await await_scoped_task()
            assert scope.cancelled_caught
    assert refs[0]() is None


@pytest.mark.usefixtures('no_gc')
async def test_outer_cancellation_releases_run_references() -> None:
    tool_started = anyio.Event()
    contexts: list[RunContext[Deps]] = []
    agent = Agent(FunctionModel(three_steps), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> None:
        contexts.append(ctx)
        tool_started.set()
        await anyio.sleep_forever()

    async def cancel_when_tool_started(scope: anyio.CancelScope) -> None:
        await tool_started.wait()
        scope.cancel()

    run = None
    with anyio.fail_after(READINESS_WAIT_TIMEOUT), anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancel_when_tool_started, scope)
            async with agent.iter('hello', deps=Deps(), infer_name=False) as run:
                async for _ in run:
                    pass

    assert scope.cancelled_caught
    assert run is not None
    assert run.result is None
    assert contexts[0].tools.keys() == {'lookup'}
    assert run._graph_run._iterator_instance is None  # pyright: ignore[reportPrivateUsage]
    iterator = run._graph_run._iterator  # pyright: ignore[reportPrivateUsage]
    assert isinstance(iterator, AsyncGeneratorType)
    assert iterator.ag_frame is None
    manager = run.ctx.deps.tool_manager
    assert manager.ctx is not None
    assert manager.ctx.tool_manager is None
