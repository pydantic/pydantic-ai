"""Run state should be released by reference counting once the application drops it.

In-process models isolate object lifetime; provider recordings add no ownership coverage.
"""

from __future__ import annotations

import asyncio
import datetime
import gc
import traceback
import weakref
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import AsyncGeneratorType
from typing import Literal, Never

import anyio
import pytest

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, _utils
from pydantic_ai.capabilities.abstract import AbstractCapability, AgentNode, NodeResult, WrapNodeRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.graph_builder import ErrorMarker, GraphRun

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
    assert ref() is None


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


def raise_failure(message: str, explicit_cause: bool) -> Never:
    try:
        raise ValueError('original cause')
    except ValueError as exc:
        if explicit_cause:
            raise RuntimeError(message) from exc
        raise RuntimeError(message)


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('explicit_cause', [False, True])
@pytest.mark.parametrize(
    ('failure', 'drive', 'node_hook'),
    [
        ('sync_tool', 'iter', 'none'),
        ('sync_tool', 'run', 'none'),
        ('async_tool', 'iter', 'none'),
        ('async_tool', 'run', 'none'),
        ('async_tool', 'iter', 'passthrough'),
        ('async_tool', 'run', 'passthrough'),
        ('async_tool', 'iter', 'replacement'),
        ('async_tool', 'run', 'replacement'),
        ('sync_model', 'iter', 'none'),
        ('sync_model', 'run', 'none'),
        ('async_model', 'iter', 'none'),
        ('async_model', 'run', 'none'),
        ('consumer', 'iter', 'none'),
    ],
)
async def test_failed_run_releases_dependencies(  # noqa: C901
    failure: Literal['sync_tool', 'async_tool', 'sync_model', 'async_model', 'consumer'],
    drive: Literal['iter', 'run'],
    node_hook: Literal['none', 'passthrough', 'replacement'],
    explicit_cause: bool,
) -> None:
    class NodeHook(AbstractCapability[Deps]):
        async def before_node_run(self, ctx: RunContext[Deps], *, node: AgentNode[Deps]) -> AgentNode[Deps]:
            if node_hook == 'replacement' and Agent.is_call_tools_node(node):
                return replace(node)
            return node

        async def wrap_node_run(
            self, ctx: RunContext[Deps], *, node: AgentNode[Deps], handler: WrapNodeRunHandler[Deps]
        ) -> NodeResult[Deps]:
            return await handler(node)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if failure in ('sync_model', 'async_model'):
            raise_failure('model failed', explicit_cause)
        return ModelResponse(parts=[ToolCallPart('lookup', {})])

    async def async_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return model(messages, info)

    agent = Agent(
        FunctionModel(async_model if failure == 'async_model' else model),
        deps_type=Deps,
        capabilities=[NodeHook()] if node_hook != 'none' else [],
    )

    def lookup(ctx: RunContext[Deps]) -> str:
        raise_failure('tool failed', explicit_cause)

    async def async_lookup(ctx: RunContext[Deps]) -> str:
        return lookup(ctx)

    agent.tool(name='lookup')(async_lookup if failure == 'async_tool' else lookup)
    errors: list[BaseException] = []
    runs: list[AgentRun[Deps, str]] = []

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        run = node = None
        try:
            if drive == 'run':
                await agent.run('hi', deps=deps, infer_name=False)
            else:
                async with agent.iter('hi', deps=deps, infer_name=False) as run:
                    runs.append(run)
                    async for node in run:
                        if failure == 'consumer' and Agent.is_call_tools_node(node):
                            raise_failure('consumer failed', explicit_cause)
        except RuntimeError as exc:
            errors.append(exc)
        del deps, run, node
        return ref

    ref = await execute()
    assert ref() is not None
    assert len(errors) == 1
    error = errors[0]
    if failure == 'consumer':
        assert str(error) == 'consumer failed'
    original_context = isinstance(error.__context__, ValueError) and str(error.__context__) == 'original cause'
    if explicit_cause:
        assert error.__cause__ is error.__context__
    else:
        assert error.__cause__ is None
    assert error.__traceback__ is not None
    names = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    origins: dict[str, str] = {
        'sync_tool': 'lookup',
        'async_tool': 'lookup',
        'sync_model': 'model',
        'async_model': 'model',
        'consumer': 'execute',
    }
    assert origins[failure] in names
    if failure in ('sync_tool', 'async_tool'):
        tb = error.__traceback__
        while tb.tb_frame.f_code.co_name != 'lookup':
            assert tb.tb_next is not None
            tb = tb.tb_next
        # A debugger can still inspect the user tool's context and dependency.
        assert tb.tb_frame.f_locals['ctx'].deps is ref()
        del tb
    if runs:
        assert runs[0].ctx.deps.user_deps is ref()
        assert runs[0].result is None
        assert runs[0].all_messages()
    del error
    errors.clear()
    if runs:
        assert ref() is not None
    runs.clear()
    if failure != 'async_model':
        # AnyIO can deliver a result before its worker drops the copied run context.
        # Reusing the worker guarantees its preceding work item has been released.
        await anyio.to_thread.run_sync(lambda: None)
    assert ref() is None
    assert original_context


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('stream_first', [False, True])
async def test_failed_tool_node_replays_during_manual_iteration(stream_first: bool) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('lookup', {})])

    agent = Agent(FunctionModel(model), deps_type=Deps)
    tool_calls = 0

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        nonlocal tool_calls
        tool_calls += 1
        raise_failure('tool failed', explicit_cause=True)

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        node = None
        async with agent.iter('hi', deps=deps, infer_name=False) as run:
            node = run.next_node
            assert Agent.is_user_prompt_node(node)
            node = await run.next(node)
            assert Agent.is_model_request_node(node)
            node = await run.next(node)
            assert Agent.is_call_tools_node(node)
            original_error: RuntimeError | None = None
            if stream_first:
                stream = None
                with pytest.raises(RuntimeError, match=r'^tool failed$') as streamed:
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass
                original_error = streamed.value
                del streamed, stream
            for _ in range(2):
                with pytest.raises(RuntimeError, match=r'^tool failed$') as raised:
                    await run.next(node)
                if original_error is None:
                    original_error = raised.value
                assert raised.value is original_error
                assert raised.value.__cause__ is raised.value.__context__
                assert str(raised.value.__cause__) == 'original cause'
                assert run.all_messages()
                assert run.result is None
                del raised
            assert tool_calls == 1
            del original_error
        del deps, run, node
        return ref

    ref = await execute()
    await anyio.to_thread.run_sync(lambda: None)
    assert tool_calls == 1
    assert ref() is None


async def test_failed_tool_node_replays_during_run_error_recovery() -> None:
    original_error = RuntimeError('tool failed')
    run: AgentRun[Deps, str] | None = None
    failed_node: AgentNode[Deps] | None = None

    class Recover(AbstractCapability[Deps]):
        async def on_run_error(self, ctx: RunContext[Deps], *, error: BaseException) -> AgentRunResult[str]:
            assert error is original_error
            assert run is not None
            assert failed_node is not None
            assert ctx.deps is deps
            with pytest.raises(RuntimeError, match=r'^tool failed$') as replayed:
                await run.next(failed_node)
            assert replayed.value is original_error
            assert run.result is None
            assert run.all_messages()
            return AgentRunResult(output='recovered')

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('lookup', {})])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[Recover()])
    tool_calls = 0

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> str:
        nonlocal tool_calls
        tool_calls += 1
        raise original_error

    deps = Deps()
    async with agent.iter('hi', deps=deps, infer_name=False) as active_run:
        run = active_run
        node = active_run.next_node
        assert Agent.is_user_prompt_node(node)
        node = await active_run.next(node)
        assert Agent.is_model_request_node(node)
        node = await active_run.next(node)
        assert Agent.is_call_tools_node(node)
        failed_node = node
        await active_run.next(node)

    assert active_run.result is not None
    assert active_run.result.output == 'recovered'
    assert active_run.all_messages()
    assert tool_calls == 1


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('cancellation', ['native', 'anyio'])
@pytest.mark.parametrize('during', ['model', 'tool'])
async def test_cancelled_run_releases_dependencies(
    cancellation: Literal['native', 'anyio'], during: Literal['model', 'tool']
) -> None:
    started = anyio.Event()
    errors: list[BaseException] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if during == 'model':
            started.set()
            await anyio.Event().wait()
        return ModelResponse(parts=[ToolCallPart('lookup', {})])

    agent = Agent(FunctionModel(model), deps_type=Deps)

    @agent.tool
    async def lookup(ctx: RunContext[Deps]) -> None:
        started.set()
        await anyio.Event().wait()

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        run = node = None
        try:
            async with agent.iter('hi', deps=deps, infer_name=False) as run:
                async for node in run:
                    pass
        except asyncio.CancelledError as exc:
            errors.append(exc)
        del deps, run, node
        return ref

    async def cancel(task: asyncio.Task[weakref.ReferenceType[Deps]], scope: anyio.CancelScope) -> None:
        await started.wait()
        if cancellation == 'native':
            task.cancel('external cancellation')
        else:
            scope.cancel()

    ref = None
    task = None
    with anyio.fail_after(READINESS_WAIT_TIMEOUT), anyio.CancelScope() as scope:
        task = asyncio.create_task(execute())
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancel, task, scope)
            ref = await task
    assert ref is not None
    assert task is not None
    assert ref() is not None
    assert len(errors) == 1
    assert isinstance(errors[0], asyncio.CancelledError)
    errors.clear()
    del task
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_custom_executor_sync_tool_error_releases_dependencies() -> None:
    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('fail', {})])

    agent = Agent(FunctionModel(model), deps_type=Deps)

    @agent.tool
    def fail(ctx: RunContext[Deps]) -> str:
        raise RuntimeError('custom worker failed')

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        with pytest.raises(RuntimeError, match='custom worker failed'):
            await agent.run('hi', deps=deps, infer_name=False)
        del deps
        return ref

    with ThreadPoolExecutor() as pool, Agent.using_thread_executor(pool):
        ref = await execute()
    await anyio.lowlevel.checkpoint()
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
async def test_inline_worker_error_releases_callback_dependencies() -> None:
    """Inline durable execution has no public Agent toggle, so exercise the internal worker path."""

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)

        def fail(dependency: Deps) -> None:
            raise RuntimeError(f'worker failed: {dependency}')

        with _utils.disable_threads(), pytest.raises(RuntimeError, match='worker failed'):
            await _utils.run_in_executor(fail, deps)
        del deps, fail
        return ref

    ref = await execute()
    assert ref() is None


async def test_worker_exception_value_is_a_result() -> None:
    """Agent serializes tool returns, so pin the worker's raw exception-as-value identity directly."""
    value = RuntimeError('returned as a value')
    assert await _utils.run_in_executor(lambda: value) is value


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('tool_kind', ['async', 'sync'])
async def test_run_stream_tool_error_releases_dependencies(tool_kind: Literal['async', 'sync']) -> None:
    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        calls: DeltaToolCalls = {0: DeltaToolCall(name='lookup', json_args='{}')}
        yield calls

    agent = Agent(FunctionModel(stream_function=model), deps_type=Deps)

    def lookup(ctx: RunContext[Deps]) -> str:
        raise_failure('streaming tool failed', explicit_cause=True)

    async def async_lookup(ctx: RunContext[Deps]) -> str:
        return lookup(ctx)

    agent.tool(name='lookup')(async_lookup if tool_kind == 'async' else lookup)
    errors: list[RuntimeError] = []

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        response = None
        try:
            async with agent.run_stream('hi', deps=deps, infer_name=False) as response:
                async for _ in response.stream_text():
                    pass
        except RuntimeError as exc:
            errors.append(exc)
        del deps, response
        return ref

    ref = await execute()
    assert ref() is not None
    assert len(errors) == 1
    assert str(errors[0]) == 'streaming tool failed'
    assert isinstance(errors[0].__cause__, ValueError)
    errors.clear()
    if tool_kind == 'sync':
        await anyio.to_thread.run_sync(lambda: None)
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
@pytest.mark.filterwarnings('ignore:Unclosed.*:ResourceWarning')
async def test_unentered_graph_run_releases_dependencies() -> None:
    builder = GraphBuilder(deps_type=Deps, output_type=int)
    builder.add(builder.edge_from(builder.start_node).transform(lambda ctx: 1).to(builder.end_node))
    graph = builder.build()
    deps = Deps()
    ref = weakref.ref(deps)
    run = GraphRun(graph, state=None, deps=deps, inputs=None, traceparent=None)
    # No task group was entered; the unentered in-memory streams' warnings are unrelated to ownership.
    del deps, run
    assert ref() is None


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('failure', ['step', 'consumer'])
async def test_failed_graph_keeps_inspection_without_retaining_dropped_dependencies(
    failure: Literal['step', 'consumer'],
) -> None:
    builder = GraphBuilder(deps_type=Deps, output_type=int)

    @builder.step
    async def fail_step(ctx: StepContext[None, Deps, None]) -> int:
        raise RuntimeError('step failed')

    builder.add(builder.edge_from(builder.start_node).to(fail_step), builder.edge_from(fail_step).to(builder.end_node))
    graph = builder.build()
    errors: list[RuntimeError] = []
    runs: list[GraphRun[None, Deps, int]] = []

    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)
        run = None
        try:
            async with graph.iter(deps=deps, infer_name=False) as run:
                runs.append(run)
                if failure == 'consumer':
                    raise RuntimeError('consumer failed')
                async for _ in run:
                    pass
        except RuntimeError as exc:
            errors.append(exc)
        del deps, run
        return ref

    ref = await execute()
    assert ref() is not None
    assert len(errors) == 1
    assert runs[0].deps is ref()
    assert runs[0].output is None
    if failure == 'step':
        marker = runs[0].next_task
        assert isinstance(marker, ErrorMarker)
        assert marker.error is errors[0]
        del marker
    errors.clear()
    assert ref() is not None
    runs.clear()
    assert ref() is None
