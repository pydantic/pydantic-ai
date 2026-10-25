"""Errors must not make a dropped run depend on cyclic garbage collection."""

from __future__ import annotations

import asyncio
import gc
import traceback
import weakref
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from types import FrameType, TracebackType
from typing import Literal, Never

import anyio
import pytest

from pydantic_ai import Agent, RunContext, _utils
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRun
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.graph_builder import ErrorMarker, GraphRun

READINESS_WAIT_TIMEOUT = 5


@dataclass
class Deps:
    pass


def referrer_name(owner: object) -> str:
    if isinstance(owner, FrameType):
        return f'frame {owner.f_code.co_filename}::{owner.f_code.co_name}'
    if isinstance(owner, TracebackType):
        return f'traceback {owner.tb_frame.f_code.co_filename}::{owner.tb_frame.f_code.co_name}'
    return type(owner).__qualname__


def dependency_referrers(ref: weakref.ReferenceType[Deps]) -> str:
    # Inspect ownership only after the release assertion fails, so diagnostics cannot retain frames beforehand.
    owners: list[object] = gc.get_referrers(ref())
    return '\n'.join(
        f'{referrer_name(owner)} <- {", ".join(referrer_name(parent) for parent in gc.get_referrers(owner))}'
        for owner in owners
    )


@pytest.fixture
def no_gc() -> Generator[None]:
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


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
    ('failure', 'drive'),
    [
        ('sync_tool', 'iter'),
        ('sync_tool', 'run'),
        ('async_tool', 'iter'),
        ('async_tool', 'run'),
        ('sync_model', 'iter'),
        ('sync_model', 'run'),
        ('async_model', 'iter'),
        ('async_model', 'run'),
        ('consumer', 'iter'),
    ],
)
async def test_failed_run_releases_dependencies(  # noqa: C901
    failure: Literal['sync_tool', 'async_tool', 'sync_model', 'async_model', 'consumer'],
    drive: Literal['iter', 'run'],
    explicit_cause: bool,
) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if failure in ('sync_model', 'async_model'):
            raise_failure('model failed', explicit_cause)
        if not any(isinstance(message, ModelResponse) for message in messages):
            return ModelResponse(parts=[ToolCallPart('lookup', {})])
        return ModelResponse(parts=[TextPart('done')])

    async def async_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return model(messages, info)

    agent = Agent(FunctionModel(async_model if failure == 'async_model' else model), deps_type=Deps)

    def lookup(ctx: RunContext[Deps]) -> str:
        if failure in ('sync_tool', 'async_tool'):
            raise_failure('tool failed', explicit_cause)
        return 'result'

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
    assert ref() is None, dependency_referrers(ref)
    assert original_context


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
    assert ref() is None, dependency_referrers(ref)


@pytest.mark.usefixtures('no_gc')
@pytest.mark.parametrize('executor', ['default', 'custom', 'inline'])
async def test_worker_error_releases_callback_dependencies(executor: Literal['default', 'custom', 'inline']) -> None:
    async def execute() -> weakref.ReferenceType[Deps]:
        deps = Deps()
        ref = weakref.ref(deps)

        def fail(dependency: Deps) -> None:
            raise RuntimeError(f'worker failed: {dependency}')

        try:
            await _utils.run_in_executor(fail, deps)
        except RuntimeError:
            pass
        del deps, fail
        return ref

    with ExitStack() as stack:
        if executor == 'custom':
            pool = stack.enter_context(ThreadPoolExecutor())
            stack.enter_context(_utils.using_thread_executor(pool))
        elif executor == 'inline':
            stack.enter_context(_utils.disable_threads())
        ref = await execute()
    # Let the awaiting task release its completed worker future before checking for cycles.
    await anyio.lowlevel.checkpoint()
    assert ref() is None, dependency_referrers(ref)


async def test_worker_exception_value_is_a_result() -> None:
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
    assert ref() is None, dependency_referrers(ref)


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
    assert ref() is None, dependency_referrers(ref)


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
    assert ref() is None, dependency_referrers(ref)
