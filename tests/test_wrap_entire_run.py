"""Tests for the `wrap_entire_run` capability hook.

`wrap_entire_run` brackets the complete run lifecycle — sandbox acquisition, model and
toolset entry, and `wrap_run` — so these tests pin its ordering relative to the other
capability hooks, the `RunPreparationContext` it receives, and its error-observation and
suppression semantics. Unit-style capability doubles are used throughout because the hook
contract (ordering, suppression detection, teardown propagation) is internal machinery
that no recorded provider exchange can exercise.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import RunPreparationContext
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model, ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import LocalSandbox, SandboxRef
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_graph import End

from .capability_models import simple_model_function

pytestmark = [
    pytest.mark.anyio,
]


@pytest.mark.parametrize('run_mode', ['run', 'iter', 'run_stream'])
async def test_wrap_entire_run_brackets_sandbox_and_complete_run_lifecycle(  # noqa: C901
    run_mode: str, tmp_path: Path
) -> None:
    events: list[str] = []

    class LifecycleToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> LifecycleToolset:
            events.append('toolset_enter')
            await self.wrapped.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            events.append('toolset_exit')
            return await self.wrapped.__aexit__(*args)

    @dataclass
    class IterCapability(AbstractCapability[Any]):
        name: str

        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            events.append(f'{self.name}_enter')
            try:
                yield
            finally:
                events.append(f'{self.name}_exit')

    @dataclass
    class SandboxCapability(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            events.append('acquire_sandbox')
            return SandboxRef(provider='local', sandbox_id=str(tmp_path))

        # No `get_sandbox`: the test's claim is hook ordering, and nothing operates on the
        # sandbox, so connection is never attempted.

        async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
            events.append('release_sandbox')

        async def for_run(self, ctx: RunContext[Any]) -> SandboxCapability:
            events.append('for_run')
            return self

        def get_toolset(self) -> AbstractToolset[Any]:
            return LifecycleToolset(wrapped=FunctionToolset())

        async def wrap_run(
            self,
            ctx: RunContext[Any],
            *,
            handler: Callable[[], Awaitable[AgentRunResult[Any]]],
        ) -> AgentRunResult[Any]:
            events.append('wrap_run')
            return await handler()

        async def before_run(self, ctx: RunContext[Any]) -> None:
            events.append('before_run')

        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            events.append('after_run')
            return result

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        events.append('model')
        return ModelResponse(parts=[TextPart('done')])

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        events.append('model')
        yield 'done'

    agent = Agent(
        FunctionModel(model_function, stream_function=stream_function),
        capabilities=[IterCapability('iter1'), IterCapability('iter2'), SandboxCapability()],
    )

    if run_mode == 'run':
        await agent.run('hello')
    elif run_mode == 'iter':
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
    else:
        async with agent.run_stream('hello') as stream:
            await stream.get_output()

    assert events == [
        'iter1_enter',
        'iter2_enter',
        'acquire_sandbox',
        'for_run',
        'wrap_run',
        'before_run',
        'toolset_enter',
        'model',
        'after_run',
        'toolset_exit',
        'release_sandbox',
        'iter2_exit',
        'iter1_exit',
    ]


async def test_wrap_entire_run_receives_preparation_context(tmp_path: Path) -> None:
    seen: list[RunPreparationContext[Any]] = []
    seen_message_counts: list[int] = []
    run_ids: list[tuple[str | None, str | None]] = []

    @dataclass
    class CaptureContext(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            seen.append(ctx)
            seen_message_counts.append(len(ctx.messages))
            ctx.messages.clear()
            yield

        async def before_run(self, ctx: RunContext[Any]) -> None:
            run_ids.append((ctx.run_id, ctx.conversation_id))

    @dataclass
    class ServeSandbox(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(provider='local', sandbox_id=str(tmp_path))

        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> LocalSandbox | None:
            if ref is None or ref.provider != 'local':
                return None
            return LocalSandbox(ref.sandbox_id)

    model = FunctionModel(simple_model_function)
    await Agent(model, capabilities=[CaptureContext(), ServeSandbox()]).run('capability served')

    backend = LocalSandbox(tmp_path)
    prior_message = ModelRequest(parts=[UserPromptPart('prior')])
    result = await Agent(model, capabilities=[CaptureContext()]).run(
        'caller supplied',
        model=model,
        sandbox=backend,
        message_history=[prior_message],
    )
    ref = SandboxRef(provider='local', sandbox_id=str(tmp_path), capability_id='serve')
    await Agent(model, capabilities=[CaptureContext(), ServeSandbox(id='serve')]).run(
        'caller supplied ref', sandbox=ref
    )

    assert seen[0].model is None
    assert seen[0].sandbox is None
    assert seen[0].messages == []
    assert seen[1].model is model
    assert seen[1].sandbox is backend
    assert seen[2].sandbox is ref
    assert seen_message_counts == [0, 1, 0]
    assert result.all_messages()[0] is prior_message
    assert [(ctx.run_id, ctx.conversation_id) for ctx in seen] == run_ids


async def test_wrap_entire_run_observes_propagated_and_recovered_errors() -> None:
    observed: list[BaseException | None] = []

    @dataclass
    class ObserveExit(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            try:
                yield
            except BaseException as exc:
                observed.append(exc)
                raise
            else:
                observed.append(None)

    @dataclass
    class RecoverError(AbstractCapability[Any]):
        async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered')

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    model = FunctionModel(failing_model)
    with pytest.raises(RuntimeError, match='model exploded'):
        await Agent(model, capabilities=[ObserveExit()]).run('fail')

    result = await Agent(model, capabilities=[ObserveExit(), RecoverError()]).run('recover')

    assert result.output == 'recovered'
    assert isinstance(observed[0], RuntimeError)
    assert observed[1] is None


async def test_wrap_entire_run_observes_model_resolution_error() -> None:
    events: list[str] = []

    @dataclass
    class ObserveResolution(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')

        async def resolve_model_id(
            self, ctx: ModelResolutionContext[Any], *, model_id: KnownModelName | str
        ) -> Model | None:
            raise RuntimeError('resolution failed')

    with pytest.raises(RuntimeError, match='resolution failed'):
        await Agent(None, capabilities=[ObserveResolution()]).run('fail', model='custom')

    assert events == ['enter', 'exit']


@pytest.mark.parametrize('combined', [False, True])
async def test_wrap_entire_run_suppression_is_a_loud_contract_violation(combined: bool) -> None:
    """A hook that swallows the run's exception must not leave the run in a broken half-state.

    The run detects the suppression after the exit stack unwinds and raises `UserError`
    with the suppressed error as the cause, instead of dying on an unrelated assertion.
    """

    @dataclass
    class SuppressError(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            try:
                yield
            except RuntimeError:
                pass

    @dataclass
    class OuterHook(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            yield

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    capabilities: list[AbstractCapability[Any]] = [SuppressError()]
    if combined:
        capabilities.insert(0, OuterHook())

    with pytest.raises(UserError, match='suppressed the run error') as exc_info:
        await Agent(FunctionModel(failing_model), capabilities=capabilities).run('fail')

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == 'model exploded'


@pytest.mark.parametrize('model_fails', [True, False], ids=['failing-run', 'clean-run'])
async def test_wrap_entire_run_cleanup_error_follows_context_manager_semantics(model_fails: bool) -> None:
    @dataclass
    class RaiseOnExit(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            try:
                yield
            finally:
                raise TypeError('teardown bug')

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    with pytest.raises(TypeError, match='teardown bug') as exc_info:
        model_function = failing_model if model_fails else simple_model_function
        await Agent(FunctionModel(model_function), capabilities=[RaiseOnExit()]).run('run')

    if model_fails:
        context = exc_info.value.__context__
        assert isinstance(context, RuntimeError)
        assert str(context) == 'model exploded'


async def test_wrap_entire_run_cannot_suppress_toolset_teardown_error() -> None:
    class RaiseOnExitToolset(WrapperToolset[Any]):
        async def __aexit__(self, *args: Any) -> bool | None:
            await self.wrapped.__aexit__(*args)
            raise RuntimeError('toolset teardown failed')

    @dataclass
    class TeardownCapability(AbstractCapability[Any]):
        def get_toolset(self) -> AbstractToolset[Any]:
            return RaiseOnExitToolset(wrapped=FunctionToolset())

    @dataclass
    class SuppressTeardown(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            try:
                yield
            except RuntimeError:
                pass

    with pytest.raises(UserError, match='suppressed the run error') as exc_info:
        await Agent(
            FunctionModel(simple_model_function),
            capabilities=[SuppressTeardown(), TeardownCapability()],
        ).run('succeed')

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == 'toolset teardown failed'


async def test_deferred_capability_wrap_entire_run_never_fires_when_loaded() -> None:
    events: list[str] = []

    @dataclass
    class DeferredIter(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:  # pragma: no cover
            events.append('entered')
            yield

        def get_instructions(self) -> str:
            return 'Deferred instructions.'

    def load_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, LoadCapabilityReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'deferred'})])
        return ModelResponse(parts=[TextPart('done')])

    capability = DeferredIter(
        id='deferred',
        description='Load this capability.',
        defer_loading=True,
    )
    result = await Agent(FunctionModel(load_then_finish), capabilities=[capability]).run('load it')

    assert result.output == 'done'
    assert any(
        isinstance(part, LoadCapabilityReturnPart) for message in result.all_messages() for part in message.parts
    )
    assert events == []


async def test_resumed_deferred_capability_wrap_entire_run_never_fires_when_already_loaded() -> None:
    events: list[str] = []

    @dataclass
    class DeferredIter(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:  # pragma: no cover
            events.append('wrap_entire_run')
            yield

        async def before_run(self, ctx: RunContext[Any]) -> None:
            events.append('before_run')

    capability = DeferredIter(
        id='deferred',
        description='Load this capability.',
        defer_loading=True,
    )
    history: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'deferred'}, tool_call_id='load-deferred')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='load-deferred')]),
    ]

    result = await Agent(FunctionModel(simple_model_function), capabilities=[capability]).run(
        'resume', message_history=history
    )

    assert result.output == 'response from model'
    assert events == ['before_run']
