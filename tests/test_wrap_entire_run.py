"""Tests for the `wrap_entire_run` capability hook.

`wrap_entire_run` brackets the complete run lifecycle — model resolution, toolset entry, and
`wrap_run` — so these tests pin its ordering relative to the other capability hooks, the
`RunPreparationContext` it receives, and its error-observation and suppression semantics.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

from .capability_models import simple_model_function

pytestmark = [
    pytest.mark.anyio,
]


@pytest.mark.parametrize('run_mode', ['run', 'iter', 'run_stream'])
async def test_wrap_entire_run_brackets_complete_run_lifecycle(run_mode: str) -> None:
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
    class LifecycleCapability(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> LifecycleCapability:
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
        capabilities=[IterCapability('iter1'), IterCapability('iter2'), LifecycleCapability()],
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
        'for_run',
        'wrap_run',
        'before_run',
        'toolset_enter',
        'model',
        'after_run',
        'toolset_exit',
        'iter2_exit',
        'iter1_exit',
    ]


async def test_wrap_entire_run_receives_preparation_context() -> None:
    seen: list[RunPreparationContext[Any]] = []
    run_ids: list[tuple[str | None, str | None]] = []

    @dataclass
    class CaptureContext(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            seen.append(ctx)
            yield

        async def before_run(self, ctx: RunContext[Any]) -> None:
            run_ids.append((ctx.run_id, ctx.conversation_id))

    model = FunctionModel(simple_model_function)
    prior_message = ModelRequest(parts=[UserPromptPart('prior')])
    await Agent(model, capabilities=[CaptureContext()]).run('agent model')
    await Agent(model, capabilities=[CaptureContext()]).run(
        'explicit model', model=model, message_history=[prior_message]
    )

    # Only an explicitly passed model is known before resolution.
    assert seen[0].model is None
    assert seen[1].model is model
    assert seen[0].messages == []
    assert seen[1].messages == [prior_message]
    assert [(ctx.run_id, ctx.conversation_id) for ctx in seen] == run_ids


async def test_wrap_entire_run_messages_are_a_copy_of_the_history() -> None:
    @dataclass
    class ClearMessages(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            ctx.messages.clear()
            yield

    prior_message = ModelRequest(parts=[UserPromptPart('prior')])
    result = await Agent(FunctionModel(simple_model_function), capabilities=[ClearMessages()]).run(
        'hello', message_history=[prior_message]
    )

    assert result.all_messages()[0] is prior_message


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


async def test_wrap_entire_run_closes_on_base_exception_between_preparation_steps() -> None:
    events: list[str] = []

    class PreparationInterrupted(BaseException):
        pass

    class InterruptingUsageLimits(UsageLimits):
        def __bool__(self) -> bool:
            raise PreparationInterrupted

    @dataclass
    class TrackExit(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')

    with pytest.raises(PreparationInterrupted):
        await Agent(FunctionModel(simple_model_function), capabilities=[TrackExit()]).run(
            'run', usage_limits=InterruptingUsageLimits()
        )

    assert events == ['enter', 'exit']


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


@pytest.mark.parametrize('loaded_from', ['run', 'history'])
async def test_deferred_capability_wrap_entire_run_never_fires(loaded_from: str) -> None:
    """The whole-run chain is entered once at run start, so a capability loaded later never joins it."""
    events: list[str] = []

    @dataclass
    class DeferredIter(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: RunPreparationContext[Any]) -> AsyncGenerator[None]:
            raise AssertionError('wrap_entire_run must not fire for a deferred capability')
            yield  # pragma: no cover

        async def before_run(self, ctx: RunContext[Any]) -> None:
            events.append('before_run')

    capability = DeferredIter(id='deferred', description='Load this capability.', defer_loading=True)

    if loaded_from == 'run':

        def load_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, LoadCapabilityReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'deferred'})])
            return ModelResponse(parts=[TextPart('done')])

        result = await Agent(FunctionModel(load_then_finish), capabilities=[capability]).run('load it')
        assert any(
            isinstance(part, LoadCapabilityReturnPart) for message in result.all_messages() for part in message.parts
        )
    else:
        history: list[ModelMessage] = [
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'deferred'}, tool_call_id='load-deferred')]),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='load-deferred')]),
        ]
        await Agent(FunctionModel(simple_model_function), capabilities=[capability]).run(
            'resume', message_history=history
        )
        assert events == ['before_run']
