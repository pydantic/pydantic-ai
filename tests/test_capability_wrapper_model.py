"""The `get_wrapper_model` capability hook: which model it wraps, in what order, and when it's skipped."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import anyio.lowlevel
import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks, WrapperCapability
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from .continuation_utils import ScriptedContinuationModel, StreamSegment


class _RecordingModel(WrapperModel):
    def __init__(self, wrapped: Model, name: str, calls: list[str]) -> None:
        super().__init__(wrapped)
        self.name = name
        self.calls = calls

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.calls.append(f'{self.name} -> {_describe(self.wrapped)}')
        return await super().request(messages, model_settings, model_request_parameters)


def _describe(model: Model) -> str:
    return model.name if isinstance(model, _RecordingModel) else model.model_name


@dataclass
class _Wrap(AbstractCapability[Any]):
    name: str
    calls: list[str]

    def get_wrapper_model(self, model: Model) -> Model | None:
        return _RecordingModel(model, self.name, self.calls)


@dataclass
class _LeaveAlone(AbstractCapability[Any]):
    def get_wrapper_model(self, model: Model) -> Model | None:
        return None


async def test_the_first_capability_wraps_outermost() -> None:
    calls: list[str] = []
    agent = Agent(TestModel(), capabilities=[_Wrap('outer', calls), _LeaveAlone(), _Wrap('inner', calls)])

    assert (await agent.run('Hello')).output == 'success (no tool calls)'
    assert calls == ['outer -> inner', 'inner -> test']


async def test_the_wrapper_applies_to_a_model_a_hook_swapped_in() -> None:
    calls: list[str] = []
    hooks = Hooks[Any]()
    swapped = TestModel(custom_output_text='swapped')
    swapped._model_name = 'swapped'  # pyright: ignore[reportPrivateUsage]

    @hooks.on.before_model_request
    async def swap(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.model = swapped
        return request_context

    seen: list[Model] = []

    @hooks.on.after_model_request
    async def record(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        seen.append(request_context.model)
        return response

    agent = Agent(TestModel(), capabilities=[_Wrap('wrapper', calls), hooks])

    assert (await agent.run('Hello')).output == 'swapped'
    assert calls == ['wrapper -> swapped']
    # Later hooks see the model that served the request.
    assert [_describe(model) for model in seen] == ['wrapper']


async def test_a_wrapper_capability_forwards_the_hook() -> None:
    calls: list[str] = []
    agent = Agent(TestModel(), capabilities=[WrapperCapability(_Wrap('wrapped', calls))])

    await agent.run('Hello')
    assert calls == ['wrapped -> test']


async def test_a_deferred_capability_does_not_wrap_the_model() -> None:
    calls: list[str] = []
    deferred = _Wrap('deferred', calls, id='deferred', description='Wraps the model.', defer_loading=True)
    agent = Agent(TestModel(call_tools=[]), capabilities=[deferred])

    await agent.run('Hello')
    assert calls == []


class _ScopedModel(WrapperModel):
    """Records being entered and exited, and whether it was entered when it served a request."""

    def __init__(self, wrapped: Model, events: list[object]) -> None:
        super().__init__(wrapped)
        self.events = events
        self.entered = False

    async def __aenter__(self) -> _ScopedModel:
        self.events.append('enter')
        self.entered = True
        return await super().__aenter__()

    async def __aexit__(self, *args: Any) -> bool | None:
        self.events.append('exit')
        self.entered = False
        return await super().__aexit__(*args)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.events.append(('request', self.entered))
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        self.events.append(('stream', self.entered))
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream
        self.events.append(('streamed', self.entered))


@dataclass
class _Scoped(AbstractCapability[Any]):
    events: list[object]

    def get_wrapper_model(self, model: Model) -> Model | None:
        return _ScopedModel(model, self.events)


async def test_the_wrapper_is_entered_around_each_request_it_serves() -> None:
    events: list[object] = []
    agent = Agent(TestModel(), capabilities=[_Scoped(events)])

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    await agent.run('Where?')
    assert events == ['enter', ('request', True), 'exit', 'enter', ('request', True), 'exit']


async def test_the_wrapper_is_entered_around_each_streamed_request_it_serves() -> None:
    events: list[object] = []
    agent = Agent(TestModel(), capabilities=[_Scoped(events)])
    async with agent.run_stream('Hello') as result:
        await result.get_output()
    assert events == ['enter', ('stream', True), ('streamed', True), 'exit']


async def test_the_wrapper_is_exited_when_its_request_fails() -> None:
    events: list[object] = []

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('Failed')

    agent = Agent(FunctionModel(fail), capabilities=[_Scoped(events)])
    with pytest.raises(ValueError, match='Failed'):
        await agent.run('Hello')
    assert events == ['enter', ('request', True), 'exit']


class _RecordingLifecycle(TestModel):
    def __init__(self, events: list[object]) -> None:
        super().__init__()
        self.events = events

    async def __aenter__(self) -> _RecordingLifecycle:
        self.events.append('base enter')
        return await super().__aenter__()

    async def __aexit__(self, *args: Any) -> bool | None:
        self.events.append('base exit')
        return await super().__aexit__(*args)


async def test_entering_the_wrapper_leaves_the_model_it_wraps_to_its_owner() -> None:
    """Exiting the model a run borrowed could close an HTTP client another run is using."""
    events: list[object] = []
    agent = Agent(_RecordingLifecycle(events), capabilities=[_Scoped(events)])
    await agent.run('Hello')
    assert events == ['enter', ('request', True), 'exit']

    events.clear()
    async with agent:
        await agent.run('Hello')
    assert events == ['base enter', 'enter', ('request', True), 'exit', 'base exit']


class _SlowExit(_ScopedModel):
    async def __aexit__(self, *args: Any) -> bool | None:
        self.events.append('exit')
        await anyio.lowlevel.checkpoint()
        self.events.append('exited')
        return None


@dataclass
class _ScopedSlowExit(AbstractCapability[Any]):
    events: list[object]

    def get_wrapper_model(self, model: Model) -> Model | None:
        return _SlowExit(model, self.events)


async def test_the_wrapper_finishes_exiting_when_its_request_is_cancelled() -> None:
    events: list[object] = []
    started = anyio.Event()

    async def wait_forever(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await anyio.sleep_forever()
        raise AssertionError('unreachable')  # pragma: no cover

    agent = Agent(FunctionModel(wait_forever), capabilities=[_ScopedSlowExit(events)])
    async with anyio.create_task_group() as tg:
        tg.start_soon(agent.run, 'Hello')
        await started.wait()
        tg.cancel_scope.cancel()
    assert events == ['enter', ('request', True), 'exit', 'exited']


class _RecordingCancel(_ScopedModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.events.append(('cancel', self.entered))
        await super().cancel_suspended_response(response)


@dataclass
class _ScopedCancel(AbstractCapability[Any]):
    events: list[object]

    def get_wrapper_model(self, model: Model) -> Model | None:
        return _RecordingCancel(model, self.events)


async def test_a_suspended_stream_the_consumer_failed_on_is_cancelled_through_the_entered_wrapper() -> None:
    events: list[object] = []
    model = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['partial'], state='suspended', provider_response_id='job', input_tokens=1, output_tokens=1
            )
        ]
    )
    agent = Agent(model, capabilities=[_ScopedCancel(events)])
    async with agent.iter('Hello') as run:
        with pytest.raises(RuntimeError, match='Consumer failed'):
            async for node in run:  # pragma: no branch
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:  # pragma: no branch
                            raise RuntimeError('Consumer failed')
    assert ('cancel', True) in events
    assert len(model.cancelled) == 1
