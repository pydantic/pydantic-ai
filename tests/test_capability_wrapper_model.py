"""The `get_wrapper_model` capability hook: which model it wraps, in what order, and when it's skipped."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks, ModelFallback, WrapperCapability
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


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

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self.calls.append(f'{self.name} -> {_describe(self.wrapped)}')
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream


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


@pytest.mark.parametrize('stream', [False, True])
async def test_the_wrapper_applies_to_each_attempt(stream: bool) -> None:
    """Each attempt is wrapped around the model serving it, including the one `ModelFallback` moved to."""
    calls: list[str] = []

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelAPIError('primary', 'down')

    async def fail_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelAPIError('primary', 'down')
        yield  # pragma: no cover

    primary = FunctionModel(fail, stream_function=fail_stream, model_name='primary')
    backup = TestModel(custom_output_text='backup')
    agent = Agent(primary, capabilities=[_Wrap('wrapper', calls), ModelFallback(backup)])

    if stream:
        async with agent.run_stream('Hello') as result:
            assert await result.get_output() == 'backup'
    else:
        assert (await agent.run('Hello')).output == 'backup'
    assert calls == ['wrapper -> primary', 'wrapper -> test']
