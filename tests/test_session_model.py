"""Interaction lifetimes and isolation need instrumented resources, not transport recordings."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel


@pytest.mark.parametrize('wrapper', ['none', 'wrapper', 'fallback'])
async def test_model_interaction_is_isolated_and_shared_across_runs(wrapper: Literal['none', 'wrapper', 'fallback']):
    opened: list[int] = []
    closed: list[int] = []
    entries: list[str] = []

    class SessionModel(TestModel):
        async def __aenter__(self):
            entries.append('enter')
            return self

        async def __aexit__(self, *args: Any):
            entries.append('exit')

        @asynccontextmanager
        async def open_session(self) -> AsyncGenerator[Model]:
            session_id = len(opened)
            opened.append(session_id)
            requests = 0

            def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                nonlocal requests
                assert session_id not in closed
                requests += 1
                return ModelResponse([TextPart(f'{session_id}:{requests}')])

            async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncGenerator[str]:
                part = respond(messages, info).parts[0]
                assert isinstance(part, TextPart)
                yield part.content

            try:
                yield FunctionModel(respond, stream_function=stream)
            finally:
                closed.append(session_id)

    original = SessionModel()
    model: Model = original
    if wrapper == 'wrapper':
        model = WrapperModel(original)
    elif wrapper == 'fallback':
        model = FallbackModel(original, TestModel())
    agent = Agent(model)

    async with agent.session() as first:
        assert (await first.run('one')).output == '0:1'
        async with agent.session() as second:
            assert (await second.run('two')).output == '1:1'
            assert closed == []
        assert closed == [1]
        async with first.run_stream_events('three') as events:
            async for _ in events:
                pass
            assert events.result is not None
            assert events.result.output == '0:2'
        assert (await first.run('four')).output == '0:3'
    assert opened == [0, 1]
    assert closed == [1, 0]
    # FallbackModel intentionally reference-counts shared client entry.
    assert entries == (['enter', 'exit'] if wrapper == 'fallback' else ['enter', 'enter', 'exit', 'exit'])
    if isinstance(model, WrapperModel):
        assert model.wrapped is original
    elif isinstance(model, FallbackModel):
        assert model.models[0] is original


async def test_implicit_model_interaction_does_not_change_client_ownership():
    events: list[str] = []

    class SessionModel(TestModel):
        async def __aenter__(self):
            events.append('client enter')
            return self

        async def __aexit__(self, *args: Any):
            events.append('client exit')

        @asynccontextmanager
        async def open_session(self) -> AsyncGenerator[Model]:
            events.append('interaction enter')
            try:
                yield TestModel(custom_output_text='bound')
            finally:
                events.append('interaction exit')

    agent = Agent(SessionModel())
    assert (await agent.run('first')).output == 'bound'
    assert (await agent.run('second')).output == 'bound'
    assert events == ['interaction enter', 'interaction exit'] * 2
    async with agent:
        assert (await agent.run('third')).output == 'bound'
    assert events[4:] == ['client enter', 'interaction enter', 'interaction exit', 'client exit']


async def test_dynamic_selection_reuses_bound_interaction():
    opened: list[str] = []
    selection_models: list[Model | None] = []

    class SessionModel(TestModel):
        @asynccontextmanager
        async def open_session(self) -> AsyncGenerator[Model]:
            opened.append('open')
            yield TestModel(custom_output_text='bound')

    model = SessionModel()

    class Select(AbstractCapability[object]):
        def get_model(self):
            def select(ctx: ModelSelectionContext[object]) -> Model:
                selection_models.append(ctx.model)
                return model

            return select

    agent = Agent(capabilities=[Select()])
    async with agent.session() as session:
        assert (await session.run('first')).output == 'bound'
        assert (await session.run('second')).output == 'bound'
    assert opened == ['open']
    assert len(selection_models) == 2
