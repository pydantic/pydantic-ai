"""Spec model selection is tested with synthetic failures, independent of provider APIs."""

from collections.abc import AsyncIterator
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from pydantic_ai import Agent, AgentSpec, ModelHTTPError
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('model_names', [['test'], ['test', 'test']])
def test_spec_model_list(model_names: list[str]):
    agent = Agent.from_spec(AgentSpec(model=model_names))
    assert isinstance(agent.model, FallbackModel)
    assert len(agent.model.models) == len(model_names)
    assert all(isinstance(model, TestModel) for model in agent.model.models)


@pytest.mark.parametrize('model', [[], [None], [123], [['test']]])
def test_spec_invalid_model_list(model: object):
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({'model': model})


@pytest.mark.parametrize('fmt', ['yaml', 'json'])
def test_spec_model_list_roundtrip(tmp_path: Path, fmt: str):
    path = tmp_path / f'agent.{fmt}'
    spec = AgentSpec(model=['test', 'test'])
    spec.to_file(path)
    assert AgentSpec.from_file(path).model == spec.model
    agent = Agent.from_file(path)
    assert isinstance(agent.model, FallbackModel)
    assert len(agent.model.models) == 2


@pytest.mark.parametrize('surface', ['construct', 'run', 'override'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('failures', [0, 1, 2])
async def test_spec_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
    surface: Literal['construct', 'run', 'override'],
    stream: bool,
    failures: int,
):
    attempts: list[str] = []

    def make_model(index: int) -> FunctionModel:
        name = f'model-{index}'

        def output(info: AgentInfo) -> str:
            attempts.append(name)
            assert info.model_settings == {'temperature': 0.2}
            if index < failures:
                raise ModelHTTPError(503, name)
            return f'Response from {name}'

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(output(info))])

        async def respond_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield output(info)

        return FunctionModel(respond, stream_function=respond_stream, model_name=name)

    models = {f'model-{index}': make_model(index) for index in range(3)}
    monkeypatch.setattr('pydantic_ai.models.fallback.infer_model', models.__getitem__)
    spec = AgentSpec(model=list(models), model_settings={'temperature': 0.2})
    agent = Agent.from_spec(spec) if surface == 'construct' else Agent('test')

    with agent.override(spec=spec) if surface == 'override' else nullcontext():
        run_spec = spec if surface == 'run' else None
        if stream:
            async with agent.run_stream('hello', spec=run_spec) as result:
                text = await result.get_output()
        else:
            text = (await agent.run('hello', spec=run_spec)).output

    assert text == f'Response from model-{failures}'
    assert attempts == list(models)[: failures + 1]


@pytest.mark.parametrize('surface', ['construct', 'run', 'override'])
async def test_spec_model_list_explicit_model_wins(surface: Literal['construct', 'run', 'override']):
    # Invalid provider names prove the overridden list is never instantiated.
    spec = AgentSpec(model=['unknown:primary', 'unknown:backup'])
    model = TestModel(custom_output_text='Explicit model')
    if surface == 'construct':
        agent = Agent.from_spec(spec, model=model)
        result = await agent.run('hello')
    elif surface == 'run':
        result = await Agent('test').run('hello', spec=spec, model=model)
    else:
        agent = Agent('test')
        with agent.override(spec=spec, model=model):
            result = await agent.run('hello')
        assert (await agent.run('hello')).output == 'success (no tool calls)'
    assert result.output == 'Explicit model'


@pytest.mark.parametrize('spec_model', ['unknown:primary', ['unknown:primary', 'unknown:backup']])
async def test_spec_model_outer_override_wins(spec_model: str | list[str]):
    agent = Agent()
    with agent.override(model=TestModel(custom_output_text='Override model')):
        result = await agent.run('hello', spec=AgentSpec(model=spec_model))
    assert result.output == 'Override model'


@pytest.mark.parametrize('api_error', [False, True])
async def test_spec_model_fallback_errors(monkeypatch: pytest.MonkeyPatch, api_error: bool):
    attempts = 0
    error = ModelHTTPError(503, 'test') if api_error else ValueError('Invalid request')

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise error

    model = FunctionModel(fail)
    models = {'primary': model, 'backup': model}
    monkeypatch.setattr('pydantic_ai.models.fallback.infer_model', models.__getitem__)
    agent = Agent.from_spec({'model': ['primary', 'backup']})

    with pytest.raises(FallbackExceptionGroup if api_error else ValueError) as exc_info:
        await agent.run('hello')

    if api_error:
        assert isinstance(exc_info.value, FallbackExceptionGroup)
        assert exc_info.value.exceptions == (error, error)
        assert attempts == 2
    else:
        assert exc_info.value is error
        assert attempts == 1
