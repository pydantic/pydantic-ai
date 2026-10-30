from __future__ import annotations

import warnings
from typing import Literal

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    CombinedCapability,
    Hooks,
    ModelFallback,
    ResolveModelId,
    SelectModel,
    WrapperCapability,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('source', ['agent', 'wrapped', 'group', 'runtime', 'factory', 'run_spec', 'spec_capability'])
@pytest.mark.parametrize('threshold', [1, {'test:chosen': 1, 'chosen': 2}])
async def test_explicit_run_model_warns_before_output_is_rejected(
    source: Literal['agent', 'wrapped', 'group', 'runtime', 'factory', 'run_spec', 'spec_capability'],
    threshold: int | dict[str, int],
) -> None:
    backup = TestModel(model_name='backup', custom_output_text='backup')
    fallback = ModelFallback[object](backup, output_retry_threshold=threshold)
    chosen = TestModel(model_name='chosen', custom_output_text='chosen')
    capabilities: list[AgentCapability[object]] = [fallback]
    run_capabilities: list[AgentCapability[object]] = []
    explicit_model: Model | None = chosen
    spec: dict[str, object] | None = None
    expected_model_id = chosen.model_id
    if source == 'wrapped':
        capabilities = [WrapperCapability(fallback)]
    elif source == 'group':
        children: list[AbstractCapability[object]] = [fallback]
        capabilities = [WrapperCapability(CombinedCapability(children))]
    elif source == 'runtime':
        capabilities = []
        run_capabilities = [fallback]
    elif source == 'factory':
        capabilities = []
        run_capabilities = [lambda ctx: fallback]
    elif source in {'run_spec', 'spec_capability'}:
        models: dict[str, Model] = {'chosen': chosen, 'backup': backup}
        capabilities = [ResolveModelId(lambda ctx, model_id: models.get(model_id))]
        spec = {'model': 'chosen'}
        if source == 'run_spec':
            capabilities.append(fallback)
        else:
            spec['capabilities'] = [{'ModelFallback': {'models': ['backup'], 'output_retry_threshold': threshold}}]
        explicit_model = None
        expected_model_id = 'chosen'
    else:
        assert source == 'agent'
    hooks = Hooks[object]()

    @hooks.on.before_model_request
    def check_warning(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        assert len(emitted) == 1
        return request_context

    capabilities.append(hooks)
    agent = Agent(TestModel(custom_output_text='default'), deps_type=object, capabilities=capabilities)

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        result = await agent.run('Answer.', model=explicit_model, spec=spec, capabilities=run_capabilities)

    assert result.output == 'chosen'
    assert len(emitted) == 1
    assert emitted[0].filename == __file__
    message = str(emitted[0].message)
    assert f'output_retry_threshold={threshold!r}' in message
    assert f'`model` {expected_model_id!r}' in message


@pytest.mark.parametrize('threshold', [None, {}])
async def test_request_only_fallback_does_not_warn(threshold: dict[str, int] | None) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=threshold)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'

    assert emitted == []


@pytest.mark.parametrize('source', ['override', 'selector'])
async def test_override_and_capability_composition_do_not_warn(source: Literal['override', 'selector']) -> None:
    chosen = TestModel(custom_output_text='chosen')
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    if source == 'selector':
        capabilities.append(SelectModel(lambda ctx: chosen))
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        if source == 'override':
            with agent.override(model=chosen):
                result = await agent.run('Answer.', model=TestModel())
        else:
            result = await agent.run('Answer.')

    assert result.output == 'chosen'
    assert emitted == []


async def test_warning_does_not_query_model_contributions(mocker: MockerFixture) -> None:
    fallback = ModelFallback[object](TestModel(), output_retry_threshold=1)
    getter = mocker.patch.object(fallback, 'get_model', wraps=fallback.get_model)
    capabilities: list[AbstractCapability[object]] = [fallback]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)
    getter.reset_mock()

    with pytest.warns(UserWarning, match='output switching.*explicit run'):
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'

    getter.assert_not_called()


@pytest.mark.parametrize('entrypoint', ['iter', 'run_stream'])
async def test_warning_points_to_async_caller(entrypoint: Literal['iter', 'run_stream']) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)
    chosen = TestModel(custom_output_text='chosen')

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        if entrypoint == 'iter':
            async with agent.iter('Answer.', model=chosen):
                pass
        else:
            assert entrypoint == 'run_stream'
            async with agent.run_stream('Answer.', model=chosen) as result:
                assert await result.get_output() == 'chosen'

    assert len(emitted) == 1
    assert emitted[0].filename == __file__


def test_warning_points_to_sync_caller() -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        assert agent.run_sync('Answer.', model=TestModel(custom_output_text='chosen')).output == 'chosen'

    assert len(emitted) == 1
    assert emitted[0].filename == __file__


async def test_warning_without_frame_inspection(mocker: MockerFixture) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), name='fallback_warning', deps_type=object, capabilities=capabilities)
    mocker.patch('pydantic_ai._warnings.inspect.currentframe', return_value=None)

    with pytest.warns(UserWarning, match='output switching.*explicit run'):
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'
