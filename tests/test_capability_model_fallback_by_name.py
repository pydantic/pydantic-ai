from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Hooks, ModelFallback
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('threshold', [1, {'test:test': 1}])
async def test_inferred_by_name_default_switches_after_output_retry(threshold: int | dict[str, int]) -> None:
    selected: list[str] = []

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        selected.append(ctx.model_id or ctx.model.model_id)
        return request_context

    agent = Agent(
        'test',
        deps_type=object,
        capabilities=[
            ModelFallback(TestModel(custom_output_text='ready', model_name='backup'), output_retry_threshold=threshold),
            Hooks(before_model_request=observe),
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value != 'ready':
            raise ModelRetry('Use the backup.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert selected == ['test:test', 'test:backup']
