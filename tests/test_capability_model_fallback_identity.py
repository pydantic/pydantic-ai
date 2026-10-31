from __future__ import annotations

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks, ModelFallback, ResolveModelId
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('alias', [False, True])
async def test_untried_model_instance_with_same_id_is_not_retired(alias: bool) -> None:
    primary = TestModel(custom_output_text='draft')
    backup = TestModel(custom_output_text='ready')
    assert primary.model_id == backup.model_id
    served: list[Model] = []

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        served.append(request_context.model)
        return request_context

    capabilities: list[AbstractCapability[object]] = [
        ModelFallback(backup, output_retry_threshold=1),
        Hooks(before_model_request=observe),
    ]
    if alias:
        capabilities.insert(0, ResolveModelId(lambda ctx, model_id: primary if model_id == primary.model_id else None))
    agent = Agent(primary.model_id if alias else primary, deps_type=object, capabilities=capabilities)

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert len(served) == 2
    assert served[0] is primary
    assert served[1] is backup
