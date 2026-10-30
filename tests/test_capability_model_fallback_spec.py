from __future__ import annotations

import pytest

from pydantic_ai import Agent, AgentSpec, ModelRetry
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize(
    'threshold', [0, -1, True, 1.5, 'one', {'model': 0}, {'model': True}, {1: 2}, {'model': 'bad'}]
)
def test_invalid_output_retry_threshold_in_spec(threshold: object) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        Agent.from_spec(
            {'capabilities': [{'ModelFallback': {'models': ['test'], 'output_retry_threshold': threshold}}]}
        )


@pytest.mark.parametrize('per_model', [False, True])
@pytest.mark.parametrize('initial_model', [False, True])
async def test_output_fallback_spec_roundtrip_and_custom_ids(per_model: bool, initial_model: bool) -> None:
    policy: int | dict[str, int] = {'app:first': 2} if per_model else 2
    specification = AgentSpec.model_validate(
        {
            'model': 'app:first' if initial_model else None,
            'retries': {'output': 2},
            'capabilities': [
                {
                    'ModelFallback': {
                        'models': ['app:next'] if initial_model else ['app:first', 'app:next'],
                        'output_retry_threshold': policy,
                    }
                },
            ],
        }
    )
    recovered = AgentSpec.model_validate_json(specification.model_dump_json())
    assert recovered == specification
    agent = Agent.from_spec(
        recovered,
        capabilities=[ResolveModelId(lambda ctx, model_id: TestModel(custom_output_text=model_id))],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'app:first':
            raise ModelRetry('Use the next candidate.')
        return value

    assert (await agent.run('Answer.')).output == 'app:next'
    assert 'output_retry_threshold' in str(AgentSpec.model_json_schema_with_capabilities())
