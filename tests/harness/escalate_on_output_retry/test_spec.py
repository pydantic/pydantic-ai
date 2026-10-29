from __future__ import annotations

import pytest

from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import EscalateOnOutputRetry


class TestSpec:
    async def test_serialized_spec_roundtrip_and_custom_ids(self) -> None:
        specification = AgentSpec.model_validate(
            {
                'retries': {'output': 2},
                'capabilities': [
                    {
                        'EscalateOnOutputRetry': {
                            'id': 'escalation',
                            'levels': [{'model': 'app:fast', 'after_failures': 2}, {'model': 'app:strong'}],
                        }
                    }
                ],
            }
        )
        recovered = AgentSpec.model_validate_json(specification.model_dump_json())
        assert recovered == specification
        agent = Agent.from_spec(
            recovered,
            custom_capability_types=[EscalateOnOutputRetry],
            capabilities=[ResolveModelId(lambda ctx, model_id: TestModel(custom_output_text=model_id))],
        )
        assert (await agent.run('Answer.')).output == 'app:fast'
        schema = AgentSpec.model_json_schema_with_capabilities(custom_capability_types=[EscalateOnOutputRetry])
        assert 'after_failures' in str(schema)

    def test_unknown_level_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='Unknown escalation level fields'):
            Agent.from_spec(
                {'capabilities': [{'EscalateOnOutputRetry': {'levels': [{'model': 'test', 'retries': 2}]}}]},
                custom_capability_types=[EscalateOnOutputRetry],
            )
