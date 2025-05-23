import os

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings

pytestmark = [pytest.mark.anyio()]


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_labels(allow_model_requests: None) -> None:
    m = GeminiModel('gemini-2.0-flash', provider='google-vertex')
    agent = Agent(m)

    result = await agent.run(
        'What is the capital of France?',
        model_settings=GeminiModelSettings(gemini_labels={'environment': 'test', 'team': 'analytics'}),
    )
    assert result.output == snapshot('The capital of France is **Paris**.\n')
