from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_deepseek_model_thinking_part(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)
    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ThinkingPart(content=IsStr()), TextPart(content=IsStr())],
                usage=Usage(
                    requests=1,
                    request_tokens=12,
                    response_tokens=789,
                    total_tokens=801,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 12,
                        'reasoning_tokens': 415,
                        'cached_tokens': 0,
                    },
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                vendor_id='181d9669-2b3a-445e-bd13-2ebff2c378f6',
            ),
        ]
    )
