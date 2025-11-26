from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_deepseek_model_thinking_part(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIChatModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)
    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='deepseek'),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=12,
                    output_tokens=789,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 12,
                        'reasoning_tokens': 415,
                    },
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='181d9669-2b3a-445e-bd13-2ebff2c378f6',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIChatModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    result: AgentRunResult | None = None
    async for event in agent.run_stream_events(user_prompt='How do I cross the street?'):
        if isinstance(event, AgentRunResultEvent):
            result = event.result

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='reasoning_content',
                        provider_name='deepseek',
                    ),
                    TextPart(content='Hello there! 😊 How can I help you today?'),
                ],
                usage=RequestUsage(
                    input_tokens=6,
                    output_tokens=212,
                    details={'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 6, 'reasoning_tokens': 198},
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='33be18fc-3842-486c-8c29-dd8e578f7f20',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
