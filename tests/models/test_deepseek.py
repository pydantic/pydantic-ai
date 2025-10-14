from __future__ import annotations as _annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
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
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
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
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIChatModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Hello') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='H', id='reasoning_content', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='mm', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' said', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' "', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Hello', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='".', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' It', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' simple', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' greeting', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wonder', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
  \n\

""",
                    provider_name='deepseek',
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='The', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' message', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' very', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' brief', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' much', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' context', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' work', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' testing', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'m", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' responsive', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' perhaps', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' new', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chatting', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' AI', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
  \n\

""",
                    provider_name='deepseek',
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' keep', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' my', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' reply', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' warm', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' inviting', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' encourage', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' further', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' conversation', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' A', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' smile', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='y', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' face', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' help', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' friendly', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Since', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' didn', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' specify', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' leave', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' open', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-ended', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' asking', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' help', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' today', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
  \n\

""",
                    provider_name='deepseek',
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='The', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tone', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cheerful', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' professional', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' -', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stiff', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' casual', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' "', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Hello', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='!"', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' feels', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' start', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Adding', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' "', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='What', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' do', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' you', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' today', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?"', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' turns', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' invitation', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rather', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' than', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mirror', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' their', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' greeting', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
  \n\

""",
                    provider_name='deepseek',
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' avoid', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assumptions', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' their', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gender', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' location', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' intent', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' since', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' zero', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' information', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' If', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' polite', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' reply', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' further', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' -', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s", provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' okay', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too', provider_name='deepseek')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.', provider_name='deepseek')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
Hmm, the user just said "Hello". It's a simple greeting but I wonder if there's more to it.  \n\

The message is very brief, so I don't have much context to work with. Maybe they're just testing if I'm responsive, or perhaps they're new to chatting with AI.  \n\

I should keep my reply warm and inviting to encourage further conversation. A smiley face would help make it friendly. Since they didn't specify a need, I'll leave it open-ended by asking how I can help them today.  \n\

The tone should be cheerful but professional - not too stiff, not too casual. "Hello there!" feels right for a start. Adding "What can I do for you today?" turns it into an invitation rather than just mirroring their greeting.  \n\

I'll avoid assumptions about their gender, location, or intent since there's zero information. If they're just being polite, they might not reply further - and that's okay too.\
""",
                    id='reasoning_content',
                    provider_name='deepseek',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Hello'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' there')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' 😊')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' How')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' I')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' help')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' today')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='?')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='')),
            PartEndEvent(index=1, part=TextPart(content='Hello there! 😊 How can I help you today?')),
        ]
    )
