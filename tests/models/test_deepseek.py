from __future__ import annotations as _annotations

from typing import Any

import pytest
from dirty_equals import IsListOrTuple
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
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


async def test_deepseek_model_thinking_stream(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Hello') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        IsListOrTuple(
            positions={
                0: PartStartEvent(index=0, part=ThinkingPart(content='H')),
                1: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='mm')),
                2: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
                198: PartStartEvent(index=1, part=TextPart(content='Hello')),
                199: FinalResultEvent(tool_name=None, tool_call_id=None),
                200: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' there')),
                201: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            },
            length=210,
        )
    )
