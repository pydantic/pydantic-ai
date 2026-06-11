from __future__ import annotations as _annotations

import json
from typing import Any, cast

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart
from pydantic_ai.direct import model_request

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.cerebras import (
        CerebrasModel,
        CerebrasModelSettings,
        _cerebras_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.cerebras import CerebrasProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_cerebras_model_simple(allow_model_requests: None, cerebras_api_key: str):
    """Test basic Cerebras model functionality."""
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('llama-3.3-70b', provider=provider)
    agent = Agent(model=model)
    result = await agent.run('What is 2 + 2?')
    assert '4' in result.output


async def test_cerebras_disable_reasoning_setting(allow_model_requests: None, cerebras_api_key: str):
    """Test that cerebras_disable_reasoning setting is properly transformed to extra_body.

    Note: disable_reasoning is only supported on reasoning models: zai-glm-4.6 and gpt-oss-120b.
    """
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('zai-glm-4.6', provider=provider)

    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)

    text_part = cast(TextPart, response.parts[0])
    assert '4' in text_part.content


async def test_cerebras_thinking_part_survives_multiturn(
    allow_model_requests: None, cerebras_api_key: str, vcr: Cassette
):
    """A reasoning model's `ThinkingPart` survives a 2-turn round-trip on Cerebras.

    Cerebras surfaces reasoning as a decorative `ThinkingPart` (parsed from the `reasoning` field, not a
    structured item the API consumes). This locks that the turn-1 part is preserved verbatim in the message
    history across turns and replayed on the second request's wire body as the assistant `reasoning` field.
    """
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('gpt-oss-120b', provider=provider)
    agent = Agent(model=model)

    result1 = await agent.run('What is 2 + 2? Think briefly first.')
    turn1_response = next(m for m in reversed(result1.all_messages()) if isinstance(m, ModelResponse))
    turn1_thinking = [p for p in turn1_response.parts if isinstance(p, ThinkingPart)]
    assert turn1_thinking, 'expected a ThinkingPart on turn 1'

    result2 = await agent.run('Now add 3 to that.', message_history=result1.all_messages())

    # The turn-1 ThinkingPart is preserved verbatim across the round-trip.
    preserved = [
        p
        for m in result2.all_messages()
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ThinkingPart)
    ]
    assert any(p.content == turn1_thinking[0].content for p in preserved)

    # On the wire, the decorative thinking is replayed as the assistant message's `reasoning` field.
    turn2_body = json.loads(vcr.requests[1].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assistant_messages = [m for m in turn2_body['messages'] if m.get('role') == 'assistant']
    assert any(m.get('reasoning') == turn1_thinking[0].content for m in assistant_messages)


async def test_cerebras_settings_transformation():
    """Test that CerebrasModelSettings are correctly transformed to OpenAIChatModelSettings."""
    from pydantic_ai.models import ModelRequestParameters

    params = ModelRequestParameters()

    # Test with disable_reasoning
    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    transformed = _cerebras_settings_to_openai_settings(settings, params)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('disable_reasoning') is True

    # Test without disable_reasoning (should not have extra_body)
    settings_empty = CerebrasModelSettings()
    transformed_empty = _cerebras_settings_to_openai_settings(settings_empty, params)
    assert transformed_empty.get('extra_body') is None

    # Test with disable_reasoning=False
    settings_false = CerebrasModelSettings(cerebras_disable_reasoning=False)
    transformed_false = _cerebras_settings_to_openai_settings(settings_false, params)
    extra_body_false = cast(dict[str, Any], transformed_false.get('extra_body', {}))
    assert extra_body_false.get('disable_reasoning') is False
