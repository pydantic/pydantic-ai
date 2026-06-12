from __future__ import annotations as _annotations

import json
from typing import Any, cast

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart
from pydantic_ai._warnings import PydanticAIDeprecationWarning
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


async def test_cerebras_disable_reasoning_setting(allow_model_requests: None, cerebras_api_key: str, vcr: Cassette):
    """The deprecated `cerebras_disable_reasoning` still disables reasoning, now via `reasoning_effort='none'`.

    Cerebras deprecated `extra_body['disable_reasoning']` in favor of the standard `reasoning_effort='none'`.
    """
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('zai-glm-4.7', provider=provider)

    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
        response = await model_request(
            model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings
        )

    text_part = cast(TextPart, response.parts[0])
    assert '4' in text_part.content

    body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert body.get('reasoning_effort') == 'none'
    assert 'disable_reasoning' not in body


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


async def test_cerebras_zai_reasoning_replayed_as_think_tags(
    allow_model_requests: None, cerebras_api_key: str, vcr: Cassette
):
    """GLM replays prior reasoning inside `<think>...</think>` tags in the assistant `content`, not a `reasoning` field.

    Unlike gpt-oss (Harmony, replayed in the `reasoning` field), Cerebras GLM/Qwen require previous reasoning to
    be wrapped in `<think>` tags inside the assistant message content, so `CerebrasProvider.model_profile()` sets
    `openai_chat_send_back_thinking_parts='tags'` for `zai`. See https://inference-docs.cerebras.ai/capabilities/reasoning.
    """
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('zai-glm-4.7', provider=provider)
    agent = Agent(model=model)

    result1 = await agent.run('What is 25 * 4? Think briefly first.')
    turn1_response = next(m for m in reversed(result1.all_messages()) if isinstance(m, ModelResponse))
    turn1_thinking = [p for p in turn1_response.parts if isinstance(p, ThinkingPart)]
    assert turn1_thinking, 'expected a ThinkingPart on turn 1'

    await agent.run('Now divide that by 2.', message_history=result1.all_messages())

    turn2_body = json.loads(vcr.requests[1].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assistant_messages = [m for m in turn2_body['messages'] if m.get('role') == 'assistant']
    start_tag, end_tag = model.profile.thinking_tags
    assert any(
        start_tag in (m.get('content') or '') and end_tag in (m.get('content') or '') for m in assistant_messages
    ), 'expected prior reasoning wrapped in think tags in the assistant content'
    assert all('reasoning' not in m for m in assistant_messages), 'reasoning must not be replayed in a separate field'


async def test_cerebras_settings_transformation():
    """`CerebrasModelSettings` are transformed to `OpenAIChatModelSettings` without the deprecated `disable_reasoning`."""
    from pydantic_ai.models import ModelRequestParameters

    params = ModelRequestParameters()

    # The deprecated `cerebras_disable_reasoning=True` maps onto `reasoning_effort='none'`.
    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
        transformed = _cerebras_settings_to_openai_settings(settings, params)
    assert transformed.get('openai_reasoning_effort') == 'none'

    # An empty settings object stays empty.
    settings_empty = CerebrasModelSettings()
    transformed_empty = _cerebras_settings_to_openai_settings(settings_empty, params)
    assert 'openai_reasoning_effort' not in transformed_empty
    assert transformed_empty.get('extra_body') is None

    # `cerebras_disable_reasoning=False` is an explicit opt-out, so no disable signal is emitted.
    settings_false = CerebrasModelSettings(cerebras_disable_reasoning=False)
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
        transformed_false = _cerebras_settings_to_openai_settings(settings_false, params)
    assert 'openai_reasoning_effort' not in transformed_false

    # `cerebras_clear_thinking` is written to `extra_body['clear_thinking']` only when explicitly set.
    settings_clear = CerebrasModelSettings(cerebras_clear_thinking=False)
    transformed_clear = _cerebras_settings_to_openai_settings(settings_clear, params)
    extra_body_clear = cast(dict[str, Any], transformed_clear.get('extra_body', {}))
    assert extra_body_clear.get('clear_thinking') is False

    transformed_no_clear = _cerebras_settings_to_openai_settings(CerebrasModelSettings(), params)
    assert transformed_no_clear.get('extra_body') is None
