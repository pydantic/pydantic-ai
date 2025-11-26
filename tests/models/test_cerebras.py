from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from pydantic_ai import Agent, ModelRequest, TextPart
from pydantic_ai.direct import model_request

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.cerebras import (
        CerebrasModel,
        CerebrasModelSettings,
        cerebras_settings_to_openai_settings,
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


async def test_cerebras_settings_transformation():
    """Test that CerebrasModelSettings are correctly transformed to OpenAIChatModelSettings."""
    # Test with disable_reasoning
    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    transformed = cerebras_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('disable_reasoning') is True

    # Test without disable_reasoning (should not have extra_body)
    settings_empty = CerebrasModelSettings()
    transformed_empty = cerebras_settings_to_openai_settings(settings_empty)
    assert transformed_empty.get('extra_body') is None

    # Test with disable_reasoning=False
    settings_false = CerebrasModelSettings(cerebras_disable_reasoning=False)
    transformed_false = cerebras_settings_to_openai_settings(settings_false)
    extra_body_false = cast(dict[str, Any], transformed_false.get('extra_body', {}))
    assert extra_body_false.get('disable_reasoning') is False
