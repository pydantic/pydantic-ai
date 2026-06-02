from __future__ import annotations as _annotations

from typing import cast

import pytest

from pydantic_ai import Agent, ModelRequest, TextPart
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
    """Test that cerebras_disable_reasoning setting is properly transformed to reasoning_effort.

    Note: disable_reasoning is only supported on reasoning models: zai-glm-4.7 and gpt-oss-120b.
    """
    provider = CerebrasProvider(api_key=cerebras_api_key)
    model = CerebrasModel('zai-glm-4.7', provider=provider)

    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)

    text_part = cast(TextPart, response.parts[0])
    assert '4' in text_part.content


async def test_cerebras_settings_transformation():
    """Test that CerebrasModelSettings are correctly transformed to OpenAIChatModelSettings."""
    from pydantic_ai.models import ModelRequestParameters

    params = ModelRequestParameters()

    # Test with disable_reasoning → maps to reasoning_effort='none'
    settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
    transformed = _cerebras_settings_to_openai_settings(settings, params)
    assert transformed.get('openai_reasoning_effort') == 'none'

    # Test without disable_reasoning (should not set reasoning_effort)
    settings_empty = CerebrasModelSettings()
    transformed_empty = _cerebras_settings_to_openai_settings(settings_empty, params)
    assert transformed_empty.get('openai_reasoning_effort') is None

    # Test with disable_reasoning=False → maps to reasoning_effort='low'
    settings_false = CerebrasModelSettings(cerebras_disable_reasoning=False)
    transformed_false = _cerebras_settings_to_openai_settings(settings_false, params)
    assert transformed_false.get('openai_reasoning_effort') == 'low'

    # Test with disable_reasoning=False but openai_reasoning_effort already set → preserves existing value
    settings_with_effort = CerebrasModelSettings(cerebras_disable_reasoning=False, openai_reasoning_effort='high')
    transformed_with_effort = _cerebras_settings_to_openai_settings(settings_with_effort, params)
    assert transformed_with_effort.get('openai_reasoning_effort') == 'high'
