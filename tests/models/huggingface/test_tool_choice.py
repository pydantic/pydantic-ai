"""Tests for HuggingFace tool_choice setting."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_tool_choice_auto_live(allow_model_requests: None, huggingface_api_key: str):
    """Test tool_choice='auto' allows model to decide whether to use tools."""
    m = HuggingFaceModel('Qwen/Qwen2.5-Coder-32B-Instruct', provider=HuggingFaceProvider(api_key=huggingface_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    result = await agent.run('What is 2+2?', model_settings={'tool_choice': 'auto'})
    assert result.output is not None
    assert '4' in result.output


async def test_tool_choice_required_live(allow_model_requests: None, huggingface_api_key: str):
    """Test tool_choice='required' forces model to use a tool."""
    m = HuggingFaceModel('Qwen/Qwen2.5-Coder-32B-Instruct', provider=HuggingFaceProvider(api_key=huggingface_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'

    result = await agent.run(
        'What is the temperature in Paris? Use the weather tool and return the temperature as a number.',
        output_type=int,
        model_settings={'tool_choice': 'required'},
    )
    assert result.output == 72


async def test_tool_choice_none_live(allow_model_requests: None, huggingface_api_key: str):
    """Test tool_choice='none' prevents model from using function tools."""
    m = HuggingFaceModel('Qwen/Qwen2.5-Coder-32B-Instruct', provider=HuggingFaceProvider(api_key=huggingface_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    result = await agent.run(
        'What is 2+2?',
        model_settings={'tool_choice': 'none'},
    )
    assert result.output is not None
    assert '4' in result.output


async def test_tool_choice_specific_live(allow_model_requests: None, huggingface_api_key: str):
    """Test tool_choice=['tool_name'] forces model to use the named tool."""
    m = HuggingFaceModel('Qwen/Qwen2.5-Coder-32B-Instruct', provider=HuggingFaceProvider(api_key=huggingface_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'

    @agent.tool_plain
    def get_time(city: str) -> str:
        """Get the current time in a city."""
        return f'The time in {city} is 3:00 PM.'  # pragma: no cover

    # Agent is forced to call get_weather even though it's unrelated to the prompt
    with pytest.raises(UsageLimitExceeded):
        await agent.run(
            'What is 2+2?',
            model_settings={'tool_choice': ['get_weather']},
            usage_limits=UsageLimits(request_limit=2),
        )
