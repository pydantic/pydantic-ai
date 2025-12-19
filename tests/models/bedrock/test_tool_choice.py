"""Tests for Bedrock tool_choice setting."""

from __future__ import annotations as _annotations

import os

import pytest

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def bedrock_api_key():
    return os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'test-api-key')


async def test_tool_choice_auto_live(allow_model_requests: None, bedrock_api_key: str):
    """Test tool_choice='auto' allows model to decide whether to use tools."""
    provider = BedrockProvider(api_key=bedrock_api_key)
    m = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=provider)
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    # Simple question that doesn't need tools - model should respond directly
    result = await agent.run('What is 2+2?', model_settings={'tool_choice': 'auto'})
    assert result.output is not None
    assert '4' in result.output


async def test_tool_choice_required_live(allow_model_requests: None, bedrock_api_key: str):
    """Test tool_choice='required' forces model to use a tool."""
    provider = BedrockProvider(api_key=bedrock_api_key)
    m = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=provider)
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'

    # Use output_type=int to force structured output and clean termination
    result = await agent.run(
        'What is the temperature in Paris? Use the weather tool and return the temperature as a number.',
        output_type=int,
        model_settings={'tool_choice': 'required'},
    )
    assert result.output == 72


async def test_tool_choice_none_live(allow_model_requests: None, bedrock_api_key: str):
    """Test tool_choice='none' prevents model from using function tools."""
    provider = BedrockProvider(api_key=bedrock_api_key)
    m = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=provider)
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    # With tool_choice='none', model cannot use tools and must respond with text
    result = await agent.run(
        'What is 2+2?',
        model_settings={'tool_choice': 'none'},
    )
    assert result.output is not None
    assert '4' in result.output


async def test_tool_choice_specific_live(allow_model_requests: None, bedrock_api_key: str):
    """Test tool_choice=['tool_name'] forces model to use the named tool."""
    provider = BedrockProvider(api_key=bedrock_api_key)
    m = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=provider)
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
