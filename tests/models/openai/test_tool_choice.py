"""Tests for OpenAI Chat tool_choice setting."""

from __future__ import annotations as _annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.settings import ToolChoiceValue
from pydantic_ai.usage import RequestUsage

from ...conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_tool_choice_auto_live(allow_model_requests: None, openai_api_key: str):
    """Test tool_choice='auto' allows model to decide whether to use tools."""
    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    result = await agent.run('What is 2+2?', model_settings={'tool_choice': 'auto'})
    assert result.output is not None
    assert '4' in result.output


async def test_tool_choice_required_live(allow_model_requests: None, openai_api_key: str):
    """Test tool_choice='required' forces model to use a tool."""
    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
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


async def test_tool_choice_none_live(allow_model_requests: None, openai_api_key: str):
    """Test tool_choice='none' prevents model from using function tools."""
    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'  # pragma: no cover

    result = await agent.run(
        'What is the weather in Paris? Use the tool if available.',
        model_settings={'tool_choice': 'none'},
    )
    assert result.output is not None
    tool_calls = [
        m
        for m in result.all_messages()
        if hasattr(m, 'parts')
        and any(hasattr(p, 'tool_name') and p.tool_name == 'get_weather' for p in getattr(m, 'parts', []))
    ]
    assert len(tool_calls) == 0


async def test_tool_choice_specific_live(allow_model_requests: None, openai_api_key: str):
    """Test tool_choice=['tool_name'] forces model to use the named tool."""
    from pydantic_ai import UsageLimits
    from pydantic_ai.exceptions import UsageLimitExceeded

    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
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


async def test_callable_tool_choice_multi_turn(allow_model_requests: None, openai_api_key: str):
    """Test callable tool_choice evaluated per model request across multiple turns."""
    evaluated_steps: list[int] = []

    def dynamic_tool_choice(ctx: RunContext[None]) -> ToolChoiceValue:
        evaluated_steps.append(ctx.run_step)
        if ctx.run_step == 1:
            return ['get_weather']
        elif ctx.run_step == 2:
            return ['get_forecast']
        return 'auto'

    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings={'tool_choice': dynamic_tool_choice})

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Current weather in {city}: 72°F, sunny'

    @agent.tool_plain
    def get_forecast(city: str) -> str:
        """Get the weather forecast for a city."""
        return f'Forecast for {city}: sunny for the next 3 days'

    result = await agent.run('What is the weather in Paris and what is the forecast?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in Paris and what is the forecast?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='call_lr3fVLODFt5q6v3aHfP4XsDD'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=161,
                    output_tokens=407,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 384,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-5-mini-2025-08-07',
                timestamp=datetime(2025, 12, 9, 17, 17, 51, tzinfo=timezone.utc),
                provider_name='openai',
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id='chatcmpl-CkvdPjYulTO4ZwjJb2Dh7KyLTyWSG',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='Current weather in Paris: 72°F, sunny',
                        tool_call_id='call_lr3fVLODFt5q6v3aHfP4XsDD',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_forecast', args='{"city":"Paris"}', tool_call_id='call_W8qSbpVjYPqA1FO6Ckm360h9'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=199,
                    output_tokens=88,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 64,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-5-mini-2025-08-07',
                timestamp=datetime(2025, 12, 9, 17, 17, 58, tzinfo=timezone.utc),
                provider_name='openai',
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id='chatcmpl-CkvdWAVMSzgFPrHMR8WhQuyLByQ6I',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_forecast',
                        content='Forecast for Paris: sunny for the next 3 days',
                        tool_call_id='call_W8qSbpVjYPqA1FO6Ckm360h9',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
Current weather in Paris: 72°F, sunny.

Forecast: sunny for the next 3 days.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=240,
                    output_tokens=24,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-5-mini-2025-08-07',
                timestamp=datetime(2025, 12, 9, 17, 18, 1, tzinfo=timezone.utc),
                provider_name='openai',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-CkvdZTfzXSbnF1UzQ0DQwnYkbkUVK',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
