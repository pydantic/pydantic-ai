"""OpenAI Chat Completions API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the OpenAI Chat
Completions API (OpenAIChatModel). Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live OpenAI API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits

from ...conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# =============================================================================
# Tool definitions (declared at module level, not using decorator pattern)
# =============================================================================


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'Sunny, 22C in {city}'


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f'14:30 in {timezone}'


def get_population(city: str) -> str:
    """Get the population of a city."""
    return f'{city} has 1 million people'


# =============================================================================
# Helper functions for direct model requests
# =============================================================================


def make_tool_def(name: str, description: str, param_name: str) -> ToolDefinition:
    """Create a ToolDefinition for testing direct model requests."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            'properties': {param_name: {'type': 'string'}},
            'required': [param_name],
        },
    )


# =============================================================================
# Structured output model
# =============================================================================


class CityInfo(BaseModel):
    """Information about a city."""

    city: str
    summary: str


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_model(openai_api_key: str) -> OpenAIChatModel:
    """Create an OpenAI Chat model for testing."""
    return OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(openai_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Paris?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Paris"}',
                            tool_call_id='call_osdBmrDQSV2FQDgMWgTI1U99',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=132,
                        output_tokens=23,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 33, 56, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZaW3DIK7iENBh1A4ecFDEnGVE43',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='call_osdBmrDQSV2FQDgMWgTI1U99',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny and 22°C in Paris (about 72°F). Would you like a forecast for later today or tomorrow, or clothing suggestions?"
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=167,
                        output_tokens=102,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 64,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 33, 58, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CoZaYdZpE7ZVU3UWlJwWNhBZJd2Ne',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, openai_model: OpenAIChatModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(openai_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Say hello in one word', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello')],
                    usage=RequestUsage(
                        input_tokens=131,
                        output_tokens=74,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 64,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 1, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CoZabQb5eSSIgt8iYFWmfIK9CL7g5',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(openai_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=900),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in London?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"London"}',
                            tool_call_id='call_MQMjozk2KGLrHt64joNHmnbo',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=132,
                        output_tokens=23,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 33, 56, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZaWp8jwPSRAtOew0QH3yjxsAxlQ',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id='call_MQMjozk2KGLrHt64joNHmnbo',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny and 22°C (≈72°F) in London right now. Would you like an hourly forecast, a multi-day forecast, or the temperature in a different unit?"
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=167,
                        output_tokens=173,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 128,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 33, 58, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CoZaYgRgJm408QiOgW5mkaWJwsyld',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run('Get weather for Tokyo and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Get weather for Tokyo and summarize', timestamp=IsNow(tz=timezone.utc))
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Tokyo"}',
                            tool_call_id='call_bsKbp47zxZp3ICsyN0RbeyoW',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=155,
                        output_tokens=279,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 256,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 3, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZadLAE1IuauBa64ehbqf8uN20Kt',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id='call_bsKbp47zxZp3ICsyN0RbeyoW',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"Sunny, 22°C in Tokyo — mild, pleasant conditions."}',
                            tool_call_id='call_Hqyiv1bioQTfoV7zSC8kDJXF',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=190,
                        output_tokens=423,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 384,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 10, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZakjjqGf8bQS7rR5eR5bdz3kAcR',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_Hqyiv1bioQTfoV7zSC8kDJXF',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceNone:
    """Tests for tool_choice='none' and tool_choice=[].

    When tool_choice is 'none' or [], function tools are disabled but output tools remain.
    """

    async def test_none_prevents_function_tool_calls(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(openai_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Berlin?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I can't access live weather data right now, so I can't fetch the current conditions for Berlin. I can either:

- Tell you how to check live weather quickly (websites/apps to use), or  \n\
- Give a quick, general summary of typical December weather in Berlin and packing tips, or  \n\
- Provide a sample (hypothetical) 3–7 day forecast if that helps.

Which would you prefer?

If you want to check right now yourself, try: search "Berlin weather" or open a weather app (Weather.com, AccuWeather, Google Weather, or the German Meteorological Service). They'll give current temperature, precipitation, wind, and an hourly forecast.\
"""
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=132,
                        output_tokens=596,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 448,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 36, 31, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CoZd1e8Rvv1doNZ2ZzG1u7GcKDhlc',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(openai_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Rome?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I can't fetch live weather data right now. I can help in other ways — for example I can tell you how to check current conditions, give a typical climate summary for this time of year, or give packing/clothing advice.

Quick options:
- To get the current weather now: check a phone weather app or a weather website (e.g., Weather.com, AccuWeather, or a local Italian weather site), or search "Rome weather" in your browser. You can also check METAR/TAF reports for Rome Fiumicino (if you want aviation-style observations).
- Typical conditions for Rome in mid-December: average daytime high about 12–14°C (54–57°F), nighttime low about 3–6°C (37–43°F). December is one of Rome's wetter months — expect a fair chance of rain and generally mild but cool temperatures; occasional chilly days or brief cold snaps are possible.
- If you tell me a specific date/time or what you're planning to do (walking around, sightseeing, evening out), I can give tailored clothing and activity advice based on typical conditions.

Which would you like?\
"""
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=132,
                        output_tokens=626,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 384,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 12, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CoZam6Lt5vwZS3o3n88gp1WceEWdj',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, openai_model: OpenAIChatModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run('Tell me about Madrid', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Tell me about Madrid', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Madrid","summary":"Provide a comprehensive overview of Madrid covering history, main attractions, neighborhoods, culture (food, nightlife), practical travel tips (transport, climate, safety), recommended day trips, best times to visit, and a brief 2-3 day sample itinerary."}',
                            tool_call_id='call_PJsoYMXC76tTcacabq53EYJz',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=153,
                        output_tokens=460,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 384,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 33, 56, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZaWOmBfGgomJQD0U80frUse26xP',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_PJsoYMXC76tTcacabq53EYJz',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceRequired:
    """Tests for tool_choice='required'.

    When tool_choice is 'required', the model must use a function tool.
    Output tools are NOT included - this is for direct model requests.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_required_forces_tool_use(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await openai_model.request(
            [ModelRequest.user_text_prompt('Hello')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"San Francisco"}',
                    tool_call_id='call_WTy91MpdXN710pJP5OAQ3Z8F',
                )
            ]
        )

    async def test_required_with_multiple_tools(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await openai_model.request(
            [ModelRequest.user_text_prompt('Give me any information')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time', args='{"timezone":"UTC"}', tool_call_id='call_psgIVKkXUkJ7pe56K1cAqevf'
                )
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='call_QtkIasmtmp9NVXfkl5yZ2q8D'
                )
            ]
        )

    async def test_multiple_tools_in_list(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model can use any tool from the specified list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time', args='{"timezone":"Asia/Tokyo"}', tool_call_id='call_HoSpoWg3SKnEkk8jR0XDhdtG'
                )
            ]
        )

    async def test_excluded_tool_not_called(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Tools not in the list are not called."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_model.request(
            [ModelRequest.user_text_prompt('What is the population of London?')],
            settings,
            params,
        )

        # Model should use get_weather since get_population is excluded
        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_eQX51ENP193UA9gct47s7z94'
                )
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, openai_model: OpenAIChatModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            openai_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}

        result = await agent.run('Get weather for Sydney and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Get weather for Sydney and summarize', timestamp=IsNow(tz=timezone.utc))
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Sydney"}',
                            tool_call_id='call_JY92QhFZQO8v2EdWhmIiTPe0',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=198,
                        output_tokens=151,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 128,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 14, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZaoKHF31NzlfCUxAahssyn6EYLh',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id='call_JY92QhFZQO8v2EdWhmIiTPe0',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"Current weather in Sydney: Sunny and 22°C (about 72°F). Conditions are clear and mild — good for outdoor activities. No rain reported; bring sunglasses and sunscreen. A light layer may be useful in the morning/evening."}',
                            tool_call_id='call_EE5gnQgZnTUt0070ldKe0TiU',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=233,
                        output_tokens=274,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 192,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 18, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZasmklX9zBzhV2Lrp0eQxF0D693',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_EE5gnQgZnTUt0070ldKe0TiU',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, openai_model: OpenAIChatModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            openai_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_population'])}

        result = await agent.run('Get weather and population for Chicago', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather and population for Chicago', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city": "Chicago"}',
                            tool_call_id='call_e7n8N6sMFfH9gMr8fu8f2rEJ',
                        ),
                        ToolCallPart(
                            tool_name='get_population',
                            args='{"city": "Chicago"}',
                            tool_call_id='call_7wdreEMf0P53pgbm59En7h6z',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=198,
                        output_tokens=245,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 192,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 13, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZanoessHVhxuU6goEwMlm0lFEim',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Chicago',
                            tool_call_id='call_e7n8N6sMFfH9gMr8fu8f2rEJ',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_population',
                            content='Chicago has 1 million people',
                            tool_call_id='call_7wdreEMf0P53pgbm59En7h6z',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Chicago","summary":"Weather: Sunny, 22°C. Population: 1,000,000 (as returned by population lookup). Note: that population figure appears lower than commonly reported — Chicago\'s city population is often cited around 2.7 million; I can fetch an official/updated estimate if you want."}',
                            tool_call_id='call_clHbjHdWUdK5teJImsN4bEwq',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=275,
                        output_tokens=470,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 384,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 19, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZatTmaIf6sB0pAJ8RDKtB03MjK9',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_clHbjHdWUdK5teJImsN4bEwq',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestNoFunctionTools:
    """Tests for scenarios without function tools.

    These tests verify tool_choice behavior when only output tools exist.
    """

    async def test_auto_with_only_output_tools(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(openai_model, output_type=CityInfo)

        result = await agent.run('Tell me about New York')

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Tell me about New York', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"New York","summary":"New York City: a major global city of about 8.5–8.8 million people, made of five boroughs (Manhattan, Brooklyn, Queens, The Bronx, Staten Island); famous for finance, culture, arts, museums, Broadway, diverse neighborhoods, and landmarks such as the Statue of Liberty, Times Square, Central Park, and the Empire State Building."}',
                            tool_call_id='call_AFfbcyzj6Jzmm7t45QhY3dlG',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=132,
                        output_tokens=871,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 768,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 30, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZb41lVNQ0rc8INkTCj81dRNiXph',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_AFfbcyzj6Jzmm7t45QhY3dlG',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(openai_model, output_type=CityInfo)
        settings: ModelSettings = {'tool_choice': 'none'}

        # No warning when there are no function tools to disable
        result = await agent.run('Tell me about Boston', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Tell me about Boston', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Boston","summary":"Boston, Massachusetts is one of the oldest U.S. cities (founded 1630), a global center for education, medicine, finance, and history. Key features include the Freedom Trail, Fenway Park, numerous museums, world-class universities (Harvard, MIT nearby), diverse neighborhoods (Back Bay, Beacon Hill, North End), robust public transit (MBTA), and a four-season climate with cold winters and warm summers."}',
                            tool_call_id='call_oxEsmAEcPV6iHds0avBuYHFa',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=131,
                        output_tokens=305,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 192,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 34, 27, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZb1mzUtZO6y0ixgbgjduUJc2c9q',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_oxEsmAEcPV6iHds0avBuYHFa',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestTextAndStructuredUnion:
    """Tests with union output types (str | BaseModel).

    When output_type is a union including str, allow_text_output is True.
    """

    async def test_auto_with_union_output(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(openai_model, output_type=str | CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Get weather for Miami and describe it',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Miami and describe it', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Miami"}',
                            tool_call_id='call_jzHuLY7fZUKhkSa6FRGVVF1k',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=156,
                        output_tokens=87,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 64,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 54, 15, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZuBMyr606UI7CtVgFAdR1l0r1LH',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id='call_jzHuLY7fZUKhkSa6FRGVVF1k',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Miami","summary":"Sunny, 22°C (≈72°F). Clear skies and pleasant temperatures — ideal for outdoor activities; bring sunscreen and sunglasses. Light layers are fine in the morning/evening."}',
                            tool_call_id='call_5O6uTlakPFhdSjQ7TY1TxGPX',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=191,
                        output_tokens=383,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 320,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 18, 54, 19, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CoZuF90X1Kr6VgAwVnrKfP6wEaAfA',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_5O6uTlakPFhdSjQ7TY1TxGPX',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, openai_model: OpenAIChatModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(openai_model, output_type=str | CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run(
                'Describe Seattle briefly',
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=900),
            )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Describe Seattle briefly',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""Seattle is a major city in the Pacific Northwest on the shores of Puget Sound, Washington. Known as the "Emerald City," it's famous for landmarks like the Space Needle and Pike Place Market, and for views of Mount Rainier on clear days. Seattle is a tech and innovation hub (home to Amazon and near Microsoft), with a strong coffee culture, vibrant music and arts scenes, and abundant outdoor recreation from city parks to nearby mountains and waterways. The climate is mild with cool, wet winters and relatively dry summers."""
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=152,
                        output_tokens=179,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 64,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 1, 1, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-Coa0jZzbmqVKd3uxpGcbM0vwK6g95',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )
