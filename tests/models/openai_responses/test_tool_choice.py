"""OpenAI Responses API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the OpenAI Responses
API (OpenAIResponsesModel). Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live OpenAI API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits

from ...conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
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
def openai_responses_model(openai_api_key: str) -> OpenAIResponsesModel:
    """Create an OpenAI Responses model for testing."""
    return OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Paris?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0098495564d87959006945ad69e2788195a88242375e92e1dd',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Paris"}',
                            tool_call_id='call_xPQM0uDwo0Vtqsu6P38OJgX6',
                            id='fc_0098495564d87959006945ad6adb9c819589ca1aa682b00807',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=84, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 17, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='call_xPQM0uDwo0Vtqsu6P38OJgX6',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny in Paris with a temperature of 22°C.",
                            id='msg_0098495564d87959006945ad6c14108195979a0f5cc2f73449',
                        )
                    ],
                    usage=RequestUsage(input_tokens=155, output_tokens=16, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 19, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Say hello in one word',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0593209bf6ea67fc006945ad6d6c2c8195b5e3489f68f680ed',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        TextPart(content='Hello', id='msg_0593209bf6ea67fc006945ad6ec3188195955ff420ad7c6518'),
                    ],
                    usage=RequestUsage(input_tokens=49, output_tokens=71, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 21, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in London?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0debae64eefe64ca006945ad6fb554819d860cda2b51f59f38',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"London"}',
                            tool_call_id='call_RHlw05OaMVowjRif4li7Cixu',
                            id='fc_0debae64eefe64ca006945ad71478c819d883def68dc0b7387',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=84, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 23, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id='call_RHlw05OaMVowjRif4li7Cixu',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny in London and about 22°C.",
                            id='msg_0debae64eefe64ca006945ad727d3c819daf303010091b94e4',
                        )
                    ],
                    usage=RequestUsage(input_tokens=176, output_tokens=14, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 25, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run('Get weather for Tokyo and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Tokyo and summarize',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_04d4446a114c6f4a006945ad73e0cc8192aa6ae0d717242cda',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Tokyo"}',
                            tool_call_id='call_feBPBYdjmBHdDh4zMVgTUCg7',
                            id='fc_04d4446a114c6f4a006945ad765d9c81929c7d63cce482bab1',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=73, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 27, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id='call_feBPBYdjmBHdDh4zMVgTUCg7',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_04d4446a114c6f4a006945ad7768c881928bd69a27d03681e7',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"Current weather in Tokyo: Sunny, 22°C."}',
                            tool_call_id='call_4kZueRSDVZZfXi0xGLr0cGb6',
                            id='fc_04d4446a114c6f4a006945ad7be1e8819285a05650a14dec9e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=295, output_tokens=290, details={'reasoning_tokens': 256}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 31, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_4kZueRSDVZZfXi0xGLr0cGb6',
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

    async def test_none_prevents_function_tool_calls(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Berlin?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_039a5882b50f882b006945afe42624819192ba6b0b7c201b70',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content="""\
Sorry—I can't fetch live weather data at the moment. I can, however, help with general conditions and how to get an up-to-the-minute forecast.

Quick summary for Berlin in late December:
- Typical temperatures: around 0–5 °C (low often near or below freezing at night).
- Conditions: often cloudy, damp, with rain, sleet or occasional snow; it can be windy.
- Time zone: Central European Time (CET, UTC+1) at this time of year.
- Practical advice: wear warm layers, a waterproof outer layer and closed shoes; keep a hat/gloves handy.

How to get the current weather now:
- Check your phone's weather app or search "Berlin weather" in a web search.
- Trusted sources: the German Weather Service (DWD), Weather.com, BBC Weather, or AccuWeather.

If you want, tell me whether you need the current temperature, an hourly forecast, or a 7-day outlook and I'll fetch the exact live data for you if you'd like me to.\
""",
                            id='msg_039a5882b50f882b006945afedd1cc819181c805111b252ba8',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=862, details={'reasoning_tokens': 640}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 20, 4, 51, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Rome?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0362cfe909ecac94006945aff6c3288192bf7f0cb8bf75a1fa',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content="""\
I can't fetch live weather right now, but I can tell you typical conditions for Rome this time of year and how to get a current forecast.

Right now (mid‑December) Rome's usual climate:
- Average high: about 12–14°C (54–57°F)  \n\
- Average low: about 4–7°C (39–45°F)  \n\
- Rain: December often has several rainy days; expect occasional showers or overcast skies  \n\
- Wind: generally light to moderate; occasional chilly breezes or cold fronts

Clothing/packing tip: layers (sweater + light coat), a waterproof jacket or umbrella, and a scarf for cooler nights.

If you want the actual current conditions or an hourly forecast, open your phone's weather app or search "Rome weather" on the web. If you paste the current temperature/forecast you find here, I can help interpret it and give recommendations. Do you want that?\
""",
                            id='msg_0362cfe909ecac94006945affd7d708192b50284bba0b1d2df',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=774, details={'reasoning_tokens': 576}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 20, 5, 10, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run('Tell me about Madrid', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about Madrid',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f49e2c2e5f4f700006945ad9b7e608193907cfc65989f5e82',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args="""{"city":"Madrid","summary":"Madrid is Spain's capital and largest city, located near the geographic center of the Iberian Peninsula. It's famous for world‑class art museums (Prado, Reina Sofía, Thyssen), grand boulevards (Gran Vía), historic plazas (Plaza Mayor, Puerta del Sol), the Royal Palace, large parks like El Retiro, a lively tapas and nightlife scene, and easy day‑trip access to historic towns such as Toledo and Segovia."}""",
                            tool_call_id='call_ZKcJLIGBda4NlWJ5no0xPy6d',
                            id='fc_0f49e2c2e5f4f700006945ada47e4881938c882570c14c40a2',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=71, output_tokens=566, details={'reasoning_tokens': 448}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 6, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_ZKcJLIGBda4NlWJ5no0xPy6d',
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

    async def test_required_forces_tool_use(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Hello')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_0a8ab955cd861bf0006945ada750548191a99d652183eca617',
                    signature=IsStr(),
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"San Francisco"}',
                    tool_call_id='call_TAUogdpeEGrCrNEjaghnmMoO',
                    id='fc_0a8ab955cd861bf0006945adaa137c819183ab43621a285b28',
                ),
            ]
        )

    async def test_required_with_multiple_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Give me any information')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_0f1df6a79361234d006945adab4b88819db764b555415861a7',
                    signature=IsStr(),
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_time',
                    args='{"timezone":"UTC"}',
                    tool_call_id='call_uhQDpmAxwngXvpVwszNO7bqj',
                    id='fc_0f1df6a79361234d006945adaea37c819dbe9df62aa6e6c34b',
                ),
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_00ee39221530168b006945adb11fac81958e2abf8faeaaaa44',
                    signature=IsStr(),
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"Paris, France"}',
                    tool_call_id='call_uvLqbR8iSddLL7gmBEff7JM4',
                    id='fc_00ee39221530168b006945adb37b748195b5aad06091b60df6',
                ),
            ]
        )

    async def test_multiple_tools_in_list(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can use any tool from the specified list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_08ac1b945ca54380006945adb4f78481a385c65812c2058dce',
                    signature=IsStr(),
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_time',
                    args='{"timezone":"Asia/Tokyo"}',
                    tool_call_id='call_18iOaNrHTxQiS2WUrtwpFn3L',
                    id='fc_08ac1b945ca54380006945adb6481081a3a7b6fcb63318393f',
                ),
            ]
        )

    async def test_excluded_tool_not_called(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Tools not in the list are not called."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('What is the population of London?')],
            settings,
            params,
        )

        # Model should use get_weather since get_population is excluded
        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_053b747a065d0b09006945adb79c7481a38c9c9a3de7113df6',
                    signature=IsStr(),
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"London"}',
                    tool_call_id='call_241hJ2RGL4zYBF4q7j8Jy3wh',
                    id='fc_053b747a065d0b09006945adbbacf481a3a955318160772077',
                ),
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            openai_responses_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}

        result = await agent.run('Get weather for Sydney and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Sydney and summarize',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adbce16481a3952b06cd588f2b38',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Sydney"}',
                            tool_call_id='call_VHaUOnJSRlDiWZxoTwnKXJwX',
                            id='fc_02930ea85df92c07006945adbea6e881a3ae47c2f6bf1e729e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=116, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 40, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id='call_VHaUOnJSRlDiWZxoTwnKXJwX',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adbf9b9c81a389c5865b779abc87',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adc4aa7081a39b9caa32e16dc26b',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"Sunny, 22°C in Sydney. Clear skies and mild — pleasant for outdoor activities. Recommend sunglasses and sunscreen for daytime; a light layer for cooler mornings/evenings. Would you like an hourly forecast or multi-day outlook?"}',
                            tool_call_id='call_lCfU4P0xSJSWrNBiBGpeKBdh',
                            id='fc_02930ea85df92c07006945adc4ec7081a392bfae053d0d27bb',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=284, output_tokens=389, details={'reasoning_tokens': 320}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 43, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_lCfU4P0xSJSWrNBiBGpeKBdh',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            openai_responses_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_population'])}

        result = await agent.run('Get weather and population for Chicago', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather and population for Chicago',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945adc8820881a2acfaaa6407b139f0',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Chicago"}',
                            tool_call_id='call_SLfTLXh0mXdyO00yhf7HrzOh',
                            id='fc_048b20bcdb65249e006945add14e1081a2b307ee869796c949',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=116, output_tokens=404, details={'reasoning_tokens': 384}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 52, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Chicago',
                            tool_call_id='call_SLfTLXh0mXdyO00yhf7HrzOh',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945add2de4481a2b2fe03b588cbbc9d',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_population',
                            args='{"city":"Chicago"}',
                            tool_call_id='call_4et2CYELlIo2iwfr6uyPz1rw',
                            id='fc_048b20bcdb65249e006945add3c19881a2b3dcef2c3656d282',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=539, output_tokens=20, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 2, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_population',
                            content='Chicago has 1 million people',
                            tool_call_id='call_4et2CYELlIo2iwfr6uyPz1rw',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945add56db881a2aba434418ef6cc56',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Chicago","summary":"Current weather in Chicago: Sunny, 22C. Population: 1 million people."}',
                            tool_call_id='call_IjvuT2OKlKY4fdSdZU7EZjG2',
                            id='fc_048b20bcdb65249e006945add8a91881a29aa0e8519736ba6e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=617, output_tokens=169, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 5, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_IjvuT2OKlKY4fdSdZU7EZjG2',
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

    async def test_auto_with_only_output_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo)

        result = await agent.run('Tell me about New York')

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about New York',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0e706bda8f41fb38006945adda7b6c8195832e7b5bf2482346',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"New York","summary":"Overview of New York City: location, population, five boroughs, major landmarks (Statue of Liberty, Central Park, Times Square, Empire State Building, Brooklyn Bridge), culture and diversity, economy (finance, media, tech), transport (subway, airports), climate, visitor tips (best times, transit, safety, tipping)."}',
                            tool_call_id='call_ruWSF42LyQm6DD0SEzmOxG8i',
                            id='fc_0e706bda8f41fb38006945ade45a8c81958d19729cc9babf44',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=604, details={'reasoning_tokens': 512}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 10, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_ruWSF42LyQm6DD0SEzmOxG8i',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo)
        settings: ModelSettings = {'tool_choice': 'none'}

        # No warning when there are no function tools to disable
        result = await agent.run('Tell me about Boston', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about Boston',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0a3de6f9c3c28af7006945ade6e8048194bc3e6c78157ee9c8',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Boston","summary":"Boston is the capital and largest city of Massachusetts, founded in 1630 and a major center of American history, culture, education, and innovation. Key historic sites include the Freedom Trail, Boston Common, Faneuil Hall, and the North End. Prominent neighborhoods: Back Bay (Victorian brownstones and Newbury Street), Beacon Hill (historic streets), South End (restaurants and galleries), Fenway/Kenmore (Fenway Park), and the Seaport District (modern waterfront development). Major institutions: Harvard and MIT (just across the Charles River in Cambridge), numerous hospitals and universities (e.g., Boston University, Northeastern, Tufts). Economy: strong in education, healthcare, finance, biotech, and technology. Transportation: Logan International Airport, MBTA subway (the "T"), commuter rail and ferries; Boston is compact and walkable. Culture and sports: world-class museums (Museum of Fine Arts, Isabella Stewart Gardner), vibrant music and theater scenes, passionate sports fans (Red Sox, Celtics, Bruins, Patriots played in nearby Foxborough). Climate: humid continental — cold snowy winters, warm summers. Travel tips: bring comfortable shoes for walking the city and cobblestones, use the T or walk instead of driving (traffic and parking are difficult), book Fenway tours or Red Sox tickets in advance, visit in spring or fall for milder weather and fewer tourists."}',
                            tool_call_id='call_IUcMIrMvIy80BPfbBf6gzo9c',
                            id='fc_0a3de6f9c3c28af7006945adec0c588194be9a95467592db7a',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=49, output_tokens=496, details={'reasoning_tokens': 192}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 22, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_IUcMIrMvIy80BPfbBf6gzo9c',
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

    async def test_auto_with_union_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(
            openai_responses_model, output_type=str | CityInfo, tools=[get_weather]
        )
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Get weather for Miami and describe it',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Miami and describe it',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f54edbaf2148c84006945adf248b08196a95096f6caea59c5',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Miami"}',
                            tool_call_id='call_lYXStgN1Yy9d1dZ7tF9Nt0OM',
                            id='fc_0f54edbaf2148c84006945adf5169c8196a7e7eade1e5dafd0',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=74, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 33, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id='call_lYXStgN1Yy9d1dZ7tF9Nt0OM',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f54edbaf2148c84006945adf62c08819698137632dff26c86',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Miami","summary":"Sunny, 22°C — clear skies and pleasant warmth. Expect bright sun; light layers or short sleeves comfortable. Use sunscreen and sunglasses if spending time outdoors; mornings/evenings might be slightly cooler. Overall great weather for outdoor activities."}',
                            tool_call_id='call_3cHnRoHfPOeJqJHc1iuPX0Aj',
                            id='fc_0f54edbaf2148c84006945adf8ca648196b667c5d79dd1b368',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=255, output_tokens=199, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 37, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_3cHnRoHfPOeJqJHc1iuPX0Aj',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(
            openai_responses_model, output_type=str | CityInfo, tools=[get_weather]
        )
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run(
                'Describe Seattle briefly',
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=1000),
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
                        ThinkingPart(
                            content='',
                            id='rs_0b9be64cebe1cb0c006945adfc8f78819e833f5eb13a7fa4ed',
                            signature=IsStr(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content='Seattle is the largest city in Washington state, located on a narrow isthmus between Puget Sound and Lake Washington in the Pacific Northwest. Known as the "Emerald City," it has a mild, maritime climate with wet winters and relatively dry summers. Landmarks include the Space Needle, Pike Place Market, and waterfront ferries, with frequent views of Mount Rainier on clear days. Seattle is a major tech and aerospace hub (Amazon, Microsoft nearby, Boeing), and it has a strong music, coffee, and outdoor culture. The city proper has roughly 700–800k residents, with a metro area of about 3.5–4 million.',
                            id='msg_0b9be64cebe1cb0c006945adfe9a4c819e9e69fa478fb9f57b',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=70, output_tokens=267, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 44, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )
