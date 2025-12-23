"""Mistral API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the Mistral API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live Mistral API.

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

from ...conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
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
    return f'14:30 in {timezone}'  # pragma: no cover


def get_population(city: str) -> str:
    """Get the population of a city."""
    return f'{city} has 1 million people'  # pragma: no cover


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
def mistral_model(mistral_api_key: str) -> MistralModel:
    """Create a Mistral model for testing."""
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key=mistral_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(mistral_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='GZecN9HXI')],
                    usage=RequestUsage(input_tokens=77, output_tokens=12),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 23, 10, 12, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='GZecN9HXI',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The current weather in **Paris** is **sunny** with a temperature of **22°C**. Enjoy your day! 😊'
                        )
                    ],
                    usage=RequestUsage(input_tokens=100, output_tokens=29),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 23, 10, 12, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, mistral_model: MistralModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(mistral_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hi!')],
                    usage=RequestUsage(input_tokens=75, output_tokens=3),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 52, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(self, mistral_model: MistralModel, allow_model_requests: None):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(mistral_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city": "London"}', tool_call_id='TCWfTKJco')],
                    usage=RequestUsage(input_tokens=77, output_tokens=12),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 53, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id='TCWfTKJco',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The current weather in **London** is **sunny** with a temperature of **22°C**.'
                        )
                    ],
                    usage=RequestUsage(input_tokens=100, output_tokens=22),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 54, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(mistral_model, output_type=CityInfo, tools=[get_weather])
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Tokyo"}', tool_call_id='dKE8phQYQ')],
                    usage=RequestUsage(input_tokens=149, output_tokens=12),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 55, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id='dKE8phQYQ',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Tokyo", "summary": "The current weather in Tokyo is sunny with a temperature of 22°C."}',
                            tool_call_id='Zm2jqOwWa',
                        )
                    ],
                    usage=RequestUsage(input_tokens=172, output_tokens=32),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 56, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='Zm2jqOwWa',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceNone:
    """Tests for tool_choice='none' and tool_choice=[].

    When tool_choice is 'none' or [], function tools are disabled but output tools remain.
    """

    async def test_none_prevents_function_tool_calls(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(mistral_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
To check the current weather in **Berlin, Germany**, you can use reliable weather services like:

- **[Weather.com (The Weather Channel)](https://weather.com/weather/today/l/Berlin+Germany)**
- **[AccuWeather](https://www.accuweather.com/en/de/berlin/10178/weather-forecast/178087)**
- **[Windy.com](https://www.windy.com/?52.520,13.405,5)** (for detailed wind/rain maps)
- **[Meteoblue](https://www.meteoblue.com/en/weather/forecast/week/berlin_germany_2950159)** (for hourly/daily forecasts)

### **Quick Check (as of my last update):**
- **Current conditions**: Likely around **10-20°C (50-68°F)** depending on the season.
- **Forecast**: Check for rain, wind, or sunshine (Berlin has a temperate climate with mild summers and cold winters).

For **real-time updates**, I recommend using one of the links above or asking a voice assistant like **Siri, Google Assistant, or Alexa**.

Would you like a **5-day forecast** or details on specific weather conditions (e.g., humidity, UV index)? Let me know!\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=10, output_tokens=287),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 22, 22, 24, 30, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, mistral_model: MistralModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(mistral_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
To check the current weather in **Rome, Italy**, you can use reliable weather services like:

- **[Weather.com (The Weather Channel)](https://weather.com)**
- **[AccuWeather](https://www.accuweather.com)**
- **[Meteo.it (Italian Meteorological Service)](https://www.meteo.it)**

### **Quick Check (as of my last update):**
- **Today's Forecast:** Typically mild to warm, depending on the season.
- **Current Conditions:** Check for real-time updates (e.g., sunny, partly cloudy, rain).

For the most accurate and up-to-date information, I recommend checking one of the links above. Would you like help interpreting a forecast? 😊\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=10, output_tokens=150),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 22, 22, 24, 37, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, mistral_model: MistralModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(mistral_model, output_type=CityInfo, tools=[get_weather])
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Madrid", "summary": "Madrid is the capital and largest city of Spain, located in the heart of the Iberian Peninsula. It is known for its rich history, vibrant culture, and stunning architecture. Madrid is a major global city and a hub for politics, economics, and culture in Spain."}',
                            tool_call_id='A82EnEevK',
                        )
                    ],
                    usage=RequestUsage(input_tokens=83, output_tokens=70),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 19, 57, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='A82EnEevK',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
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

    async def test_required_forces_tool_use(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await mistral_model.request(
            [ModelRequest.user_text_prompt("What's the weather in Paris?")],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='YFSMyukN7')]
        )

    async def test_required_with_multiple_tools(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await mistral_model.request(
            [ModelRequest.user_text_prompt('What time is it in London?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_time', args='{"timezone": "Europe/London"}', tool_call_id='3CnJ1w2Ur')]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await mistral_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='MfYZxFe2z')]
        )

    async def test_multiple_tools_in_list(self, mistral_model: MistralModel, allow_model_requests: None):
        """Multiple tools in list - model must use one from the filtered set."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await mistral_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_time', args='{"timezone": "Asia/Tokyo"}', tool_call_id='B4aonuKWf')]
        )

    async def test_excluded_tool_not_called(self, mistral_model: MistralModel, allow_model_requests: None):
        """Tools not in the list are filtered out - model only sees allowed tools."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await mistral_model.request(
            [ModelRequest.user_text_prompt("What's the weather in London?")],
            settings,
            params,
        )

        # Only get_weather is sent to the API, get_population is filtered out
        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city": "London"}', tool_call_id='fjE67l0q3')]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, mistral_model: MistralModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            mistral_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Sydney"}', tool_call_id='WwdMOqAu1')],
                    usage=RequestUsage(input_tokens=149, output_tokens=12),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 5, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id='WwdMOqAu1',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Sydney", "summary": "The current weather in Sydney is sunny with a temperature of 22 degrees Celsius."}',
                            tool_call_id='CEw5nSa3y',
                        )
                    ],
                    usage=RequestUsage(input_tokens=172, output_tokens=33),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 6, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='CEw5nSa3y',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, mistral_model: MistralModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            mistral_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_population'])}

        result = await agent.run('Get weather for Denver and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Denver and summarize',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='get_weather', args='{"city": "Denver"}', tool_call_id='ENXLc6EwO'),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Denver", "summary": "Getting weather information..."}',
                            tool_call_id='ujVp0kyuT',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=212, output_tokens=33),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 7, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='ujVp0kyuT',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='ENXLc6EwO',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )


class TestNoFunctionTools:
    """Tests for scenarios without function tools.

    These tests verify tool_choice behavior when only output tools exist.
    """

    async def test_auto_with_only_output_tools(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(mistral_model, output_type=CityInfo)

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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "New York", "summary": "New York City is the most populous city in the United States, located in the state of New York. It is a global hub for finance, culture, art, fashion, and entertainment. NYC is composed of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. \\n\\nKey highlights include:\\n1. **Iconic Landmarks**: Times Square, Statue of Liberty, Empire State Building, Central Park, and Broadway.\\n2. **Cultural Diversity**: Home to people from all over the world, with over 800 languages spoken.\\n3. **Economy**: A major center for banking, finance, media, and technology.\\n4. **Arts and Entertainment**: World-class museums like the Metropolitan Museum of Art, theaters, music venues, and galleries.\\n5. **History**: Founded in 1624 as a trading post by the Dutch, it played a pivotal role in the American Revolution and the growth of the United States."}',
                            tool_call_id='8IdErjooG',
                        )
                    ],
                    usage=RequestUsage(input_tokens=84, output_tokens=223),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 9, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='8IdErjooG',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, mistral_model: MistralModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(mistral_model, output_type=CityInfo)
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Boston", "summary": "Overview"}',
                            tool_call_id='4m4MYUhpn',
                        )
                    ],
                    usage=RequestUsage(input_tokens=83, output_tokens=17),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 14, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='4m4MYUhpn',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )


class TestTextAndStructuredUnion:
    """Tests with union output types (str | BaseModel).

    When output_type is a union including str, allow_text_output is True.
    """

    async def test_auto_with_union_output(self, mistral_model: MistralModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(mistral_model, output_type=str | CityInfo, tools=[get_weather])
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Miami"}', tool_call_id='00Ec68tKC')],
                    usage=RequestUsage(input_tokens=150, output_tokens=13),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 17, 32, 52, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id='00Ec68tKC',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
The current weather in **Miami** is **sunny** with a temperature of **22°C (72°F)**.

Here's a brief description:
- **Conditions**: Clear skies and plenty of sunshine.
- **Temperature**: Warm and comfortable, typical for Miami's tropical climate.
- **What to expect**: Perfect weather for outdoor activities like beach visits, walking, or enjoying a café outdoors.

Would you like more details or recommendations for activities?\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=174, output_tokens=94),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 22, 17, 32, 53, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, mistral_model: MistralModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(mistral_model, output_type=str | CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run(
                'Describe Seattle briefly',
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=500),
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Seattle", "summary": "A brief overview of the city."}',
                            tool_call_id='QGVN6nMrO',
                        )
                    ],
                    usage=RequestUsage(input_tokens=83, output_tokens=24),
                    model_name='mistral-large-latest',
                    timestamp=IsDatetime(),
                    provider_name='mistral',
                    provider_url='https://api.mistral.ai',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 22, 14, 20, 19, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='QGVN6nMrO',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )
