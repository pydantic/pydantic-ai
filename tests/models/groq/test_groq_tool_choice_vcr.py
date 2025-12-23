"""Groq API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the Groq API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live Groq API.

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
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='groq not installed'),
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
def groq_model(groq_api_key: str) -> GroqModel:
    """Create a Groq model for testing."""
    return GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=groq_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(self, groq_model: GroqModel, allow_model_requests: None):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(groq_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
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
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='as64z675v')],
                    usage=RequestUsage(input_tokens=717, output_tokens=29),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 39, tzinfo=timezone.utc),
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
                            tool_call_id='as64z675v',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The weather in Paris is sunny with a temperature of 22C.')],
                    usage=RequestUsage(input_tokens=774, output_tokens=15),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 39, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, groq_model: GroqModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(groq_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
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
                    parts=[TextPart(content='Hello!')],
                    usage=RequestUsage(input_tokens=716, output_tokens=3),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 40, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(self, groq_model: GroqModel, allow_model_requests: None):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(groq_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=900),
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
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city":"London"}', tool_call_id='vj9mvwsqj')],
                    usage=RequestUsage(input_tokens=717, output_tokens=29),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 40, tzinfo=timezone.utc),
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
                            tool_call_id='vj9mvwsqj',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The weather in London is sunny with a temperature of 22C.')],
                    usage=RequestUsage(input_tokens=774, output_tokens=15),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 41, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, groq_model: GroqModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(groq_model, output_type=CityInfo, tools=[get_weather])
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
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Tokyo"}',
                            tool_call_id='bp56cg0tk',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"weather summary"}',
                            tool_call_id='e77c6etjd',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=776, output_tokens=63),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 42, tzinfo=timezone.utc),
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
                            tool_call_id='e77c6etjd',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='bp56cg0tk',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
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

    async def test_none_prevents_function_tool_calls(self, groq_model: GroqModel, allow_model_requests: None):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(groq_model, tools=[get_weather])
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I'd love to help! However, I'm a large language model, I don't have real-time access to current weather conditions. But I can suggest a few options to help you find out the current weather in Berlin:

1. **Check a weather website or app**: You can visit websites like weather.com, accuweather.com, or wunderground.com, or download apps like Dark Sky or Weather Underground to get the current weather conditions in Berlin.
2. **Use a search engine**: Simply type "Berlin weather" or "weather in Berlin" in a search engine like Google, and you'll get the current weather conditions along with a forecast.
3. **Check a local news website**: You can also visit a local news website from Berlin, such as the Berlin-based newspaper "Tagesspiegel" or "Berliner Zeitung", which often have up-to-date weather forecasts.

As of my knowledge cutoff in 2022, Berlin's climate is generally temperate, with warm summers and cold winters. But for the most accurate and up-to-date information, I recommend checking one of the sources mentioned above.\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=16, output_tokens=219),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 43, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, groq_model: GroqModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(groq_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=900),
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
Rome has a Mediterranean climate, characterized by hot, dry summers and mild, wet winters. Here's a general overview of the weather in Rome:

**Current Weather:** \n\
To give you the most up-to-date information, I'd recommend checking a reliable weather website or app, such as AccuWeather, Weather.com, or the Italian Meteorological Society (it might be in Italian). \n\

However, I can tell you that Rome's current weather can be checked through various online platforms.

**Seasonal Weather Patterns:**

* **Spring (March to May)**: Mild temperatures, ranging from 12°C (54°F) to 22°C (72°F), with occasional rain showers.
* **Summer (June to August)**: Hot and dry, with temperatures often reaching 30°C (86°F) or more, sometimes up to 35°C (95°F) or 40°C (104°F) during heatwaves.
* **Autumn (September to November)**: Comfortable temperatures, ranging from 10°C (50°F) to 25°C (77°F), with some rain.
* **Winter (December to February)**: Cool and wet, with temperatures ranging from 2°C (36°F) to 12°C (54°F).

**Monthly Average Weather:**

Here's a rough idea of what to expect:

* January: 8°C (46°F) / 12°C (54°F)
* February: 9°C (48°F) / 13°C (56°F)
* March: 12°C (54°F) / 17°C (63°F)
* April: 15°C (59°F) / 20°C (68°F)
* May: 19°C (66°F) / 24°C (75°F)
* June: 23°C (73°F) / 28°C (82°F)
* July: 26°C (79°F) / 31°C (88°F)
* August: 26°C (79°F) / 31°C (88°F)
* September: 22°C (72°F) / 27°C (81°F)
* October: 18°C (64°F) / 23°C (73°F)
* November: 14°C (57°F) / 18°C (64°F)
* December: 10°C (50°F) / 14°C (57°F)

Keep in mind that these are general temperature ranges, and actual weather conditions can vary from year to year.

If you're planning a trip to Rome, I recommend checking the weather forecast before your visit to ensure you're prepared for any conditions.\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=16, output_tokens=527),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 45, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, groq_model: GroqModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(groq_model, output_type=CityInfo, tools=[get_weather])
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
                            args='{"city":"Madrid","summary":"Madrid, the capital city of Spain, is known for its vibrant culture, rich history, and beautiful architecture. The city is home to many world-class museums, including the Prado Museum, which features an extensive collection of European art. Madrid is also famous for its lively nightlife, with many bars, restaurants, and clubs to choose from. The city\'s historic center is filled with charming streets, picturesque plazas, and stunning landmarks like the Royal Palace of Madrid and the Almudena Cathedral. Visitors can also enjoy the city\'s beautiful parks and gardens, such as the Retiro Park, which offers a peaceful escape from the hustle and bustle of city life. Overall, Madrid is a must-visit destination for anyone interested in exploring the best of Spanish culture, history, and cuisine."}',
                            tool_call_id='bh6yp4bhz',
                        )
                    ],
                    usage=RequestUsage(input_tokens=733, output_tokens=189),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 46, tzinfo=timezone.utc),
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
                            tool_call_id='bh6yp4bhz',
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

    async def test_required_forces_tool_use(self, groq_model: GroqModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await groq_model.request(
            [ModelRequest.user_text_prompt("What's the weather in Paris?")],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='4r042bem6')]
        )

    async def test_required_with_multiple_tools(self, groq_model: GroqModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await groq_model.request(
            [ModelRequest.user_text_prompt('What time is it in London?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_time', args='{"timezone":"Europe/London"}', tool_call_id='td5txahq5')]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, groq_model: GroqModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await groq_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='k21yyzx9y')]
        )

    async def test_multiple_tools_in_list(self, groq_model: GroqModel, allow_model_requests: None):
        """Multiple tools in list - model must use one from the filtered set."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await groq_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_time', args='{"timezone":"Tokyo"}', tool_call_id='10t0tpw88')]
        )

    async def test_excluded_tool_not_called(self, groq_model: GroqModel, allow_model_requests: None):
        """Tools not in the list are filtered out - model only sees allowed tools."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await groq_model.request(
            [ModelRequest.user_text_prompt("What's the weather in London?")],
            settings,
            params,
        )

        # Only get_weather is sent to the API, get_population is filtered out
        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args='{"city":"London"}', tool_call_id='6bgt2m94v')]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(self, groq_model: GroqModel, allow_model_requests: None):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            groq_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Sydney"}',
                            tool_call_id='4bwrtnfj6',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"Current weather in Sydney"}',
                            tool_call_id='4gpzmy0wg',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=776, output_tokens=67),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 50, tzinfo=timezone.utc),
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
                            tool_call_id='4gpzmy0wg',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='4bwrtnfj6',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(self, groq_model: GroqModel, allow_model_requests: None):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            groq_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                        ToolCallPart(tool_name='get_weather', args='{"city":"Denver"}', tool_call_id='1r6z4xm35'),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Denver","summary":"weather summary for Denver"}',
                            tool_call_id='pnkj8t2hc',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=817, output_tokens=67),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 51, tzinfo=timezone.utc),
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
                            tool_call_id='pnkj8t2hc',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='1r6z4xm35',
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

    async def test_auto_with_only_output_tools(self, groq_model: GroqModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(groq_model, output_type=CityInfo)

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
                            args='{"city":"New York","summary":"New York is a state in the northeastern United States. It is known for its diverse culture, iconic landmarks like the Statue of Liberty and Central Park, and being a major hub for finance, media, and entertainment."}',
                            tool_call_id='e54c62maj',
                        )
                    ],
                    usage=RequestUsage(input_tokens=734, output_tokens=80),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 52, tzinfo=timezone.utc),
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
                            tool_call_id='e54c62maj',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, groq_model: GroqModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(groq_model, output_type=CityInfo)
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
                            args='{"city":"Boston","summary":"Boston is the capital and most populous city of the Commonwealth of Massachusetts in the United States. It is also the cultural and economic center of the New England region."}',
                            tool_call_id='bm3v7a482',
                        )
                    ],
                    usage=RequestUsage(input_tokens=733, output_tokens=67),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 52, tzinfo=timezone.utc),
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
                            tool_call_id='bm3v7a482',
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

    async def test_auto_with_union_output(self, groq_model: GroqModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(groq_model, output_type=str | CityInfo, tools=[get_weather])
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
                            content='Get weather for Miami and describe it',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city":"Miami"}', tool_call_id='14mvk25vk')],
                    usage=RequestUsage(input_tokens=764, output_tokens=29),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'tool_calls',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 53, tzinfo=timezone.utc),
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
                            tool_call_id='14mvk25vk',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The current weather in Miami is sunny with a temperature of 22C.')],
                    usage=RequestUsage(input_tokens=821, output_tokens=16),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 53, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, groq_model: GroqModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(groq_model, output_type=str | CityInfo, tools=[get_weather])
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
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
Seattle is a vibrant city located in the Pacific Northwest region of the United States. It is known for its stunning natural beauty, with mountains, forests, and waterways surrounding the city. Seattle is home to iconic landmarks like the Space Needle and Pike Place Market, as well as being the birthplace of grunge music and the headquarters of tech giants like Amazon and Microsoft. The city has a thriving arts and culture scene, a diverse food scene, and a strong sense of community. With its mild oceanic climate, Seattle is a popular destination for outdoor enthusiasts and those who love urban exploration. \n\

Would you like to know more about Seattle?\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=719, output_tokens=129),
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={
                        'finish_reason': 'stop',
                        'timestamp': datetime(2025, 12, 19, 22, 31, 54, tzinfo=timezone.utc),
                    },
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )
