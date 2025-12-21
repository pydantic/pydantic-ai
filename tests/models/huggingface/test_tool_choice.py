"""HuggingFace API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the HuggingFace API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live HuggingFace Inference API.

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
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface not installed'),
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
def huggingface_model(huggingface_api_key: str, request: pytest.FixtureRequest) -> HuggingFaceModel:
    """Create a HuggingFace model for testing.

    Defaults to Llama-4-Scout via Together provider. Can be customized via pytest.mark.parametrize
    or by setting `model_name` and `provider_name` markers on the test.
    """
    model_name = getattr(request, 'param', {}).get('model_name', 'meta-llama/Llama-4-Scout-17B-16E-Instruct')
    provider_name = getattr(request, 'param', {}).get('provider_name', 'together')
    return HuggingFaceModel(
        model_name,
        provider=HuggingFaceProvider(provider_name=provider_name, api_key=huggingface_api_key),
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(huggingface_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=2000),
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
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Paris"}',
                            tool_call_id='call_f766mdwhj0oq0z0v375b5gpo',
                        )
                    ],
                    usage=RequestUsage(input_tokens=773, output_tokens=29),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 6, 21, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='call_f766mdwhj0oq0z0v375b5gpo',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='[get_weather("Paris")]')],
                    usage=RequestUsage(input_tokens=797, output_tokens=6),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 6, 21, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(huggingface_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=2000),
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
                    parts=[TextPart(content='Hello')],
                    usage=RequestUsage(input_tokens=772, output_tokens=2),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 6, 22, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """When tool_choice is not set (None), behaves like 'auto' - model can respond directly."""
        agent: Agent[None, str] = Agent(huggingface_model, tools=[get_weather])

        result = await agent.run(
            'Say hello',
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Say hello',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello!')],
                    usage=RequestUsage(input_tokens=769, output_tokens=3),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 32, 31, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(huggingface_model, output_type=CityInfo, tools=[get_weather])
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
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Tokyo"}',
                            tool_call_id='call_yriz1tydjscsvnsytw354i0a',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"weather summary"}',
                            tool_call_id='call_qbu72noklg7hj7l1qe89fbov',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=881, output_tokens=63),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_qbu72noklg7hj7l1qe89fbov',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_yriz1tydjscsvnsytw354i0a',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
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
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(huggingface_model, tools=[get_weather])
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
                        TextPart(
                            content="""\
I'm not able to provide real-time information, but I can suggest some ways for you to find out the current weather in Berlin:

1. Check a weather website or app: You can check websites like weather.com, accuweather.com, or wunderground.com for current weather conditions and forecasts in Berlin.
2. Use a search engine: Simply type "Berlin weather" or "weather in Berlin" into a search engine like Google, and you'll get the current weather conditions and forecast.
3. Check a weather service: You can also check the website of a national weather service, such as the German Weather Service (Deutscher Wetterdienst) or the European Centre for Medium-Range Weather Forecasts (ECMWF).

As of my knowledge cutoff, Berlin has a temperate climate with four distinct seasons. The city experiences cold winters, with average temperatures ranging from -2°C to 2°C (28°F to 36°F) in January, the coldest month. Summers are mild, with average temperatures ranging from 18°C to 23°C (64°F to 73°F) in July, the warmest month.

Please note that weather conditions can change quickly, so it's always a good idea to check a reliable weather source for the most up-to-date information.\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=17, output_tokens=255),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 3, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(huggingface_model, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I'm not able to provide real-time information, but I can give you a general overview of Rome's climate. Rome has a Mediterranean climate, characterized by warm summers and mild winters. \n\

- **Summer (June to August):** Expect high temperatures, often reaching above 30°C (86°F), and occasional heatwaves.

- **Winter (December to February):** It's mild, with temperatures ranging from 2°C to 12°C (36°F to 54°F). \n\

- **Spring (March to May) and Autumn (September to November):** These seasons are generally pleasant, with temperatures ranging from 10°C to 25°C (50°F to 77°F), making them ideal times to visit.

For the most accurate and up-to-date weather forecast, I recommend checking a reliable weather website or app.\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=17, output_tokens=167),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 5, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(huggingface_model, output_type=CityInfo, tools=[get_weather])
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
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Madrid","summary":"Madrid, the capital of Spain, is known for its vibrant culture, rich history, and beautiful architecture. The city is home to many famous landmarks such as the Royal Palace of Madrid, the Prado Museum, and the Retana Park. Madrid is also famous for its nightlife, with many bars, clubs, and restaurants to choose from."}',
                            tool_call_id='call_6n5q5145abxn90lj0iptzbyy',
                        )
                    ],
                    usage=RequestUsage(input_tokens=787, output_tokens=101),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 6, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_6n5q5145abxn90lj0iptzbyy',
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

    async def test_required_forces_tool_use(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await huggingface_model.request(
            [ModelRequest.user_text_prompt("What's the weather in Paris?")],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='call_gihofucdc84ah6xxwh23i3na'
                )
            ]
        )

    async def test_required_with_multiple_tools(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await huggingface_model.request(
            [ModelRequest.user_text_prompt('What time is it in London?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time',
                    args='{"timezone":"Europe/London"}',
                    tool_call_id='call_4zpadw1mg6gsyj5tuqzhflu4',
                )
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await huggingface_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                TextPart(
                    content="""\
Paris, the capital of France, is known as the "City of Light" for its role in the Enlightenment and its many famous intellectuals and artists. It's famous for its stunning architecture, art museums, fashion industry, and romantic atmosphere. Some must-see attractions include the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. \n\

Would you like to know more about Paris, or perhaps check the current weather there?\
"""
                )
            ]
        )

    async def test_multiple_tools_in_list(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Multiple tools in list - model must use one from the filtered set."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await huggingface_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time', args='{"timezone":"Tokyo"}', tool_call_id='call_dvvajjk9h4b675ascgmxjq79'
                )
            ]
        )

    async def test_excluded_tool_not_called(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Tools not in the list are filtered out - model only sees allowed tools."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await huggingface_model.request(
            [ModelRequest.user_text_prompt("What's the weather in London?")],
            settings,
            params,
        )

        # Only get_weather is sent to the API, get_population is filtered out
        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_dj8h707vyzy6oxkuqsv6o32n'
                )
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            huggingface_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Sydney"}',
                            tool_call_id='call_3mswl7o9n2zlk60jpibwseec',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"weather summary"}',
                            tool_call_id='call_rdto0toj7y3o4juwnkyow18i',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=881, output_tokens=65),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 12, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_rdto0toj7y3o4juwnkyow18i',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_3mswl7o9n2zlk60jpibwseec',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, huggingface_model: HuggingFaceModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            huggingface_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Denver"}',
                            tool_call_id='call_nycsfyh5qhhkz4t49m3vcg46',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Denver","summary":"weather summary for Denver"}',
                            tool_call_id='call_n6197dew35ec29r6bvp6m6z8',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=973, output_tokens=67),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 13, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_n6197dew35ec29r6bvp6m6z8',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_nycsfyh5qhhkz4t49m3vcg46',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestNoFunctionTools:
    """Tests for scenarios without function tools.

    These tests verify tool_choice behavior when only output tools exist.
    """

    async def test_auto_with_only_output_tools(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(huggingface_model, output_type=CityInfo)

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
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"New York","summary":"New York is a state in the northeastern United States. It is one of the most populous states and is known for its diverse culture, iconic cities, and natural beauty. The state capital is Albany, while New York City is the largest city and a global hub for finance, media, and entertainment. New York is also famous for its historical landmarks, such as the Statue of Liberty and Central Park."}',
                            tool_call_id='call_956g1vy0rvpbqmwm6y9wd794',
                        )
                    ],
                    usage=RequestUsage(input_tokens=788, output_tokens=118),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 15, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_956g1vy0rvpbqmwm6y9wd794',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(huggingface_model, output_type=CityInfo)
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
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Boston","summary":"Boston is the capital and most populous city of the Commonwealth of Massachusetts in the United States. Boston is also the seat of Suffolk County, although the county government was disbanded on July 1, 1997. The city proper covers 48.4 square miles (125.4 km²) with an estimated 694,583 residents in 2023, making it the 24th most populous city in the United States, and the 118th most populous city in the world. The city is the economic and cultural anchor of a substantially larger metropolitan area known as Greater Boston, which has a population of 4,896,149 and ranks as the 10th most populous metropolitan area in the United States. Boston is a global center of education, research, medical care and higher education. The city is known for its academic institutions, including Harvard University and the Massachusetts Institute of Technology.  Boston was the site of several key events of the American Revolution, including the Boston Massacre, the Boston Tea Party, the Siege of Boston, and the Battles of Lexington and Concord."}',
                            tool_call_id='call_yyrfsjevuwaw937eo4orgumq',
                        )
                    ],
                    usage=RequestUsage(input_tokens=787, output_tokens=260),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 18, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_yyrfsjevuwaw937eo4orgumq',
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

    async def test_auto_with_union_output(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(huggingface_model, output_type=str | CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Describe Tokyo in one sentence',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Describe Tokyo in one sentence',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='Tokyo, the capital of Japan, is a bustling metropolis known for its vibrant culture, rich history, and cutting-edge technology, offering a unique blend of traditional and modern attractions.'
                        )
                    ],
                    usage=RequestUsage(input_tokens=880, output_tokens=37),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 32, 32, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, huggingface_model: HuggingFaceModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(huggingface_model, output_type=str | CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run(
                'Describe Seattle briefly',
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=2000),
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
                            content='Seattle is a vibrant city located in the Pacific Northwest region of the United States. It is known for its stunning natural beauty, with mountains, forests, and waterways surrounding the city. Seattle is home to iconic landmarks such as the Space Needle, Pike Place Market, and the Seattle Waterfront. The city is also famous for its coffee culture, being the birthplace of Starbucks, and is a hub for technology and innovation, with companies like Amazon and Microsoft headquartered there. With a thriving arts and music scene, Seattle is a popular destination for tourists and a great place to live for its residents.'
                        )
                    ],
                    usage=RequestUsage(input_tokens=786, output_tokens=121),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 21, 4, 7, 59, tzinfo=timezone.utc),
                    provider_name='huggingface',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id=IsStr(),
                    run_id=IsStr(),
                ),
            ]
        )
