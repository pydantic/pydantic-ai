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
                            tool_call_id='call_mgvl52du237y4xpaykoc611g',
                        )
                    ],
                    usage=RequestUsage(input_tokens=773, output_tokens=29),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 19, tzinfo=timezone.utc),
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
                            tool_call_id='call_mgvl52du237y4xpaykoc611g',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='[get_weather, {"city": "Paris"}]')],
                    usage=RequestUsage(input_tokens=797, output_tokens=11),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 20, tzinfo=timezone.utc),
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
                    parts=[TextPart(content='Hello!')],
                    usage=RequestUsage(input_tokens=772, output_tokens=3),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 21, tzinfo=timezone.utc),
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
                    parts=[TextPart(content='Hello! How can I assist you today?')],
                    usage=RequestUsage(input_tokens=769, output_tokens=10),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 22, tzinfo=timezone.utc),
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
                            tool_call_id='call_hkh5cm1mdj44gu0tmk2r4pyy',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"weather summary"}',
                            tool_call_id='call_4jc2lqunzd54sw7qcq8p9bev',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=881, output_tokens=63),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 23, tzinfo=timezone.utc),
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
                            tool_call_id='call_4jc2lqunzd54sw7qcq8p9bev',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_hkh5cm1mdj44gu0tmk2r4pyy',
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
Berlin! As a virtual assistant, I don't have real-time access to current weather conditions. However, I can suggest a few ways for you to find out the current weather in Berlin:

1. **Check a weather website or app**: You can visit websites like AccuWeather, Weather.com, or Deutsche Wetterdienst (DWD) to get the current weather conditions and forecast for Berlin.
2. **Use a search engine**: Simply type "Berlin weather" or "weather Berlin" in a search engine like Google, and you'll get the current weather conditions and forecast.
3. **Check a weather service**: You can also check services like Dark Sky (for iOS or Android) or Weather Underground (for iOS or Android) for hyperlocal weather forecasts.

That being said, Berlin has a temperate climate with four distinct seasons. Here's a rough idea of what you can expect:

* **Spring (March to May)**: Mild temperatures (around 10-20°C/50-68°F) with occasional rain showers.
* **Summer (June to August)**: Warm temperatures (around 20-25°C/68-77°F) with occasional heatwaves.
* **Autumn (September to November)**: Cooler temperatures (around 10-15°C/50-59°F) with some rain.
* **Winter (December to February)**: Cold temperatures (around 0-5°C/32-41°F) with occasional snowfall.

Keep in mind that these are general temperature ranges, and actual weather conditions can vary from year to year.

If you have a specific question about Berlin's weather or need more information, feel free to ask!\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=17, output_tokens=335),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 26, tzinfo=timezone.utc),
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
Rome! The Eternal City!

I'm a large language model, I don't have real-time access to current weather conditions. However, I can suggest some ways for you to find out the current weather in Rome:

1. **Check a weather website or app**: You can visit websites like AccuWeather, Weather.com, or Rome's official website to get the current weather conditions and forecast.
2. **Use a weather API**: If you're looking for a more programmatic way to get the weather, you can use a weather API like OpenWeatherMap or Dark Sky.
3. **Ask a voice assistant**: If you have a voice assistant like Siri, Google Assistant, or Alexa, you can ask them "What's the weather in Rome?" and they'll provide you with the current conditions.

That being said, Rome has a Mediterranean climate with warm summers and mild winters. Here's a general idea of what you can expect:

* **Summer (June to August)**: Hot and dry, with average highs around 32°C (90°F) and lows around 18°C (64°F).
* **Autumn (September to November)**: Mild and pleasant, with average highs around 22°C (72°F) and lows around 12°C (54°F).
* **Winter (December to February)**: Cool and wet, with average highs around 12°C (54°F) and lows around 2°C (36°F).
* **Spring (March to May)**: Mild and sunny, with average highs around 20°C (68°F) and lows around 10°C (50°F).

Keep in mind that these are general temperature ranges, and actual weather conditions can vary from year to year.

Hope this helps!\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=17, output_tokens=345),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 32, tzinfo=timezone.utc),
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
                            args='{"city":"Madrid","summary":"Madrid, the capital of Spain, is known for its vibrant culture, rich history, and beautiful architecture. The city is home to many world-class museums, including the Prado Museum, which features an extensive collection of European art. Madrid is also famous for its lively nightlife, with many bars, clubs, and restaurants to choose from. The city\'s historic center is filled with picturesque plazas, such as the Puerta del Sol and Plaza Mayor, and the Royal Palace of Madrid is a must-visit attraction."}',
                            tool_call_id='call_93nzyeobxbthyrrr1xxsy67y',
                        )
                    ],
                    usage=RequestUsage(input_tokens=787, output_tokens=134),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 29, 28, tzinfo=timezone.utc),
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
                            tool_call_id='call_93nzyeobxbthyrrr1xxsy67y',
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
                    tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='call_kv9xh65ghrljbxclmz4xcuzh'
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
                    tool_call_id='call_p72l9inf42dt3a39z3focfkq',
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
        """Model can use specified tool or respond with text when `allow_text_output=True`."""
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
Paris, the capital of France, is known as the "City of Light" for its historical and cultural significance. The city is famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which houses the Mona Lisa. Paris is also renowned for its fashion industry, art museums, and romantic atmosphere. \n\

Would you like to know more about a specific aspect of Paris or perhaps the current weather there?\
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
                    tool_name='get_time', args='{"timezone":"Tokyo"}', tool_call_id='call_pxec3y1ko9fx2l0l9dtxgm0z'
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
                    tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_awshb3sadh758srpz3mx4c2f'
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
                            tool_call_id='call_o0znvkn2csjm46xsqb1cvydx',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"weather summary"}',
                            tool_call_id='call_o7r2bzt63k9mez3xi7cuc64i',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=881, output_tokens=65),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 27, 56, tzinfo=timezone.utc),
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
                            tool_call_id='call_o7r2bzt63k9mez3xi7cuc64i',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_o0znvkn2csjm46xsqb1cvydx',
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
                            tool_call_id='call_yxd2o7y1dj6wnpjkcvewbbum',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Denver","summary":"weather summary"}',
                            tool_call_id='call_xja0kznyv967t4emgk7qiu03',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=973, output_tokens=65),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 27, 57, tzinfo=timezone.utc),
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
                            tool_call_id='call_xja0kznyv967t4emgk7qiu03',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id='call_yxd2o7y1dj6wnpjkcvewbbum',
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
                            args='{"city":"New York","summary":"New York is a state in the northeastern United States. It is one of the most populous states and is often referred to as \\"The Empire State\\". The state is known for its diverse geography, including mountains, forests, and coastlines along the Atlantic Ocean and Lake Ontario. New York City, the most populous city in the United States, is located in the southeastern part of the state and is a global hub for finance, culture, and entertainment."}',
                            tool_call_id='call_fua91omx53wxt8k2lyh4xw50',
                        )
                    ],
                    usage=RequestUsage(input_tokens=788, output_tokens=128),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 27, 59, tzinfo=timezone.utc),
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
                            tool_call_id='call_fua91omx53wxt8k2lyh4xw50',
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
                            args='{"city":"Boston","summary":"Boston is the capital and most populous city of the Commonwealth of Massachusetts in the United States. Boston is also the seat of Suffolk County, although the county government was disbanded on July 1, 1997. The city proper covers 48.4 square miles (125.4 km²), with an adjacent inner harbor. Boston is one of the oldest municipalities in the United States, founded on the Shawmut Peninsula in 1630 by Puritan settlers. It was named after Boston, Lincolnshire, England; the area was called Trimountain by the local Tawahe Indigenous people."}',
                            tool_call_id='call_3u767fywqob9o3q7wr8op7c7',
                        )
                    ],
                    usage=RequestUsage(input_tokens=787, output_tokens=156),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 1, tzinfo=timezone.utc),
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
                            tool_call_id='call_3u767fywqob9o3q7wr8op7c7',
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
                    timestamp=datetime(2025, 12, 22, 16, 28, 2, tzinfo=timezone.utc),
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
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Seattle","summary":"Seattle is a major city in the U.S. state of Washington."}',
                            tool_call_id='call_i1bfus9kj6jgy1wvhxhpl59i',
                        )
                    ],
                    usage=RequestUsage(input_tokens=786, output_tokens=63),
                    model_name='meta-llama/Llama-4-Scout-17B-16E-Instruct',
                    timestamp=datetime(2025, 12, 22, 16, 28, 4, tzinfo=timezone.utc),
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
                            tool_call_id='call_i1bfus9kj6jgy1wvhxhpl59i',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )
