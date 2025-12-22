"""Bedrock tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the Bedrock API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live Bedrock API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.

Bedrock-specific behavior:
- Bedrock doesn't support native tool_choice='none', so we filter out tools instead
- Bedrock only supports forcing a single tool; multiple tools use 'any' mode
"""

from __future__ import annotations

import os
from datetime import timezone

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
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 not installed'),
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
def bedrock_model():
    """Create a Bedrock model for testing.

    Uses AWS_BEARER_TOKEN_BEDROCK env var for authentication.
    """
    provider = BedrockProvider(api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK') or 'dummy', region_name='us-east-2')
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)
    yield model
    provider.client.close()


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(bedrock_model, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Paris'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=572, output_tokens=53),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="The weather in Paris is currently sunny with a temperature of 22°C (about 72°F). It's a beautiful day!"
                        )
                    ],
                    usage=RequestUsage(input_tokens=646, output_tokens=31),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'end_turn'},
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(bedrock_model, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello!')],
                    usage=RequestUsage(input_tokens=570, output_tokens=5),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'end_turn'},
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(bedrock_model, tools=[get_weather])

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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'London'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=572, output_tokens=53),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The weather in London is currently sunny with a temperature of 22°C (about 72°F).'
                        )
                    ],
                    usage=RequestUsage(input_tokens=646, output_tokens=25),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'end_turn'},
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(bedrock_model, output_type=CityInfo, tools=[get_weather])
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
                            args={'city': 'Tokyo'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=737, output_tokens=38),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Tokyo',
                                'summary': "Tokyo is currently experiencing sunny weather with a pleasant temperature of 22°C (approximately 72°F). It's a beautiful day in Japan's capital city with clear skies and comfortable conditions.",
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=810, output_tokens=93),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceNone:
    """Tests for tool_choice='none' and tool_choice=[].

    When tool_choice is 'none' or [], function tools are disabled.
    Bedrock handles this by not sending tools (no native 'none' support).
    """

    async def test_none_prevents_function_tool_calls(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Model responds with text when tool_choice='none', even with tools available.

        Bedrock doesn't support native 'none' mode, so we handle it by not sending tools.
        """
        agent: Agent[None, str] = Agent(bedrock_model, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I don't have access to real-time weather information. To get the current weather in Berlin, I recommend:

1. **Weather websites**: Check weather.com, accuweather.com, or weather.gov
2. **Search engines**: Simply search "Berlin weather" on Google
3. **Weather apps**: Use apps like Weather Channel, AccuWeather, or your phone's built-in weather app
4. **Voice assistants**: Ask Siri, Google Assistant, or Alexa

Would you like me to help you with something else about Berlin, such as typical weather patterns by season?\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=14, output_tokens=133),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'end_turn'},
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(bedrock_model, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
I don't have access to real-time weather data, so I can't tell you the current weather in Rome. \n\

To get current weather information for Rome, you could:
- Check weather websites like weather.com or accuweather.com
- Search "Rome weather" in any search engine
- Use a weather app on your phone
- Ask a voice assistant with internet access

Is there anything else about Rome I can help you with?\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=14, output_tokens=98),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'end_turn'},
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(bedrock_model, output_type=CityInfo, tools=[get_weather])
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
                            args={
                                'city': 'Madrid',
                                'summary': 'Madrid is the capital and largest city of Spain, located in the center of the country. It is known for its rich cultural heritage, world-class museums including the Prado Museum, Royal Palace, vibrant nightlife, and beautiful parks like Retiro Park. The city is a major political, economic, and cultural center of Spain, with a metropolitan population of over 6 million people. Madrid features stunning architecture ranging from medieval to modern, excellent cuisine including tapas, and is home to famous football clubs like Real Madrid. The city enjoys a continental Mediterranean climate with hot summers and cool winters.',
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=677, output_tokens=171),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
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

    async def test_required_forces_tool_use(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await bedrock_model.request(
            [ModelRequest.user_text_prompt("What's the weather in Paris?")],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id=IsStr())]
        )

    async def test_required_with_multiple_tools(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await bedrock_model.request(
            [ModelRequest.user_text_prompt('What time is it in London?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time',
                    args={'timezone': 'Europe/London'},
                    tool_call_id=IsStr(),
                )
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().

    Bedrock-specific: Only supports forcing a single tool; multiple tools use 'any' mode.
    """

    async def test_single_tool_in_list(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await bedrock_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id=IsStr())]
        )

    async def test_multiple_tools_in_list_uses_any_mode(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Multiple tools in list - Bedrock uses 'any' mode (must use one).

        Bedrock only supports forcing a single tool. When multiple tools are
        specified, we fall back to 'any' mode which requires the model to use
        one of the available tools.
        """
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await bedrock_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_time', args={'timezone': 'Asia/Tokyo'}, tool_call_id=IsStr())]
        )

    async def test_excluded_tool_not_called(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Tools not in the list are filtered out - model only sees allowed tools."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await bedrock_model.request(
            [ModelRequest.user_text_prompt("What's the weather in London?")],
            settings,
            params,
        )

        # Only get_weather is sent to the API, get_population is filtered out
        assert response.parts == snapshot(
            [ToolCallPart(tool_name='get_weather', args={'city': 'London'}, tool_call_id=IsStr())]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            bedrock_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                            args={'city': 'Sydney'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=737, output_tokens=38),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Sydney',
                                'summary': "Sydney is experiencing sunny weather with a pleasant temperature of 22°C (approximately 72°F). It's a beautiful day with clear skies.",
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=810, output_tokens=84),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, bedrock_model: BedrockConverseModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput.

        With multiple tools, Bedrock uses 'any' mode.
        """
        agent: Agent[None, CityInfo] = Agent(
            bedrock_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                            args={'city': 'Denver'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=798, output_tokens=38),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Denver',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Denver',
                                'summary': 'The weather in Denver is sunny with a temperature of 22°C (approximately 72°F).',
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=871, output_tokens=75),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
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

    async def test_auto_with_only_output_tools(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(bedrock_model, output_type=CityInfo)

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
                            args={
                                'city': 'New York',
                                'summary': 'New York, often called New York City (NYC) or simply "The Big Apple," is the most populous city in the United States with over 8 million residents. Located in the state of New York, it comprises five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. NYC is a global hub for finance, culture, media, fashion, art, and entertainment. It\'s home to iconic landmarks including the Statue of Liberty, Empire State Building, Central Park, Times Square, and Broadway theaters. The city hosts the New York Stock Exchange on Wall Street and is headquarters to numerous Fortune 500 companies. Known for its diverse population, world-class museums like the Metropolitan Museum of Art, and renowned culinary scene, New York attracts millions of tourists annually and serves as a major center for international diplomacy, housing the United Nations headquarters.',
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=673, output_tokens=234),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(bedrock_model, output_type=CityInfo)
        settings: ModelSettings = {'tool_choice': 'none'}

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
                            args={
                                'city': 'Boston',
                                'summary': "Boston is the capital and largest city of Massachusetts, located in the northeastern United States. Founded in 1630, it is one of America's oldest cities and played a pivotal role in the American Revolution, with historic sites like the Freedom Trail, Boston Tea Party Ships & Museum, and Paul Revere's House. The city is a major center for education, home to prestigious institutions including Harvard University and MIT in the greater Boston area, as well as Boston University and Northeastern University. Known for its rich history, diverse neighborhoods like Beacon Hill and the North End, Boston is also famous for its sports teams (Red Sox, Celtics, Bruins, Patriots), world-class hospitals and medical research facilities, thriving technology and finance sectors, and cultural attractions including the Boston Symphony Orchestra and Museum of Fine Arts. The city has a population of approximately 675,000 in the city proper and over 4.8 million in the metropolitan area.",
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=672, output_tokens=250),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
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

    async def test_auto_with_union_output(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(bedrock_model, output_type=str | CityInfo, tools=[get_weather])
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
                        TextPart(content="I'll get the weather information for Miami and describe it for you."),
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Miami'},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=645, output_tokens=68),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Miami',
                                'summary': "Miami is currently experiencing sunny weather with a pleasant temperature of 22°C (approximately 72°F). It's a beautiful day with clear skies and comfortable conditions - perfect for outdoor activities like visiting the beach, taking a walk along Ocean Drive, or enjoying outdoor dining.",
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=733, output_tokens=124),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, bedrock_model: BedrockConverseModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(bedrock_model, output_type=str | CityInfo, tools=[get_weather])
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
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Seattle',
                                'summary': 'Seattle is a major coastal city in the Pacific Northwest region of Washington State, known for its iconic Space Needle, thriving tech industry (home to Amazon and Microsoft), vibrant coffee culture (birthplace of Starbucks), and stunning natural surroundings including Puget Sound and Mount Rainier. The city has a reputation for frequent rain, a strong music scene (grunge movement), and a progressive, innovative culture.',
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=580, output_tokens=157),
                    model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    timestamp=IsDatetime(),
                    provider_name='bedrock',
                    provider_url='https://bedrock-runtime.us-east-2.amazonaws.com',
                    provider_details={'finish_reason': 'tool_use'},
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )
