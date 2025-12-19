"""Anthropic tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the Anthropic API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live Anthropic API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.

Anthropic-specific constraints:
- `tool_choice='required'` (maps to type: 'any') is not supported with thinking mode
- Forcing specific tools is not supported with thinking mode
"""

from __future__ import annotations

from datetime import timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits

from ...conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
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
def anthropic_model(anthropic_api_key: str) -> AnthropicModel:
    """Create an Anthropic model for testing."""
    return AnthropicModel('claude-sonnet-4-20250514', provider=AnthropicProvider(api_key=anthropic_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(anthropic_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Paris?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(content="I'll check the current weather in Paris for you."),
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Paris'},
                            tool_call_id='toolu_01F1CKp2MXxLpG8xW9nEFimY',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=389,
                        output_tokens=65,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 389,
                            'output_tokens': 65,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01EAUXkyTiSJJfNTdJ6f5n83',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='toolu_01F1CKp2MXxLpG8xW9nEFimY',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="The current weather in Paris is sunny with a temperature of 22°C (about 72°F). It's a beautiful day there!"
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=474,
                        output_tokens=32,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 474,
                            'output_tokens': 32,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'end_turn'},
                    provider_response_id='msg_017vnRWGtD2b8bgpwzv7KQhd',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(anthropic_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Say hello in one word', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello!')],
                    usage=RequestUsage(
                        input_tokens=387,
                        output_tokens=5,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 387,
                            'output_tokens': 5,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'end_turn'},
                    provider_response_id='msg_01JdsYtnSGgso2Lb5FcxMfeb',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(anthropic_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in London?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(content="I'll check the current weather in London for you."),
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'London'},
                            tool_call_id='toolu_01TN7KgeW22qUYXpAZ1km5x1',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=389,
                        output_tokens=65,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 389,
                            'output_tokens': 65,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01Vv5XNafJW91mtvzWpquzKE',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id='toolu_01TN7KgeW22qUYXpAZ1km5x1',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="The current weather in London is sunny with a temperature of 22°C (about 72°F). It's a lovely day there!"
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=474,
                        output_tokens=32,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 474,
                            'output_tokens': 32,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'end_turn'},
                    provider_response_id='msg_018dCRfVee6LBNBk2mVcdSXD',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(anthropic_model, output_type=CityInfo, tools=[get_weather])
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
                            args={'city': 'Tokyo'},
                            tool_call_id='toolu_01K689q7kyiUpAJLmUrcgv5U',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=456,
                        output_tokens=38,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 456,
                            'output_tokens': 38,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01PYDKvwUXerptD6bPZPW6DB',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id='toolu_01K689q7kyiUpAJLmUrcgv5U',
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
                                'summary': "Tokyo is experiencing pleasant sunny weather with a temperature of 22°C (approximately 72°F). It's a beautiful day with clear skies and comfortable conditions for outdoor activities.",
                            },
                            tool_call_id='toolu_018A3vYtYhhY3rCUn7PXhghD',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=529,
                        output_tokens=90,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 529,
                            'output_tokens': 90,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01F1q4S2AYVheYxmwHSJQ8gC',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_018A3vYtYhhY3rCUn7PXhghD',
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

    async def test_none_prevents_function_tool_calls(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(anthropic_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Berlin?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="I'll get the current weather information for Berlin for you.")],
                    usage=RequestUsage(
                        input_tokens=389,
                        output_tokens=21,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 389,
                            'output_tokens': 21,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'end_turn'},
                    provider_response_id='msg_01BmeHTqb3j7MH86LrMcpBP6',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(anthropic_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the weather in Rome?", timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="I'll check the current weather in Rome for you.")],
                    usage=RequestUsage(
                        input_tokens=389,
                        output_tokens=20,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 389,
                            'output_tokens': 20,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'end_turn'},
                    provider_response_id='msg_01SeiSYVEpY5JmufuJcv7c8h',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(anthropic_model, output_type=CityInfo, tools=[get_weather])
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
                            args={
                                'city': 'Madrid',
                                'summary': "Madrid is the capital and largest city of Spain, located in the center of the country. With a population of over 3.3 million in the city proper and over 6.7 million in the metropolitan area, it's one of Europe's major cities. Madrid serves as the political, economic, and cultural center of Spain. The city is known for its rich history, world-renowned museums like the Prado and Reina Sofía, beautiful architecture including the Royal Palace, vibrant nightlife, and excellent cuisine. Madrid is also famous for its parks such as Retiro Park, its bustling squares like Puerta del Sol and Plaza Mayor, and being home to the Real Madrid football club. The city has a continental Mediterranean climate with hot summers and cool winters, and serves as a major transportation hub connecting Spain with the rest of Europe and Latin America.",
                            },
                            tool_call_id='toolu_01Gx72YwbVShHXoCADCx8si7',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=458,
                        output_tokens=229,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 458,
                            'output_tokens': 229,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_017U2zZ3egRP38362ezoqRDu',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01Gx72YwbVShHXoCADCx8si7',
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

    async def test_required_forces_tool_use(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await anthropic_model.request(
            [ModelRequest.user_text_prompt('Hello')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather',
                    args={'city': '<UNKNOWN>'},
                    tool_call_id='toolu_01X1UCKMiVV1cNcoJdVrZ1xh',
                )
            ]
        )

    async def test_required_with_multiple_tools(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await anthropic_model.request(
            [ModelRequest.user_text_prompt('Give me any information')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args={'city': 'New York'}, tool_call_id='toolu_01Cic7FT22GXi95knH2Ck33D'
                ),
                ToolCallPart(
                    tool_name='get_time', args={'timezone': 'UTC'}, tool_call_id='toolu_01MLA3JnFVxix421rMVzKjLC'
                ),
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await anthropic_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='toolu_01QW4RLuAsiA6ULWBdAGtAHB'
                ),
                ToolCallPart(
                    tool_name='get_population', args={'city': 'Paris'}, tool_call_id='toolu_0136CCBY7wvVS5XwkvE87bwo'
                ),
                ToolCallPart(
                    tool_name='get_time',
                    args={'timezone': 'Europe/Paris'},
                    tool_call_id='toolu_01U5oE5fAVMqK7GViGeGtYZE',
                ),
            ]
        )

    async def test_multiple_tools_in_list(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model can use any tool from the specified list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await anthropic_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time',
                    args={'timezone': 'Asia/Tokyo'},
                    tool_call_id='toolu_012FecB8EFyGW8yc4jWAcRRZ',
                )
            ]
        )

    async def test_excluded_tool_not_called(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Tools not in the list are not called."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await anthropic_model.request(
            [ModelRequest.user_text_prompt('What is the population of London?')],
            settings,
            params,
        )

        # Model must use get_weather since get_population is excluded
        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather', args={'city': 'London'}, tool_call_id='toolu_01Mwwq1TDTkfnaQn9YweCBj1'
                ),
                ToolCallPart(
                    tool_name='get_population', args={'city': 'London'}, tool_call_id='toolu_01TLoZJe8FkmD6DU1SVowAwF'
                ),
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            anthropic_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                            args={'city': 'Sydney'},
                            tool_call_id='toolu_01R24xmyRW3XMgWaHzEYUSXh',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=456,
                        output_tokens=38,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 456,
                            'output_tokens': 38,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_019dWhKVME8NTPfgs95H3Tpa',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id='toolu_01R24xmyRW3XMgWaHzEYUSXh',
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
                                'summary': 'The weather in Sydney is currently sunny with a pleasant temperature of 22°C, making it a beautiful day to enjoy outdoor activities.',
                            },
                            tool_call_id='toolu_01XpDx3DKwGaaVzZDzR7KDwJ',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=529,
                        output_tokens=80,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 529,
                            'output_tokens': 80,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01XoR3y5G1gJJ1kZW3c6NjL4',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01XpDx3DKwGaaVzZDzR7KDwJ',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            anthropic_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                            args={'city': 'Chicago'},
                            tool_call_id='toolu_01NPNj8rNoPbhst5ojfMzMce',
                        ),
                        ToolCallPart(
                            tool_name='get_population',
                            args={'city': 'Chicago'},
                            tool_call_id='toolu_01HDMPCroxEX9F8m3rWgga1H',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=516,
                        output_tokens=74,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 516,
                            'output_tokens': 74,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01S4v5FLCubmPAPRHM5dpqDL',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Chicago',
                            tool_call_id='toolu_01NPNj8rNoPbhst5ojfMzMce',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_population',
                            content='Chicago has 1 million people',
                            tool_call_id='toolu_01HDMPCroxEX9F8m3rWgga1H',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={
                                'city': 'Chicago',
                                'summary': 'Chicago currently has sunny weather at 22°C and a population of 1 million people.',
                            },
                            tool_call_id='toolu_01NYBEifhas93vutbdagJ596',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=683,
                        output_tokens=73,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 683,
                            'output_tokens': 73,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_014Y3FiQyhmQaZsEgA8KFFUd',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01NYBEifhas93vutbdagJ596',
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

    async def test_auto_with_only_output_tools(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(anthropic_model, output_type=CityInfo)

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
                            args={
                                'city': 'New York',
                                'summary': 'New York is the most populous city in the United States, located in New York State. It consists of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. Known as "The Big Apple" and "The City That Never Sleeps," New York is a global hub for finance, arts, fashion, and culture. It\'s home to iconic landmarks like the Statue of Liberty, Empire State Building, Times Square, Central Park, and Broadway. The city serves as a major center for international business and commerce, housing the New York Stock Exchange and numerous Fortune 500 companies. With its diverse population of over 8 million people, New York offers world-class museums, restaurants, entertainment, and is considered one of the world\'s most influential cities.',
                            },
                            tool_call_id='toolu_01WGHe291SmRoxGn8pNQWqMh',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=392,
                        output_tokens=216,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 392,
                            'output_tokens': 216,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_018LjvWcznHP8pExEEY5FUrb',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01WGHe291SmRoxGn8pNQWqMh',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(anthropic_model, output_type=CityInfo)
        settings: ModelSettings = {'tool_choice': 'none'}

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
                            args={
                                'city': 'Boston',
                                'summary': "Boston is the capital and largest city of Massachusetts, located in the northeastern United States. Founded in 1630, it's one of America's oldest cities and played a crucial role in the American Revolution, with historic sites like the Freedom Trail, Boston Tea Party Ships, and Paul Revere's House. The city is renowned for its world-class educational institutions including Harvard University and MIT in nearby Cambridge. Boston has a rich cultural scene with the Boston Symphony Orchestra, numerous museums, and is famous for its sports teams including the Red Sox, Celtics, Bruins, and Patriots. The city features distinct neighborhoods like Back Bay, North End, and Beacon Hill, and is known for its seafood, particularly clam chowder and lobster rolls. Boston's economy centers on finance, technology, healthcare, and education, making it a major hub for innovation and research.",
                            },
                            tool_call_id='toolu_01U6EvJNNJqGCETfejUatr82',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=391,
                        output_tokens=233,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 391,
                            'output_tokens': 233,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01SncZRQFPXD7Pgr569Jc2dU',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01U6EvJNNJqGCETfejUatr82',
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

    async def test_auto_with_union_output(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(anthropic_model, output_type=str | CityInfo, tools=[get_weather])
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
                            content='Get weather for Miami and describe it', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="I'll get the current weather for Miami and then provide you with a description."
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Miami'},
                            tool_call_id='toolu_017aKpUeGYUZVqnPFq1BLMt3',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=454,
                        output_tokens=70,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 454,
                            'output_tokens': 70,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01UGMqnUQmr9wTPRSPK5mZdx',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id='toolu_017aKpUeGYUZVqnPFq1BLMt3',
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
                                'summary': "The current weather in Miami is sunny with a pleasant temperature of 22°C (approximately 72°F). It's a beautiful day with clear skies and comfortable conditions - perfect weather for outdoor activities, beach visits, or simply enjoying the warm Florida sunshine.",
                            },
                            tool_call_id='toolu_01CQJRJ8dZjaNWSu9iG1SCPD',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=544,
                        output_tokens=121,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 544,
                            'output_tokens': 121,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01VmDELcvrJMQvRTRpahKiZP',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01CQJRJ8dZjaNWSu9iG1SCPD',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(self, anthropic_model: AnthropicModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(anthropic_model, output_type=str | CityInfo, tools=[get_weather])
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
                                'summary': 'Seattle is a major city in Washington State, located in the Pacific Northwest region of the United States. Known for its iconic Space Needle, thriving tech industry (home to companies like Amazon and Microsoft), vibrant coffee culture, and stunning natural setting between Puget Sound and the Cascade Mountains. The city experiences a temperate oceanic climate with mild, wet winters and relatively dry summers. Seattle is also famous for its music scene, having been the birthplace of grunge music, and offers rich cultural attractions including museums, theaters, and diverse neighborhoods.',
                            },
                            tool_call_id='toolu_01Byp8N9VkMnqUL7Qc6cvQeA',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=389,
                        output_tokens=181,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 389,
                            'output_tokens': 181,
                        },
                    ),
                    model_name='claude-sonnet-4-20250514',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                    provider_url='https://api.anthropic.com',
                    provider_details={'finish_reason': 'tool_use'},
                    provider_response_id='msg_01VGVCR4FiPje6ZYef5G4wCE',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='toolu_01Byp8N9VkMnqUL7Qc6cvQeA',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestThinkingModeRestrictions:
    """Tests for Anthropic-specific thinking mode restrictions.

    Anthropic does not support tool_choice='required' or forcing specific tools when
    thinking mode is enabled.
    """

    async def test_required_with_thinking_mode_raises_error(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """UserError is raised when tool_choice='required' with thinking mode enabled."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: AnthropicModelSettings = {
            'tool_choice': 'required',
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        with pytest.raises(UserError, match="Anthropic does not support `tool_choice='required'` with thinking mode"):
            await anthropic_model.request(
                [ModelRequest.user_text_prompt('Hello')],
                settings,
                params,
            )

    async def test_specific_tools_with_thinking_mode_raises_error(
        self, anthropic_model: AnthropicModel, allow_model_requests: None
    ):
        """UserError is raised when forcing specific tools with thinking mode enabled."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: AnthropicModelSettings = {
            'tool_choice': ['get_weather'],
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        with pytest.raises(UserError, match='Anthropic does not support forcing specific tools with thinking mode'):
            await anthropic_model.request(
                [ModelRequest.user_text_prompt('Hello')],
                settings,
                params,
            )
