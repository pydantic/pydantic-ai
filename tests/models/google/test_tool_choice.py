"""Google tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the Google API.
Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live Google API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.

Google-specific behavior:
- `tool_choice='auto'` maps to mode: AUTO
- `tool_choice='none'` maps to mode: NONE
- `tool_choice='required'` maps to mode: ANY
- Specific tools use `allowed_function_names` parameter
"""

from __future__ import annotations

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
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# =============================================================================
# Tool definitions
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
def google_model(gemini_api_key: str) -> GoogleModel:
    """Create a Google model for testing."""
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    Google maps 'auto' to mode: AUTO.
    """

    async def test_auto_with_function_tools_uses_tool(self, google_model: GoogleModel, allow_model_requests: None):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(google_model, tools=[get_weather])
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
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=49, output_tokens=72, details={'thoughts_tokens': 57, 'text_prompt_tokens': 49}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='VsVFabT9Kbbhz7IPuqzrmA8',
                    finish_reason='stop',
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
                    parts=[TextPart(content='The weather in Paris is sunny with a temperature of 22C.')],
                    usage=RequestUsage(input_tokens=88, output_tokens=15, details={'text_prompt_tokens': 88}),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='V8VFacv5FM-Gz7IPuICtsQQ',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, google_model: GoogleModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(google_model, tools=[get_weather])
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
                    parts=[TextPart(content='Hello')],
                    usage=RequestUsage(input_tokens=46, output_tokens=1, details={'text_prompt_tokens': 46}),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='WMVFafynEfiHz7IPqsGuuAQ',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(self, google_model: GoogleModel, allow_model_requests: None):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(google_model, tools=[get_weather])

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
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=49, output_tokens=82, details={'thoughts_tokens': 67, 'text_prompt_tokens': 49}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='WMVFaZSmOP2nmtkPvb_b0Q0',
                    finish_reason='stop',
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
                    parts=[TextPart(content='The weather in London is Sunny, 22C.')],
                    usage=RequestUsage(input_tokens=88, output_tokens=12, details={'text_prompt_tokens': 88}),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='W8VFaez2C-6eqtsP_Ye5kAQ',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(self, google_model: GoogleModel, allow_model_requests: None):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(google_model, output_type=CityInfo, tools=[get_weather])
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
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=95, output_tokens=123, details={'thoughts_tokens': 108, 'text_prompt_tokens': 95}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='XMVFacLELZS9z7IP7Ybl0Qk',
                    finish_reason='stop',
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
                                'summary': 'The weather in Tokyo is Sunny with a temperature of 22C.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=134, output_tokens=141, details={'thoughts_tokens': 108, 'text_prompt_tokens': 134}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='XsVFaZiQCPWcz7IPjPDb6QU',
                    finish_reason='stop',
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

    When tool_choice is 'none' or [], function tools are disabled but output tools remain.
    Google maps 'none' to mode: NONE.
    """

    async def test_none_prevents_function_tool_calls(self, google_model: GoogleModel, allow_model_requests: None):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(google_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1500),
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
The weather in Berlin, Germany is currently:

*   **Temperature:** 14°C (57°F)
*   **Conditions:** Overcast Clouds
*   **Feels like:** 13°C (55°F)
*   **Humidity:** 86%
*   **Wind:** 10 km/h (6 mph) from the Northwest

There's a chance of light rain later this afternoon.\
"""
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=49, output_tokens=983, details={'thoughts_tokens': 888, 'text_prompt_tokens': 49}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='Y8VFaaaNL7bhz7IPyJ-O6QQ',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(self, google_model: GoogleModel, allow_model_requests: None):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(google_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1500),
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
The weather in Rome, Italy is currently **22°C (72°F)** and it's **partly cloudy**.

Here's a bit more detail:
*   **Feels like:** 23°C (73°F)
*   **Wind:** Light breeze from the West at about 8 km/h (5 mph)
*   **Humidity:** Around 65%

For the rest of today, you can expect a high of around **25°C (77°F)** with continued partly cloudy skies. Tonight will be mostly clear with a low around **16°C (61°F)**.\
"""
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=49, output_tokens=779, details={'thoughts_tokens': 639, 'text_prompt_tokens': 49}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='Z8VFaca-O6yDz7IPjrHK-QU',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, google_model: GoogleModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(google_model, output_type=CityInfo, tools=[get_weather])
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
                                'summary': 'Madrid is the capital and most populous city of Spain. The city has a rich history, evident in its architecture and museums.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=93, output_tokens=104, details={'thoughts_tokens': 61, 'text_prompt_tokens': 93}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='acVFaeTzF9XRz7IP9Za-kQY',
                    finish_reason='stop',
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
    Google maps 'required' to mode: ANY.
    Output tools are NOT included - this is for direct model requests.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_required_forces_tool_use(self, google_model: GoogleModel, allow_model_requests: None):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await google_model.request(
            [ModelRequest.user_text_prompt('Hello')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather',
                    args={'city': 'London'},
                    tool_call_id=IsStr(),
                    provider_details={
                        'thought_signature': IsStr()
                    },
                )
            ]
        )

    async def test_required_with_multiple_tools(self, google_model: GoogleModel, allow_model_requests: None):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await google_model.request(
            [ModelRequest.user_text_prompt('Give me any information')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time',
                    args={'timezone': 'America/Los_Angeles'},
                    tool_call_id=IsStr(),
                    provider_details={
                        'thought_signature': IsStr()
                    },
                )
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    Google uses `allowed_function_names` parameter for this.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, google_model: GoogleModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await google_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather',
                    args={'city': 'Paris'},
                    tool_call_id=IsStr(),
                    provider_details={
                        'thought_signature': IsStr()
                    },
                )
            ]
        )

    async def test_multiple_tools_in_list(self, google_model: GoogleModel, allow_model_requests: None):
        """Model can use any tool from the specified list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await google_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_time',
                    args={'timezone': 'Asia/Tokyo'},
                    tool_call_id=IsStr(),
                    provider_details={
                        'thought_signature': IsStr()
                    },
                )
            ]
        )

    async def test_excluded_tool_not_called(self, google_model: GoogleModel, allow_model_requests: None):
        """Tools not in the list are not called."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await google_model.request(
            [ModelRequest.user_text_prompt('What is the population of London?')],
            settings,
            params,
        )

        # Model must use get_weather since get_population is excluded
        assert response.parts == snapshot(
            [
                ToolCallPart(
                    tool_name='get_weather',
                    args={'city': 'London'},
                    tool_call_id=IsStr(),
                    provider_details={
                        'thought_signature': IsStr()
                    },
                )
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, google_model: GoogleModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            google_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=174, output_tokens=121, details={'thoughts_tokens': 106, 'text_prompt_tokens': 174}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='ccVFaZHOE7uDz7IPnZCmyQY',
                    finish_reason='stop',
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
                            args={'city': 'Sydney', 'summary': 'The weather in Sydney is Sunny, 22C.'},
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=213, output_tokens=100, details={'thoughts_tokens': 70, 'text_prompt_tokens': 213}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='csVFaeLtG8-Gz7IPuYCtsQQ',
                    finish_reason='stop',
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
        self, google_model: GoogleModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            google_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
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
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Chicago'},
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        ),
                        ToolCallPart(
                            tool_name='get_population',
                            args={'city': 'Chicago'},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=174, output_tokens=184, details={'thoughts_tokens': 154, 'text_prompt_tokens': 174}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='dMVFab2qLojVz7IP4__06AU',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Chicago',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='get_population',
                            content='Chicago has 1 million people',
                            tool_call_id=IsStr(),
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
                                'summary': 'The weather in Chicago is Sunny, 22C. Chicago has 1 million people.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=248, output_tokens=147, details={'thoughts_tokens': 110, 'text_prompt_tokens': 248}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='dsVFaen6DbHVz7IPr4jS6AU',
                    finish_reason='stop',
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

    async def test_auto_with_only_output_tools(self, google_model: GoogleModel, allow_model_requests: None):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(google_model, output_type=CityInfo)

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
                                'summary': 'New York City is a major global hub for finance, fashion, art, and culture. It is known for its iconic landmarks like the Statue of Liberty, Times Square, and Central Park, as well as its diverse neighborhoods and Broadway shows.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=54, output_tokens=245, details={'thoughts_tokens': 178, 'text_prompt_tokens': 54}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='ecVFabTpKMaHz7IPmYyeWA',
                    finish_reason='stop',
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

    async def test_none_with_only_output_tools(self, google_model: GoogleModel, allow_model_requests: None):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(google_model, output_type=CityInfo)
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
                                'summary': 'Boston is a historic city known for its role in the American Revolution, its prestigious universities like Harvard and MIT, and its vibrant cultural scene.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=53, output_tokens=105, details={'thoughts_tokens': 59, 'text_prompt_tokens': 53}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='e8VFafu0B7uDz7IP-pimyQY',
                    finish_reason='stop',
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

    async def test_auto_with_union_output(self, google_model: GoogleModel, allow_model_requests: None):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(google_model, output_type=str | CityInfo, tools=[get_weather])
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
                        ToolCallPart(
                            tool_name='get_weather',
                            args={'city': 'Miami'},
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=96, output_tokens=139, details={'thoughts_tokens': 124, 'text_prompt_tokens': 96}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='fMVFaZnUKdXRz7IP9Za-kQY',
                    finish_reason='stop',
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
                                'summary': 'The weather in Miami is Sunny and 22C.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=135, output_tokens=205, details={'thoughts_tokens': 175, 'text_prompt_tokens': 135}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='fsVFaZ7qEKutz7IP56Dm-AI',
                    finish_reason='stop',
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

    async def test_none_with_union_output(self, google_model: GoogleModel, allow_model_requests: None):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(google_model, output_type=str | CityInfo, tools=[get_weather])
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
                                'summary': 'Seattle is a major coastal city in the Pacific Northwest region of the United States. It is known for its rainy weather, vibrant tech industry, iconic Space Needle, and stunning natural beauty with proximity to mountains and water.',
                            },
                            tool_call_id=IsStr(),
                            provider_details={
                                'thought_signature': IsStr()
                            },
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=52, output_tokens=118, details={'thoughts_tokens': 57, 'text_prompt_tokens': 52}
                    ),
                    model_name='gemini-2.5-flash',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='f8VFadeqMNXRz7IP9Za-kQY',
                    finish_reason='stop',
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
