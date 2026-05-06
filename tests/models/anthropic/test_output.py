"""Tests for Anthropic native JSON schema output and strict tool support.

This module tests the implementation of Anthropic's structured outputs feature,
including native JSON schema output for final responses and strict tool calling.

Test organization:
1. Strict Tools - Model Support
2. Strict Tools - Schema Compatibility
3. Native Output - Model Support
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.output import NativeOutput

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..test_anthropic import MockAnthropic, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, omit as OMIT
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from pydantic_ai.models.anthropic import AnthropicModel

    ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]

from ..test_anthropic import completion_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        "ignore:The model 'claude-sonnet-4-0' is deprecated and will reach end-of-life.*:DeprecationWarning"
    ),
]


# =============================================================================
# STRICT TOOLS - Model Support
# =============================================================================


def test_strict_tools_supported_model_auto_enabled(
    allow_model_requests: None, weather_tool_responses: list[BetaMessage]
):
    """sonnet-4-5: strict=None + compatible schema → no strict field."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(location: str) -> str:
        return f'Weather in {location}'

    agent.run_sync('What is the weather in Paris?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs['betas']
    tool = tools[0]
    assert 'strict' not in tool  # strict was not explicitly set

    assert tools == snapshot(
        [
            {
                'name': 'get_weather',
                'description': '',
                'input_schema': {
                    'type': 'object',
                    'properties': {'location': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['location'],
                },
            }
        ]
    )
    assert betas == OMIT


def test_strict_tools_supported_model_explicit_false(
    allow_model_requests: None, weather_tool_responses: list[BetaMessage]
):
    """sonnet-4-5: strict=False → no strict field."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def get_weather(location: str) -> str:
        return f'Weather in {location}'

    agent.run_sync('What is the weather in Paris?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs.get('betas')

    assert 'strict' not in tools[0]
    assert tools[0]['input_schema']['additionalProperties'] is False
    assert betas is OMIT


def test_strict_tools_unsupported_model_no_strict_sent(
    allow_model_requests: None, weather_tool_responses: list[BetaMessage]
):
    """sonnet-4-0: strict=None → no strict field (model doesn't support strict)."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(location: str) -> str:
        return f'Weather in {location}'

    agent.run_sync('What is the weather in Paris?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs.get('betas')

    # sonnet-4-0 doesn't support strict tools, so no strict field or beta header
    assert 'strict' not in tools[0]
    assert betas is OMIT


# =============================================================================
# STRICT TOOLS - Schema Compatibility
# =============================================================================


def test_strict_tools_incompatible_schema_not_auto_enabled(allow_model_requests: None):
    """sonnet-4-5: strict=None → no strict field."""
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='Sure', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    )
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def constrained_tool(username: Annotated[str, Field(min_length=3)]) -> str:  # pragma: no cover
        return username

    agent.run_sync('Test')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs.get('betas')

    # strict is not auto-enabled, so no strict field
    assert 'strict' not in tools[0]
    # because the schema wasn't transformed, it keeps the pydantic constraint
    assert tools[0]['input_schema']['properties']['username']['minLength'] == 3
    assert betas is OMIT


# =============================================================================
# NATIVE OUTPUT - Model Support
# =============================================================================


def test_native_output_supported_model(
    allow_model_requests: None,
    mock_sonnet_4_5: tuple[AnthropicModel, AsyncAnthropic],
    city_location_schema: type[BaseModel],
):
    """sonnet-4-5: NativeOutput → strict=True + output_config."""
    model, mock_client = mock_sonnet_4_5
    agent = Agent(model, output_type=NativeOutput(city_location_schema))

    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_config = completion_kwargs['output_config']

    assert output_config['format']['type'] == 'json_schema'
    assert output_config['format']['schema']['type'] == 'object'
    assert completion_kwargs['betas'] is OMIT


# =============================================================================
# COMPREHENSIVE INTEGRATION TESTS - All Combinations
# =============================================================================


class CityInfo(BaseModel):
    """Information about a city."""

    city: str
    country: str
    population: int


# =============================================================================
# Supported Model Tests (claude-sonnet-4-5)
# =============================================================================


@pytest.mark.vcr
def test_no_tools_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with no tools and no output_type."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model)
    agent.run_sync('Tell me a brief fact about Paris')


@pytest.mark.vcr
def test_no_tools_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with no tools and BaseModel output_type."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=CityInfo)
    agent.run_sync('Give me information about Tokyo')


@pytest.mark.vcr
def test_no_tools_native_output_strict_true(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with NativeOutput(strict=True) → output_config."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo, strict=True))
    result = agent.run_sync('Tell me about London')

    assert isinstance(result.output, CityInfo)


@pytest.mark.vcr
def test_no_tools_native_output_strict_none(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with NativeOutput(strict=None) → forces strict=True, output_config."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo))
    result = agent.run_sync('Give me facts about Berlin')

    assert isinstance(result.output, CityInfo)


def test_no_tools_native_output_strict_false(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with NativeOutput(strict=False) → raises UserError."""
    model = anthropic_model('claude-sonnet-4-5')

    agent = Agent(model, output_type=NativeOutput(CityInfo, strict=False))

    with pytest.raises(
        UserError,
        match='Setting `strict=False` on `output_type=NativeOutput\\(\\.\\.\\.\\)` is not allowed for Anthropic models.',
    ):
        agent.run_sync('Tell me about Rome')


@pytest.mark.vcr
def test_strict_true_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=True, no output_type → tool has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 22°C'

    agent.run_sync("What's the weather in San Francisco?")


@pytest.mark.vcr
def test_strict_true_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=True, BaseModel output_type → tool has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 8_000_000 if city == 'New York' else 1_000_000

    agent.run_sync('Get me info about New York including its population')


@pytest.mark.vcr
def test_strict_true_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=True, NativeOutput → tool has strict field + output_config."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=True)
    def lookup_country(city: str) -> str:
        return 'France' if city == 'Paris' else 'Unknown'

    result = agent.run_sync('Give me details about Paris')

    assert isinstance(result.output, CityInfo)


@pytest.mark.vcr
def test_strict_none_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, no output_type → tool has no strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model)

    @agent.tool_plain
    def search_database(query: str) -> str:
        return f'Found 42 results for "{query}"'

    agent.run_sync('Find cities in Europe')


@pytest.mark.vcr
def test_strict_none_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, BaseModel output_type → tool has no strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain
    def get_timezone(city: str) -> str:  # pragma: no cover
        return 'UTC+10:00' if city == 'Sydney' else 'UTC+1:00'

    agent.run_sync('Give me info about Sydney including its timezone')


@pytest.mark.vcr
def test_strict_none_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, NativeOutput → output_config, tool has no strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain
    def get_coordinates(city: str) -> str:
        return '41.3874° N, 2.1686° E' if city == 'Barcelona' else 'Unknown'

    result = agent.run_sync('Give me details about Barcelona')

    assert isinstance(result.output, CityInfo)


@pytest.mark.vcr
def test_strict_false_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=False, no output_type."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def calculate_distance(city_a: str, city_b: str) -> str:
        return f'Distance from {city_a} to {city_b}: 504 km'

    agent.run_sync('How far is Madrid from Lisbon?')


@pytest.mark.vcr
def test_strict_false_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=False, NativeOutput → output_config."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo))
    tool_called = False

    @agent.tool_plain(strict=False)
    def get_currency(country: str) -> str:
        nonlocal tool_called
        tool_called = True
        return 'Mexican Peso (MXN)' if country == 'Mexico' else 'Unknown'

    result = agent.run_sync('Give me details about Mexico City. Use available background tools where helpful.')

    # Ensure the cassette keeps exercising the tool-result turn.
    assert tool_called
    assert isinstance(result.output, CityInfo)


@pytest.mark.vcr
def test_mixed_tools_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), no output_type → only strict=True has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 22°C'

    @agent.tool_plain
    def get_elevation(city: str) -> str:
        return f'Elevation of {city}: 650m above sea level'

    agent.run_sync("What's the weather and elevation in Denver?")


@pytest.mark.vcr
def test_mixed_tools_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), BaseModel output_type → only strict=True has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 8_900_000 if city == 'London' else 1_000_000

    @agent.tool_plain
    def get_area(city: str) -> str:
        return f'Area of {city}: 1,572 km²'

    agent.run_sync('Tell me about London including population and area')


@pytest.mark.vcr
def test_mixed_tools_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), NativeOutput → only strict=True has strict field + output_config."""
    model = anthropic_model('claude-sonnet-4-5')
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=True)
    def lookup_country(city: str) -> str:
        return 'Japan' if city == 'Tokyo' else 'Unknown'

    @agent.tool_plain
    def get_founded_year(city: str) -> str:
        return '1457' if city == 'Tokyo' else 'Unknown'

    result = agent.run_sync('Give me complete details about Tokyo')

    assert isinstance(result.output, CityInfo)


# =============================================================================
# Unsupported Model Tests (claude-sonnet-4-0)
# =============================================================================


@pytest.mark.vcr
def test_unsupported_strict_true_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: tool with strict=True, no output_type → no strict field."""
    model = anthropic_model('claude-sonnet-4-0')
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 18°C'

    agent.run_sync("What's the weather in Amsterdam?")


@pytest.mark.vcr
def test_unsupported_strict_true_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: tool with strict=True, BaseModel output_type → no strict field."""
    model = anthropic_model('claude-sonnet-4-0')
    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 850_000 if city == 'Amsterdam' else 1_000_000

    agent.run_sync('Get me details about Amsterdam including its population')


def test_unsupported_native_output_raises(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: NativeOutput → raises UserError."""
    model = anthropic_model('claude-sonnet-4-0')

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        agent.run_sync('Tell me about Berlin')
