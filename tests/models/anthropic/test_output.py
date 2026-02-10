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

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.output import NativeOutput

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
]


# =============================================================================
# STRICT TOOLS - Model Support
# =============================================================================


def test_strict_tools_supported_model_auto_enabled(
    allow_model_requests: None, weather_tool_responses: list[BetaMessage]
):
    """sonnet-4-5: strict=None + compatible schema → no strict field, no beta header."""
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
    """sonnet-4-5: strict=False → no strict field, no beta header."""
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
    """sonnet-4-0: strict=None → no strict field, no beta header (model doesn't support strict)."""
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
    """sonnet-4-5: strict=None → no strict field, no beta header."""
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
    """sonnet-4-5: NativeOutput → strict=True + beta header + output_config."""
    model, mock_client = mock_sonnet_4_5
    agent = Agent(model, output_type=NativeOutput(city_location_schema))

    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_config = completion_kwargs['output_config']
    betas = completion_kwargs['betas']

    assert output_config['format']['type'] == 'json_schema'
    assert output_config['format']['schema']['type'] == 'object'
    assert betas == snapshot(['structured-outputs-2025-11-13'])


# =============================================================================
# COMPREHENSIVE INTEGRATION TESTS - All Combinations
# =============================================================================


class CityInfo(BaseModel):
    """Information about a city."""

    city: str
    country: str
    population: int


def create_header_verification_hook(expect_beta: bool, test_name: str):
    """Create an httpx event hook to verify request headers.

    NOTE: the vcr config doesn't record anthropic-beta headers.
    This hook allows us to verify them in live API tests.

    TODO: remove when structured outputs is generally available and no longer a beta feature.
    """
    errors: list[str] = []

    async def verify_headers(request: httpx.Request):
        # Only verify for messages endpoint
        if '/messages' in str(request.url):  # pragma: no branch
            beta_header = request.headers.get('anthropic-beta', '')

            # excluded from coverage cause the if's shouldn't trigger when the tests are passing
            if expect_beta:
                if 'structured-outputs-2025-11-13' not in beta_header:  # pragma: no cover
                    errors.append(
                        f'Test "{test_name}": Expected beta header '
                        f'"structured-outputs-2025-11-13" but got: {beta_header!r}'
                    )
            else:
                if 'structured-outputs-2025-11-13' in beta_header:  # pragma: no cover
                    errors.append(
                        f'Test "{test_name}": Did not expect beta header '
                        f'"structured-outputs-2025-11-13" but got: {beta_header!r}'
                    )

    verify_headers.errors = errors  # type: ignore[attr-defined]
    return verify_headers


# =============================================================================
# Supported Model Tests (claude-sonnet-4-5)
# =============================================================================


@pytest.mark.vcr
def test_no_tools_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with no tools and no output_type → no beta header."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_no_tools_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)
    agent.run_sync('Tell me a brief fact about Paris')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_no_tools_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with no tools and BaseModel output_type → no beta header."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_no_tools_basemodel_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=CityInfo)
    agent.run_sync('Give me information about Tokyo')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_no_tools_native_output_strict_true(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with NativeOutput(strict=True) → beta header + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_no_tools_native_output_strict_true')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo, strict=True))
    result = agent.run_sync('Tell me about London')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_no_tools_native_output_strict_none(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Agent with NativeOutput(strict=None) → forces strict=True, beta header + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_no_tools_native_output_strict_none')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo))
    result = agent.run_sync('Give me facts about Berlin')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


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
    """Tool with strict=True, no output_type → beta header, tool has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_strict_true_tool_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 22°C'

    agent.run_sync("What's the weather in San Francisco?")

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_true_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=True, BaseModel output_type → beta header, tool has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_strict_true_tool_basemodel_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 8_000_000 if city == 'New York' else 1_000_000

    agent.run_sync('Get me info about New York including its population')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_true_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=True, NativeOutput → beta header, tool has strict field + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_strict_true_tool_native_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=True)
    def lookup_country(city: str) -> str:
        return 'France' if city == 'Paris' else 'Unknown'

    result = agent.run_sync('Give me details about Paris')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_none_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, no output_type → no beta header, tool has no strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_strict_none_tool_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)

    @agent.tool_plain
    def search_database(query: str) -> str:
        return f'Found 42 results for "{query}"'

    agent.run_sync('Find cities in Europe')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_none_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, BaseModel output_type → no beta header, tool has no strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_strict_none_tool_basemodel_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain
    def get_timezone(city: str) -> str:  # pragma: no cover
        return 'UTC+10:00' if city == 'Sydney' else 'UTC+1:00'

    agent.run_sync('Give me info about Sydney including its timezone')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_none_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=None, NativeOutput → beta from native only, tool has no strict field + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_strict_none_tool_native_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain
    def get_coordinates(city: str) -> str:
        return '41.3874° N, 2.1686° E' if city == 'Barcelona' else 'Unknown'

    result = agent.run_sync('Give me details about Barcelona')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_false_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=False, no output_type → no beta header."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_strict_false_tool_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def calculate_distance(city_a: str, city_b: str) -> str:
        return f'Distance from {city_a} to {city_b}: 504 km'

    agent.run_sync('How far is Madrid from Lisbon?')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_strict_false_tool_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Tool with strict=False, NativeOutput → beta from native only + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_strict_false_tool_native_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=False)
    def get_currency(country: str) -> str:
        return 'Mexican Peso (MXN)' if country == 'Mexico' else 'Unknown'

    result = agent.run_sync('Give me details about Mexico City')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_mixed_tools_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), no output_type → beta, only strict=True has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_mixed_tools_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 22°C'

    @agent.tool_plain
    def get_elevation(city: str) -> str:
        return f'Elevation of {city}: 650m above sea level'

    agent.run_sync("What's the weather and elevation in Denver?")

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_mixed_tools_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), BaseModel output_type → beta, only strict=True has strict field."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_mixed_tools_basemodel_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 8_900_000 if city == 'London' else 1_000_000

    @agent.tool_plain
    def get_area(city: str) -> str:
        return f'Area of {city}: 1,572 km²'

    agent.run_sync('Tell me about London including population and area')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_mixed_tools_native_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Mixed tools (one strict=True, one strict=None), NativeOutput → beta, only strict=True has strict field + output_format."""
    model = anthropic_model('claude-sonnet-4-5')
    hook = create_header_verification_hook(expect_beta=True, test_name='test_mixed_tools_native_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=True)
    def lookup_country(city: str) -> str:
        return 'Japan' if city == 'Tokyo' else 'Unknown'

    @agent.tool_plain
    def get_founded_year(city: str) -> str:
        return '1457' if city == 'Tokyo' else 'Unknown'

    result = agent.run_sync('Give me complete details about Tokyo')

    assert isinstance(result.output, CityInfo)
    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


# =============================================================================
# Unsupported Model Tests (claude-sonnet-4-0)
# =============================================================================


@pytest.mark.vcr
def test_unsupported_strict_true_tool_no_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: tool with strict=True, no output_type → no beta, no strict field."""
    model = anthropic_model('claude-sonnet-4-0')
    hook = create_header_verification_hook(expect_beta=False, test_name='test_unsupported_strict_true_tool_no_output')
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 18°C'

    agent.run_sync("What's the weather in Amsterdam?")

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


@pytest.mark.vcr
def test_unsupported_strict_true_tool_basemodel_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: tool with strict=True, BaseModel output_type → no beta, no strict field."""
    model = anthropic_model('claude-sonnet-4-0')
    hook = create_header_verification_hook(
        expect_beta=False, test_name='test_unsupported_strict_true_tool_basemodel_output'
    )
    model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(model, output_type=CityInfo)

    @agent.tool_plain(strict=True)
    def get_population(city: str) -> int:
        return 850_000 if city == 'Amsterdam' else 1_000_000

    agent.run_sync('Get me details about Amsterdam including its population')

    if errors := hook.errors:  # type: ignore[attr-defined]
        assert False, '\n'.join(sorted(errors))


def test_unsupported_native_output_raises(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Unsupported model: NativeOutput → raises UserError."""
    model = anthropic_model('claude-sonnet-4-0')

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        agent.run_sync('Tell me about Berlin')
