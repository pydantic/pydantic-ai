""" """

from __future__ import annotations as _annotations

from collections.abc import Callable
from datetime import timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, ConfigDict

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.output import NativeOutput
from pydantic_ai.usage import RequestUsage

from ...conftest import IsNow, IsStr, try_import

# Import reusable test utilities from parent test module
from ..test_anthropic import MockAnthropic, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, omit as OMIT
    from anthropic.types.beta import BetaMessage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

# Type alias for the make_agent fixture
MakeAgentType = Callable[..., Agent]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_anthropic_native_output_sonnet_4_5(
    allow_model_requests: None,
    anthropic_sonnet_4_5: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test native JSON schema output with claude-sonnet-4-5 (supporting model)."""
    agent = make_agent(anthropic_sonnet_4_5, output_type=NativeOutput(city_location_schema))

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(city_location_schema(city='Paris', country='France'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Paris","country":"France"}')],
                usage=RequestUsage(
                    input_tokens=177,
                    output_tokens=12,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 177,
                        'output_tokens': 12,
                    },
                ),
                model_name=IsStr(),
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_native_output_with_tools(
    allow_model_requests: None,
    anthropic_sonnet_4_5: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test native output combined with tool calls."""
    agent = make_agent(anthropic_sonnet_4_5, output_type=NativeOutput(city_location_schema))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(city_location_schema(city='Mexico City', country='Mexico'))


async def test_anthropic_no_structured_output_support_sonnet_4_0(
    allow_model_requests: None,
    anthropic_sonnet_4_0: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test that claude-sonnet-4-0 raises when NativeOutput is used."""
    agent = make_agent(anthropic_sonnet_4_0, output_type=NativeOutput(city_location_schema))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        await agent.run('What is the capital of France?')


async def test_anthropic_no_structured_output_support_haiku_4_5(
    allow_model_requests: None,
    anthropic_haiku_4_5: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test haiku-4-5 behavior with native output (expected to raise)."""
    agent = make_agent(anthropic_haiku_4_5, output_type=NativeOutput(city_location_schema))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        await agent.run('What is the capital of France?')


def test_anthropic_native_output_strict_mode(
    allow_model_requests: None,
    mock_sonnet_4_5: tuple[AnthropicModel, AsyncAnthropic],
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test strict mode settings for native output."""
    model, mock_client = mock_sonnet_4_5

    # Explicit strict=True
    agent = make_agent(model, output_type=NativeOutput(city_location_schema, strict=True))
    agent.run_sync('What is the capital of Mexico?')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_format = completion_kwargs['output_format']
    betas = completion_kwargs['betas']
    assert output_format == snapshot(
        {
            'type': 'json_schema',
            'schema': snapshot(
                {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['city', 'country'],
                }
            ),
        }
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])

    # Explicit strict=False
    agent = make_agent(model, output_type=NativeOutput(city_location_schema, strict=False))
    agent.run_sync('What is the capital of Mexico?')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_format = completion_kwargs['output_format']
    betas = completion_kwargs['betas']
    assert output_format == snapshot(
        {
            'type': 'json_schema',
            'schema': snapshot(
                {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['city', 'country'],
                }
            ),
        }
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])

    # Strict-compatible (should auto-enable)
    agent = make_agent(model, output_type=NativeOutput(city_location_schema))
    agent.run_sync('What is the capital of Mexico?')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_format = completion_kwargs['output_format']
    betas = completion_kwargs['betas']
    assert output_format == snapshot(
        {
            'type': 'json_schema',
            'schema': snapshot(
                {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['city', 'country'],
                }
            ),
        }
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])

    # Strict-incompatible (with extras='allow')
    city_location_schema.model_config = ConfigDict(extra='allow')
    agent = make_agent(model, output_type=NativeOutput(city_location_schema))
    agent.run_sync('What is the capital of Mexico?')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_format = completion_kwargs['output_format']
    betas = completion_kwargs['betas']
    assert output_format == snapshot(
        {
            'type': 'json_schema',
            'schema': snapshot(
                {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['city', 'country'],
                }
            ),
        }
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])


def test_anthropic_native_output_merge_beta_headers(
    allow_model_requests: None,
    mock_sonnet_4_5: tuple[AnthropicModel, AsyncAnthropic],
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test that custom anthropic-beta headers are merged with structured output beta features."""
    from pydantic_ai.models.anthropic import AnthropicModelSettings

    model, mock_client = mock_sonnet_4_5

    # User provides their own beta feature via extra_headers
    agent = make_agent(
        model,
        output_type=NativeOutput(city_location_schema),
        model_settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-2025-01-01'}),
    )
    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    betas = completion_kwargs['betas']

    # Should merge custom beta with structured-outputs beta
    assert betas == snapshot(['custom-feature-2025-01-01', 'structured-outputs-2025-11-13'])


def test_anthropic_native_output_merge_beta_headers_comma_separated(
    allow_model_requests: None,
    mock_sonnet_4_5: tuple[AnthropicModel, AsyncAnthropic],
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test that comma-separated custom anthropic-beta headers are properly split and merged."""
    from pydantic_ai.models.anthropic import AnthropicModelSettings

    model, mock_client = mock_sonnet_4_5

    # User provides multiple beta features via comma-separated header
    agent = make_agent(
        model,
        output_type=NativeOutput(city_location_schema),
        model_settings=AnthropicModelSettings(
            extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2, custom-feature-3'}
        ),
    )
    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    betas = completion_kwargs['betas']

    # Should split comma-separated values and merge with structured-outputs beta (sorted)
    assert betas == snapshot(
        ['custom-feature-1', 'custom-feature-2', 'custom-feature-3', 'structured-outputs-2025-11-13']
    )


def test_anthropic_strict_tools_sonnet_4_5(allow_model_requests: None, weather_tool_responses: list[BetaMessage]):
    """Test that strict tool definitions are properly sent for supporting models."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(location: str) -> str:  # pragma: no cover
        return f'Weather in {location}'

    agent.run_sync('What is the weather in Paris?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs['betas']
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
                'strict': True,
            }
        ]
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])


def test_anthropic_strict_tools_sonnet_4_0(allow_model_requests: None, weather_tool_responses: list[BetaMessage]):
    """Test that strict is NOT sent for non-supporting models."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(location: str) -> str:  # pragma: no cover
        return f'Weather in {location}'

    agent.run_sync('What is the weather in Paris?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs.get('betas')
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
    assert betas is OMIT


async def test_anthropic_native_output_multiple(
    allow_model_requests: None,
    anthropic_sonnet_4_5: AnthropicModel,
    city_location_schema: type[BaseModel],
    country_language_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test native output with union of multiple schemas."""
    agent = make_agent(anthropic_sonnet_4_5, output_type=NativeOutput([city_location_schema, country_language_schema]))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'France'

    result = await agent.run('What is the capital of the user country?')
    # Should return CityLocation since we asked about capital
    assert isinstance(result.output, city_location_schema | country_language_schema)
    if isinstance(result.output, city_location_schema):
        assert result.output.city == 'Paris'  # type: ignore[attr-defined]
        assert result.output.country == 'France'  # type: ignore[attr-defined]
    else:  # pragma: no cover
        # This branch is not hit in this test, but we keep the structure for completeness
        pass


async def test_anthropic_auto_mode_sonnet_4_5(
    allow_model_requests: None,
    anthropic_sonnet_4_5: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test auto mode with sonnet-4.5 (should use native output automatically)."""
    agent = make_agent(anthropic_sonnet_4_5, output_type=city_location_schema)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(city_location_schema(city='Paris', country='France'))


async def test_anthropic_auto_mode_sonnet_4_0(
    allow_model_requests: None,
    anthropic_sonnet_4_0: AnthropicModel,
    city_location_schema: type[BaseModel],
    make_agent: MakeAgentType,
):
    """Test auto mode with sonnet-4.0 (should fall back to prompted output)."""
    agent = make_agent(anthropic_sonnet_4_0, output_type=city_location_schema)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(city_location_schema(city='Paris', country='France'))
