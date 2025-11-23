"""Tests for Anthropic native JSON schema output and strict tool support.

This module tests the implementation of Anthropic's structured outputs feature,
including native JSON schema output for final responses and strict tool calling.

Test organization:
1. Strict Tools - Model Support
2. Strict Tools - Schema Compatibility
3. Native Output - Model Support
4. Auto Mode Selection
5. Beta Header Management
6. Edge Cases
"""

from __future__ import annotations as _annotations

from typing import Annotated

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

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

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
    """sonnet-4-5: strict=None + compatible schema → auto strict=True + beta header."""
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
                # strict is set automatically because the model supports it
                'strict': True,
            }
        ]
    )
    assert betas == snapshot(['structured-outputs-2025-11-13'])


def test_strict_tools_supported_model_explicit_false(
    allow_model_requests: None, weather_tool_responses: list[BetaMessage]
):
    """sonnet-4-5: strict=False → no strict field, no beta header."""
    mock_client = MockAnthropic.create_mock(weather_tool_responses)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    def get_weather(location: str) -> str:  # pragma: no cover
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
    def get_weather(location: str) -> str:  # pragma: no cover
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
    """sonnet-4-5: strict=None + lossy schema → no strict field, no beta header."""
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

    # Lossy schema: strict is not auto-enabled, so no strict field
    assert 'strict' not in tools[0]
    # Schema still has the constraint (not removed)
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
    """sonnet-4-5: NativeOutput → strict=True + beta header + output_format."""
    model, mock_client = mock_sonnet_4_5
    agent = Agent(model, output_type=NativeOutput(city_location_schema))

    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    output_format = completion_kwargs['output_format']
    betas = completion_kwargs['betas']

    assert output_format['type'] == 'json_schema'
    assert output_format['schema']['type'] == 'object'
    assert betas == snapshot(['structured-outputs-2025-11-13'])


def test_native_output_unsupported_model_raises_error(
    allow_model_requests: None, city_location_schema: type[BaseModel]
):
    """sonnet-4-0: NativeOutput → raises UserError."""
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='test', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    )
    model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model, output_type=NativeOutput(city_location_schema))

    with pytest.raises(UserError, match='Native structured output is not supported by this model'):
        agent.run_sync('What is the capital of France?')


# =============================================================================
# AUTO MODE Selection
# =============================================================================


def test_auto_mode_model_profile_check(allow_model_requests: None):
    """Verify profile.supports_json_schema_output is set correctly."""
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='test', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    )

    sonnet_4_5 = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    assert sonnet_4_5.profile.supports_json_schema_output is True

    sonnet_4_0 = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(anthropic_client=mock_client))
    assert sonnet_4_0.profile.supports_json_schema_output is False


# =============================================================================
# BETA HEADER Management
# =============================================================================


def test_beta_header_merge_custom_headers(
    allow_model_requests: None,
    mock_sonnet_4_5: tuple[AnthropicModel, AsyncAnthropic],
    city_location_schema: type[BaseModel],
):
    """Custom beta headers merge with structured-outputs beta."""
    model, mock_client = mock_sonnet_4_5

    agent = Agent(
        model,
        output_type=NativeOutput(city_location_schema),
        model_settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2'}),
    )
    agent.run_sync('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    betas = completion_kwargs['betas']

    assert betas == snapshot(['custom-feature-1', 'custom-feature-2', 'structured-outputs-2025-11-13'])
