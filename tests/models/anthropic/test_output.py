"""Tests for Anthropic native JSON schema output and strict tool support.

This module tests the implementation of Anthropic's structured outputs feature,
including native JSON schema output for final responses and strict tool calling.

Test organization:
1. Strict Tools - Model Support
2. Strict Tools - Schema Compatibility
3. Native Output - Model Support
4. Auto Mode Selection
5. Beta Header Management
6. Comprehensive Parametrized Tests - All Combinations (24 test cases)
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from typing_extensions import assert_never

from pydantic_ai import Agent, Tool
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
    def get_weather(location: str) -> str:
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


# =============================================================================
# COMPREHENSIVE PARAMETRIZED TESTS - All Combinations
# =============================================================================


class LosslessSchema(BaseModel):
    """Simple schema with no validation constraints - fully strict-compatible."""

    location: str


class LossySchema(BaseModel):
    """Schema with validation constraints that get dropped - not strict-compatible."""

    username: Annotated[str, Field(min_length=3, pattern=r'^[a-z]+$')]


@dataclass
class StrictTestCase:
    """Defines a test case for strict mode behavior across models, schemas, and modes."""

    name: str
    model_name: str
    strict: bool | None
    schema_type: Literal['lossless', 'lossy']
    mode: Literal['tool', 'native']
    expect_strict_field: bool | None  # None means expect error
    expect_beta_header: bool | None  # None means expect error
    expect_error: type[Exception] | None = None


# =============================================================================
# TOOL CASES - Supported Model (claude-sonnet-4-5)
# =============================================================================

SUPPORTED_TOOL_STRICT_TRUE = [
    StrictTestCase(
        name='supported_tool-strict_true-lossless',
        model_name='claude-sonnet-4-5',
        strict=True,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
    StrictTestCase(
        name='supported_tool-strict_true-lossy',
        model_name='claude-sonnet-4-5',
        strict=True,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
]

SUPPORTED_TOOL_STRICT_NONE = [
    StrictTestCase(
        name='supported_tool-strict_auto-lossless-AUTO_ENABLED',
        model_name='claude-sonnet-4-5',
        strict=None,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
    StrictTestCase(
        name='supported_tool-strict_auto-lossy-NOT_AUTO_ENABLED',
        model_name='claude-sonnet-4-5',
        strict=None,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
]

SUPPORTED_TOOL_STRICT_FALSE = [
    StrictTestCase(
        name='supported_tool-strict_false-lossless',
        model_name='claude-sonnet-4-5',
        strict=False,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
    StrictTestCase(
        name='supported_tool-strict_false-lossy',
        model_name='claude-sonnet-4-5',
        strict=False,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
]


# =============================================================================
# TOOL CASES - Unsupported Model (claude-sonnet-4-0)
# =============================================================================

UNSUPPORTED_TOOL_STRICT_TRUE = [
    StrictTestCase(
        name='unsupported_tool-strict_true-lossless-MODEL_IGNORES',
        model_name='claude-sonnet-4-0',
        strict=True,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
    StrictTestCase(
        name='unsupported_tool-strict_true-lossy-MODEL_IGNORES',
        model_name='claude-sonnet-4-0',
        strict=True,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
]

UNSUPPORTED_TOOL_STRICT_NONE = [
    StrictTestCase(
        name='unsupported_tool-strict_auto-lossless-MODEL_UNSUPPORTED',
        model_name='claude-sonnet-4-0',
        strict=None,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
    StrictTestCase(
        name='unsupported_tool-strict_auto-lossy-MODEL_UNSUPPORTED',
        model_name='claude-sonnet-4-0',
        strict=None,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
]

UNSUPPORTED_TOOL_STRICT_FALSE = [
    StrictTestCase(
        name='unsupported_tool-strict_false-lossless',
        model_name='claude-sonnet-4-0',
        strict=False,
        schema_type='lossless',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
    StrictTestCase(
        name='unsupported_tool-strict_false-lossy',
        model_name='claude-sonnet-4-0',
        strict=False,
        schema_type='lossy',
        mode='tool',
        expect_strict_field=False,
        expect_beta_header=False,
    ),
]


# =============================================================================
# NATIVE OUTPUT CASES - Supported Model (claude-sonnet-4-5)
# =============================================================================

SUPPORTED_NATIVE_STRICT_TRUE = [
    StrictTestCase(
        name='supported_native-strict_true-lossless',
        model_name='claude-sonnet-4-5',
        strict=True,
        schema_type='lossless',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
    StrictTestCase(
        name='supported_native-strict_true-lossy',
        model_name='claude-sonnet-4-5',
        strict=True,
        schema_type='lossy',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
]

SUPPORTED_NATIVE_STRICT_NONE = [
    StrictTestCase(
        name='supported_native-strict_auto-lossless-FORCES_TRUE',
        model_name='claude-sonnet-4-5',
        strict=None,
        schema_type='lossless',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
    StrictTestCase(
        name='supported_native-strict_auto-lossy-FORCES_TRUE',
        model_name='claude-sonnet-4-5',
        strict=None,
        schema_type='lossy',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
]

SUPPORTED_NATIVE_STRICT_FALSE = [
    StrictTestCase(
        name='supported_native-strict_false-lossless-FORCES_TRUE',
        model_name='claude-sonnet-4-5',
        strict=False,
        schema_type='lossless',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
    StrictTestCase(
        name='supported_native-strict_false-lossy-FORCES_TRUE',
        model_name='claude-sonnet-4-5',
        strict=False,
        schema_type='lossy',
        mode='native',
        expect_strict_field=True,
        expect_beta_header=True,
    ),
]


# =============================================================================
# NATIVE OUTPUT CASES - Unsupported Model (claude-sonnet-4-0)
# =============================================================================

UNSUPPORTED_NATIVE_ALL = [
    StrictTestCase(
        name='unsupported_native-strict_true-lossless-RAISES',
        model_name='claude-sonnet-4-0',
        strict=True,
        schema_type='lossless',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
    StrictTestCase(
        name='unsupported_native-strict_true-lossy-RAISES',
        model_name='claude-sonnet-4-0',
        strict=True,
        schema_type='lossy',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
    StrictTestCase(
        name='unsupported_native-strict_auto-lossless-RAISES',
        model_name='claude-sonnet-4-0',
        strict=None,
        schema_type='lossless',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
    StrictTestCase(
        name='unsupported_native-strict_auto-lossy-RAISES',
        model_name='claude-sonnet-4-0',
        strict=None,
        schema_type='lossy',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
    StrictTestCase(
        name='unsupported_native-strict_false-lossless-RAISES',
        model_name='claude-sonnet-4-0',
        strict=False,
        schema_type='lossless',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
    StrictTestCase(
        name='unsupported_native-strict_false-lossy-RAISES',
        model_name='claude-sonnet-4-0',
        strict=False,
        schema_type='lossy',
        mode='native',
        expect_strict_field=None,
        expect_beta_header=None,
        expect_error=UserError,
    ),
]


# =============================================================================
# Combine All Cases
# =============================================================================

ALL_CASES = (
    SUPPORTED_TOOL_STRICT_TRUE
    + SUPPORTED_TOOL_STRICT_NONE
    + SUPPORTED_TOOL_STRICT_FALSE
    + UNSUPPORTED_TOOL_STRICT_TRUE
    + UNSUPPORTED_TOOL_STRICT_NONE
    + UNSUPPORTED_TOOL_STRICT_FALSE
    + SUPPORTED_NATIVE_STRICT_TRUE
    + SUPPORTED_NATIVE_STRICT_NONE
    + SUPPORTED_NATIVE_STRICT_FALSE
    + UNSUPPORTED_NATIVE_ALL
)


# =============================================================================
# Parametrized Test
# =============================================================================

ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]


def create_header_verification_hook(case: StrictTestCase):
    """Create an httpx event hook to verify request headers.

    NOTE: the vcr config doesn't record anthropic-beta headers.
    This hook allows us to verify them in live API tests.

    TODO: remove when structured outputs is generally available and no longer a beta feature.
    """

    async def verify_headers(request: httpx.Request):
        # Only verify for messages endpoint (the actual API calls)
        if '/messages' in str(request.url):  # pragma: no branch
            beta_header = request.headers.get('anthropic-beta', '')

            if case.expect_beta_header:
                assert 'structured-outputs-2025-11-13' in beta_header, (
                    f'Expected beta header for {case.name}, got: {beta_header}'
                )
            else:
                assert 'structured-outputs-2025-11-13' not in beta_header, (
                    f'Did not expect beta header for {case.name}, got: {beta_header}'
                )

    return verify_headers


@pytest.mark.parametrize('case', ALL_CASES, ids=lambda c: c.name)
@pytest.mark.vcr(record_mode='new_episodes')  # Allow recording new API interactions
def test_combinations_live_api(
    case: StrictTestCase,
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
) -> None:
    """Test strict mode across all combinations of models, schemas, and output modes with live API."""
    # live API model factory
    model = anthropic_model(case.model_name)

    # Add httpx event hook to verify headers on requests
    if case.expect_beta_header is not None:
        hook = create_header_verification_hook(case)
        model.client._client.event_hooks['request'].append(hook)  # pyright: ignore[reportPrivateUsage]

    assert model.profile.supports_json_schema_output == (case.model_name == 'claude-sonnet-4-5')

    if case.mode == 'tool':
        if case.schema_type == 'lossless':

            def pydantic_questions_tool(  # pyright: ignore[reportRedeclaration]
                question: LosslessSchema,
            ) -> str:  # pragma: no cover
                return 'Ask Samuel'
        else:  # lossy

            def pydantic_questions_tool(question: LossySchema) -> str:  # pragma: no cover
                return 'Ask Samuel'

        agent = Agent(model, tools=[Tool(pydantic_questions_tool, strict=case.strict)])

    elif case.mode == 'native':
        output_schema = LosslessSchema if case.schema_type == 'lossless' else LossySchema
        # we're using `NativeOutput` here, but supported models will automatically set its output_mode='native' without using `NativeOutput`.
        output_type = (
            NativeOutput(output_schema, strict=case.strict) if case.strict is not None else NativeOutput(output_schema)
        )
        agent = Agent(model, output_type=output_type)
    else:
        assert_never(case.mode)

    if case.expect_error:
        with pytest.raises(case.expect_error, match='Native structured output is not supported'):
            agent.run_sync('what is Pydantic?')
    else:
        # The request will include strict field and beta headers as configured
        result = agent.run_sync('what is Logfire?')

        # Verify we got a response
        assert result is not None, f'Expected response for {case.name}'

        # For native output, verify we got structured output
        if case.mode == 'native':
            assert hasattr(result, 'output'), f'Expected structured output for {case.name}'
            assert result.output is not None, f'Expected non-None output for {case.name}'
