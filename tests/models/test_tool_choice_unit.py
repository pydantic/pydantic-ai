"""Unit tests for the `resolve_tool_choice` function and provider-specific tool_choice handling.

The provider-specific tests use model.request() directly because they test error paths that are
only reachable during direct model requests. Agent.run() validates tool_choice at a higher level
and blocks 'required' and list[str] values before they reach the model-specific validation code.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._tool_choice import resolve_tool_choice
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _support_tool_forcing as anthropic_support_tool_forcing,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import (
        BedrockConverseModel,
        BedrockModelSettings,
        _support_tool_forcing as bedrock_support_tool_forcing,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.profiles.grok import GrokModelProfile
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.anyio


def make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name)


# =============================================================================
# resolve_tool_choice tests
# =============================================================================


SIMPLE_CASES = [
    dict(
        id='auto_with_text_output',
        tool_choice='auto',
        params_kwargs={'allow_text_output': True},
        expected='auto',
    ),
    dict(
        id='auto_without_text_output',
        tool_choice='auto',
        params_kwargs={'function_tools': [make_tool('x')], 'allow_text_output': False},
        expected='required',
    ),
    dict(
        id='none_defaults_to_auto',
        tool_choice=None,
        params_kwargs={'allow_text_output': True},
        expected='auto',
    ),
    dict(
        id='none_with_text_output',
        tool_choice='none',
        params_kwargs={'function_tools': [make_tool('x')], 'allow_text_output': True},
        expected='none',
    ),
    dict(
        id='none_only_output_tools_no_direct_output',
        tool_choice='none',
        params_kwargs={'function_tools': [], 'output_tools': [make_tool('final_result')], 'allow_text_output': False},
        expected='required',
    ),
    dict(
        id='required_with_function_tools',
        tool_choice='required',
        params_kwargs={'function_tools': [make_tool('x')], 'allow_text_output': True},
        expected='required',
    ),
    dict(
        id='list_exact_match',
        tool_choice=['a', 'b'],
        params_kwargs={'function_tools': [make_tool('a'), make_tool('b')], 'allow_text_output': True},
        expected='required',
    ),
    dict(
        id='tool_or_output_empty_no_output_tools',
        tool_choice=ToolOrOutput(function_tools=[]),
        params_kwargs={'allow_text_output': True},
        expected='none',
    ),
    dict(
        id='tool_or_output_exact_match_no_direct_output',
        tool_choice=ToolOrOutput(function_tools=['a']),
        params_kwargs={
            'function_tools': [make_tool('a')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': False,
        },
        expected='required',
    ),
    dict(
        id='tool_or_output_exact_match_with_direct_output',
        tool_choice=ToolOrOutput(function_tools=['a']),
        params_kwargs={
            'function_tools': [make_tool('a')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        expected='auto',
    ),
]


@pytest.mark.parametrize('case', SIMPLE_CASES, ids=lambda c: c['id'])
def test_resolve_tool_choice_simple(case: dict[str, Any]):
    """Tests where resolve_tool_choice returns a simple string mode."""
    tool_choice = case['tool_choice']
    params_kwargs = case['params_kwargs']
    expected = case['expected']

    settings: ModelSettings | None = {'tool_choice': tool_choice} if tool_choice is not None else None
    params = ModelRequestParameters(**params_kwargs)
    assert resolve_tool_choice(settings, params) == expected


TUPLE_CASES = [
    dict(
        id='none_output_tools_direct_output',
        tool_choice='none',
        params_kwargs={
            'function_tools': [make_tool('func')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        expected_mode='auto',
        expected_tools={'final_result'},
    ),
    dict(
        id='none_output_tools_no_direct_output',
        tool_choice='none',
        params_kwargs={
            'function_tools': [make_tool('func')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': False,
        },
        expected_mode='required',
        expected_tools={'final_result'},
    ),
    dict(
        id='list_subset',
        tool_choice=['a', 'c'],
        params_kwargs={'function_tools': [make_tool('a'), make_tool('b'), make_tool('c')], 'allow_text_output': True},
        expected_mode='required',
        expected_tools={'a', 'c'},
    ),
    dict(
        id='list_partial_invalid',
        tool_choice=['a', 'invalid'],
        params_kwargs={'function_tools': [make_tool('a'), make_tool('b')], 'allow_text_output': True},
        expected_mode='required',
        expected_tools={'a', 'invalid'},
    ),
    dict(
        id='tool_or_output_empty_with_output_tools_direct_output',
        tool_choice=ToolOrOutput(function_tools=[]),
        params_kwargs={'output_tools': [make_tool('final_result')], 'allow_text_output': True},
        expected_mode='auto',
        expected_tools={'final_result'},
    ),
    dict(
        id='tool_or_output_empty_with_output_tools_no_direct_output',
        tool_choice=ToolOrOutput(function_tools=[]),
        params_kwargs={'output_tools': [make_tool('final_result')], 'allow_text_output': False},
        expected_mode='required',
        expected_tools={'final_result'},
    ),
    dict(
        id='tool_or_output_partial_invalid',
        tool_choice=ToolOrOutput(function_tools=['a', 'invalid']),
        params_kwargs={
            'function_tools': [make_tool('a'), make_tool('b')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        expected_mode='auto',
        expected_tools={'a', 'final_result', 'invalid'},
    ),
    dict(
        id='tool_or_output_subset_with_direct_output',
        tool_choice=ToolOrOutput(function_tools=['a']),
        params_kwargs={
            'function_tools': [make_tool('a'), make_tool('b')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        expected_mode='auto',
        expected_tools={'a', 'final_result'},
    ),
    dict(
        id='tool_or_output_subset_without_direct_output',
        tool_choice=ToolOrOutput(function_tools=['a']),
        params_kwargs={
            'function_tools': [make_tool('a'), make_tool('b')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': False,
        },
        expected_mode='required',
        expected_tools={'a', 'final_result'},
    ),
]


@pytest.mark.parametrize('case', TUPLE_CASES, ids=lambda c: c['id'])
def test_resolve_tool_choice_tuple(case: dict[str, Any]):
    """Tests where resolve_tool_choice returns a (mode, tools) tuple."""
    tool_choice = case['tool_choice']
    params_kwargs = case['params_kwargs']
    expected_mode = case['expected_mode']
    expected_tools = case['expected_tools']

    params = ModelRequestParameters(**params_kwargs)
    result = resolve_tool_choice({'tool_choice': tool_choice}, params)
    assert result[0] == expected_mode and set(result[1]) == expected_tools


RAISES_CASES = [
    dict(
        id='required_no_function_tools',
        tool_choice='required',
        params_kwargs={'allow_text_output': True},
        match='no function tools are defined',
    ),
    dict(
        id='list_all_invalid',
        tool_choice=['x', 'y'],
        params_kwargs={'function_tools': [make_tool('a'), make_tool('b')], 'allow_text_output': True},
        match=r'Invalid tool names in `tool_choice`:.*Available tools:',
    ),
    dict(
        id='list_invalid_no_function_tools',
        tool_choice=['x'],
        params_kwargs={'function_tools': [], 'allow_text_output': True},
        match=r'Invalid tool names.*Available tools: none',
    ),
    dict(
        id='tool_or_output_all_invalid',
        tool_choice=ToolOrOutput(function_tools=['x', 'y']),
        params_kwargs={
            'function_tools': [make_tool('a'), make_tool('b')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        match=r'Invalid tool names in `tool_choice`:.*Available function tools:',
    ),
]


@pytest.mark.parametrize('case', RAISES_CASES, ids=lambda c: c['id'])
def test_resolve_tool_choice_raises(case: dict[str, Any]):
    """Tests where resolve_tool_choice raises UserError."""
    tool_choice = case['tool_choice']
    params_kwargs = case['params_kwargs']
    match = case['match']

    params = ModelRequestParameters(**params_kwargs)
    with pytest.raises(UserError, match=match):
        resolve_tool_choice({'tool_choice': tool_choice}, params)


# =============================================================================
# Provider-specific tool_choice tests
# =============================================================================


@pytest.mark.parametrize('tool_choice', ['required', ['my_tool']], ids=['required', 'list'])
@pytest.mark.parametrize('provider_name', ['anthropic', 'bedrock'])
async def test_thinking_with_forced_tool_choice_raises(
    provider_name: str, tool_choice: Any, allow_model_requests: None
):
    """Providers don't support forcing tool use with thinking mode enabled."""
    if provider_name == 'anthropic':
        pytest.importorskip('anthropic')
        m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
        settings: Any = {
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024},
            'tool_choice': tool_choice,
        }
        match = 'Anthropic does not support .* with thinking mode'
    else:  # bedrock
        pytest.importorskip('boto3')
        mock_client = MagicMock()
        provider = BedrockProvider(bedrock_client=mock_client)
        profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
        m = BedrockConverseModel('test-model', provider=provider, profile=profile)
        settings = {
            'bedrock_additional_model_requests_fields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
            'tool_choice': tool_choice,
        }
        match = 'Bedrock does not support forcing specific tools with thinking mode'

    params = ModelRequestParameters(function_tools=[make_tool('my_tool')], allow_text_output=True)
    with pytest.raises(UserError, match=match):
        await m.request([ModelRequest.user_text_prompt('test')], settings, params)


@pytest.mark.parametrize('tool_choice', ['required', ['my_tool']], ids=['required', 'list'])
@pytest.mark.parametrize('provider_name', ['bedrock', 'openai'])
async def test_unsupported_profile_with_forced_tool_choice_raises(
    provider_name: str, tool_choice: Any, allow_model_requests: None
):
    """Models without tool_choice support raise UserError when forcing tool use."""
    mock_client = MagicMock()
    if provider_name == 'bedrock':
        pytest.importorskip('boto3')
        provider = BedrockProvider(bedrock_client=mock_client)
        profile = BedrockModelProfile(bedrock_supports_tool_choice=False)
        m = BedrockConverseModel('us.amazon.nova-lite-v1:0', provider=provider, profile=profile)
    else:  # openai
        pytest.importorskip('openai')
        provider = OpenAIProvider(openai_client=mock_client)
        profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
        m = OpenAIChatModel('gpt-4o-mini', provider=provider, profile=profile)

    params = ModelRequestParameters(function_tools=[make_tool('my_tool')], allow_text_output=True)
    with pytest.raises(UserError, match='tool_choice=.* is not supported by model'):
        await m.request([ModelRequest.user_text_prompt('test')], {'tool_choice': tool_choice}, params)


FORCING_CASES = [
    'required',
    ('required', {'tool_a'}),
    ('auto', {'tool_a'}),
    'auto',
    'none',
]


@pytest.mark.parametrize(
    'resolved_tool_choice',
    FORCING_CASES,
    ids=['required', 'tuple_required', 'tuple_auto', 'auto', 'none'],
)
@pytest.mark.parametrize('provider_name', ['anthropic', 'bedrock'])
def test_support_tool_forcing_implicit_resolution(provider_name: str, resolved_tool_choice: Any):
    """With thinking enabled but no explicit tool_choice, returns based on resolved value."""
    expected = resolved_tool_choice in ('auto', 'none')

    if provider_name == 'anthropic':
        pytest.importorskip('anthropic')
        settings: AnthropicModelSettings = {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024}}
        result = anthropic_support_tool_forcing(settings, resolved_tool_choice)
    else:  # bedrock
        pytest.importorskip('boto3')
        profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
        settings_bedrock: BedrockModelSettings = {
            'bedrock_additional_model_requests_fields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}}
        }
        result = bedrock_support_tool_forcing('test-model', profile, settings_bedrock, resolved_tool_choice)
    assert result is expected


# =============================================================================
# Provider-specific tests that don't fit the consolidated patterns
# =============================================================================


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
@pytest.mark.parametrize(
    'supports_json_schema,expected_output_mode',
    [
        pytest.param(True, 'native', id='native'),
        pytest.param(False, 'prompted', id='prompted'),
    ],
)
def test_bedrock_prepare_request_thinking_auto_output_mode(supports_json_schema: bool, expected_output_mode: str):
    """When thinking + output tools + auto mode, convert to native or prompted based on profile."""
    mock_client = MagicMock()
    provider = BedrockProvider(bedrock_client=mock_client)
    profile = BedrockModelProfile(supports_json_schema_output=supports_json_schema)
    m = BedrockConverseModel('test-model', provider=provider, profile=profile)

    settings: BedrockModelSettings = {
        'bedrock_additional_model_requests_fields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}}
    }
    params = ModelRequestParameters(
        output_tools=[make_tool('final_result')],
        output_mode='auto',
        allow_text_output=True,
    )

    _, result_params = m.prepare_request(settings, params)
    assert result_params.output_mode == expected_output_mode


@pytest.mark.skipif(not google_available(), reason='google not installed')
def test_google_auto_tuple_filters_tool_defs():
    """When resolve_tool_choice returns ('auto', [...]), Google filters tool_defs to only include allowed tools."""
    mock_client = MagicMock()
    provider = GoogleProvider(client=mock_client)
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    params = ModelRequestParameters(
        function_tools=[make_tool('func')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )

    tools, tool_config, _ = m._get_tool_config(params, {'tool_choice': 'none'})  # pyright: ignore[reportPrivateUsage]

    assert tools is not None
    assert len(tools) == 1
    assert tools[0]['function_declarations'][0]['name'] == 'final_result'  # pyright: ignore[reportTypedDictNotRequiredAccess,reportOptionalSubscript]
    assert tool_config is not None
    assert tool_config['function_calling_config']['mode'].name == 'AUTO'  # pyright: ignore[reportTypedDictNotRequiredAccess,reportOptionalMemberAccess,reportOptionalSubscript,reportUnknownMemberType]


@pytest.mark.skipif(not xai_available(), reason='xai not installed')
async def test_xai_fallback_single_tool_without_required_support(allow_model_requests: None):
    """Single tool with unsupported required falls back to auto."""
    mock_client = MagicMock()
    provider = XaiProvider(xai_client=mock_client)
    profile = GrokModelProfile(grok_supports_tool_choice_required=False)
    m = XaiModel('grok-3-fast', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=True)

    tool_defs, tool_choice = m._get_tool_choice({'tool_choice': ['tool_a']}, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == 'auto'
    assert 'tool_a' in tool_defs


@pytest.mark.skipif(not xai_available(), reason='xai not installed')
async def test_xai_fallback_multiple_tools_without_required_support(allow_model_requests: None):
    """Multiple tools with unsupported required falls back to auto with filtering."""
    mock_client = MagicMock()
    provider = XaiProvider(xai_client=mock_client)
    profile = GrokModelProfile(grok_supports_tool_choice_required=False)
    m = XaiModel('grok-3-fast', provider=provider, profile=profile)
    params = ModelRequestParameters(
        function_tools=[make_tool('tool_a'), make_tool('tool_b'), make_tool('tool_c')], allow_text_output=True
    )

    tool_defs, tool_choice = m._get_tool_choice({'tool_choice': ['tool_a', 'tool_c']}, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == 'auto'
    assert set(tool_defs.keys()) == {'tool_a', 'tool_c'}


@pytest.mark.skipif(not xai_available(), reason='xai not installed')
async def test_xai_required_with_no_text_output_and_supported(allow_model_requests: None):
    """Required mode used when text output disabled and profile supports it."""
    mock_client = MagicMock()
    provider = XaiProvider(xai_client=mock_client)
    profile = GrokModelProfile(grok_supports_tool_choice_required=True)
    m = XaiModel('grok-3-fast', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('tool_a')], allow_text_output=False)

    _, tool_choice = m._get_tool_choice({}, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == 'required'
