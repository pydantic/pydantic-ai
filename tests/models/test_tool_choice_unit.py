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
from pydantic_ai.settings import ToolOrOutput
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel
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


def test_auto_with_text_output():
    params = ModelRequestParameters(allow_text_output=True)
    assert resolve_tool_choice({'tool_choice': 'auto'}, params) == 'auto'


def test_auto_without_text_output():
    params = ModelRequestParameters(function_tools=[make_tool('x')], allow_text_output=False)
    assert resolve_tool_choice({'tool_choice': 'auto'}, params) == 'required'


def test_none_defaults_to_auto():
    params = ModelRequestParameters(allow_text_output=True)
    assert resolve_tool_choice(None, params) == 'auto'


def test_none_with_text_output_allowed():
    params = ModelRequestParameters(function_tools=[make_tool('x')], allow_text_output=True)
    assert resolve_tool_choice({'tool_choice': 'none'}, params) == 'none'


def test_none_with_output_tools_and_direct_output():
    """Disabling function tools with output tools and direct output returns auto mode."""
    params = ModelRequestParameters(
        function_tools=[make_tool('func')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    result = resolve_tool_choice({'tool_choice': 'none'}, params)
    assert result[0] == 'auto' and set(result[1]) == {'final_result'}


def test_none_with_output_tools_no_direct_output_with_function_tools():
    """Disabling function tools with output tools but no direct output forces required mode."""
    params = ModelRequestParameters(
        function_tools=[make_tool('func')],
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
    )
    result = resolve_tool_choice({'tool_choice': 'none'}, params)
    assert result[0] == 'required' and set(result[1]) == {'final_result'}


def test_none_with_only_output_tools_no_direct_output():
    """Only output tools exist with no direct output allowed forces required mode."""
    params = ModelRequestParameters(
        function_tools=[],
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
    )
    result = resolve_tool_choice({'tool_choice': 'none'}, params)
    assert result == 'required'


def test_required_with_function_tools():
    params = ModelRequestParameters(function_tools=[make_tool('x')], allow_text_output=True)
    assert resolve_tool_choice({'tool_choice': 'required'}, params) == 'required'


def test_required_without_function_tools_raises():
    """Requiring tool use without function tools raises UserError."""
    params = ModelRequestParameters(allow_text_output=True)
    with pytest.raises(UserError, match='no function tools are defined'):
        resolve_tool_choice({'tool_choice': 'required'}, params)


def test_list_all_invalid_raises():
    """Specifying only invalid tool names raises UserError."""
    params = ModelRequestParameters(function_tools=[make_tool('a'), make_tool('b')], allow_text_output=True)
    with pytest.raises(UserError, match='Invalid tool names'):
        resolve_tool_choice({'tool_choice': ['x', 'y']}, params)


def test_list_exact_match_returns_required():
    """Specifying all available tools returns simple required mode."""
    params = ModelRequestParameters(function_tools=[make_tool('a'), make_tool('b')], allow_text_output=True)
    result = resolve_tool_choice({'tool_choice': ['a', 'b']}, params)
    assert result == 'required'


def test_list_subset_returns_tuple():
    params = ModelRequestParameters(
        function_tools=[make_tool('a'), make_tool('b'), make_tool('c')], allow_text_output=True
    )
    result = resolve_tool_choice({'tool_choice': ['a', 'c']}, params)
    assert result[0] == 'required' and set(result[1]) == {'a', 'c'}


def test_list_partial_invalid_filters_silently():
    """Partial invalid tools are filtered, not errored."""
    params = ModelRequestParameters(function_tools=[make_tool('a'), make_tool('b')], allow_text_output=True)
    result = resolve_tool_choice({'tool_choice': ['a', 'invalid']}, params)
    assert result[0] == 'required' and set(result[1]) == {'a', 'invalid'}


def test_tool_or_output_empty_function_tools_with_output_tools_direct_output():
    """Empty function tools with output tools and direct output returns auto."""
    params = ModelRequestParameters(
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=[])}, params)
    assert result == 'auto'


def test_tool_or_output_empty_function_tools_with_output_tools_no_direct_output():
    """Empty function tools with output tools but no direct output returns required."""
    params = ModelRequestParameters(
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=[])}, params)
    assert result == 'required'


def test_tool_or_output_empty_function_tools_no_output_tools():
    """Empty function tools with no output tools returns none."""
    params = ModelRequestParameters(allow_text_output=True)
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=[])}, params)
    assert result == 'none'


def test_tool_or_output_all_invalid_function_tools_raises():
    """All invalid function tools in ToolOrOutput raises UserError."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a'), make_tool('b')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    with pytest.raises(UserError, match='Invalid tool names'):
        resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['x', 'y'])}, params)


def test_tool_or_output_partial_invalid_function_tools_passes():
    """Mixing valid and invalid function tools passes silently (at least one valid)."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a'), make_tool('b')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['a', 'invalid'])}, params)
    assert result[0] == 'auto' and set(result[1]) == {'a', 'final_result', 'invalid'}


def test_tool_or_output_exact_match_no_direct_output_returns_required():
    """ToolOrOutput matching all available tools without direct output returns required."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['a'])}, params)
    assert result == 'required'


def test_tool_or_output_exact_match_with_direct_output_returns_auto():
    """ToolOrOutput matching all available tools with direct output returns auto."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['a'])}, params)
    assert result == 'auto'


def test_tool_or_output_subset_with_direct_output_returns_auto_tuple():
    """ToolOrOutput subset with direct output allowed returns auto tuple."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a'), make_tool('b')],
        output_tools=[make_tool('final_result')],
        allow_text_output=True,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['a'])}, params)
    assert result[0] == 'auto' and set(result[1]) == {'a', 'final_result'}


def test_tool_or_output_subset_without_direct_output_returns_required_tuple():
    """ToolOrOutput subset without direct output returns required tuple."""
    params = ModelRequestParameters(
        function_tools=[make_tool('a'), make_tool('b')],
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
    )
    result = resolve_tool_choice({'tool_choice': ToolOrOutput(function_tools=['a'])}, params)
    assert result[0] == 'required' and set(result[1]) == {'a', 'final_result'}


# =============================================================================
# Anthropic tool_choice tests (direct model.request() - blocked by Agent)
# =============================================================================


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'tool_choice',
    [
        pytest.param('required', id='required'),
        pytest.param(['my_tool'], id='list'),
    ],
)
async def test_anthropic_thinking_with_forced_tool_choice_raises(tool_choice: Any, allow_model_requests: None):
    """Anthropic does not support forcing tool use with thinking mode enabled."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    settings: AnthropicModelSettings = {
        'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024},
        'tool_choice': tool_choice,
    }
    params = ModelRequestParameters(function_tools=[make_tool('my_tool')], allow_text_output=True)

    with pytest.raises(UserError, match='Anthropic does not support .* with thinking mode'):
        await m.request([ModelRequest.user_text_prompt('test')], settings, params)


# =============================================================================
# Bedrock tool_choice tests (direct model.request() - blocked by Agent)
# =============================================================================


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
@pytest.mark.parametrize(
    'tool_choice',
    [
        pytest.param('required', id='required'),
        pytest.param(['my_tool'], id='list'),
    ],
)
async def test_bedrock_unsupported_profile_with_forced_tool_choice_raises(tool_choice: Any, allow_model_requests: None):
    """Models without tool_choice support raise UserError when forcing tool use."""
    mock_client = MagicMock()
    provider = BedrockProvider(bedrock_client=mock_client)
    profile = BedrockModelProfile(bedrock_supports_tool_choice=False)
    m = BedrockConverseModel('us.amazon.nova-lite-v1:0', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('my_tool')], allow_text_output=True)

    with pytest.raises(UserError, match='tool_choice=.* is not supported by model'):
        await m.request([ModelRequest.user_text_prompt('test')], {'tool_choice': tool_choice}, params)


# =============================================================================
# OpenAI tool_choice tests (direct model.request() - blocked by Agent)
# =============================================================================


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize(
    'tool_choice',
    [
        pytest.param('required', id='required'),
        pytest.param(['my_tool'], id='list'),
    ],
)
async def test_openai_unsupported_profile_with_forced_tool_choice_raises(tool_choice: Any, allow_model_requests: None):
    """Models without tool_choice support raise UserError when forcing tool use."""
    mock_client = MagicMock()
    provider = OpenAIProvider(openai_client=mock_client)
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    m = OpenAIChatModel('gpt-4o-mini', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('my_tool')], allow_text_output=True)

    with pytest.raises(UserError, match='tool_choice=.* is not supported by model'):
        await m.request([ModelRequest.user_text_prompt('test')], {'tool_choice': tool_choice}, params)


# =============================================================================
# Google tool_choice tests
# =============================================================================


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


# =============================================================================
# xAI tool_choice tests (direct model.request() - tests fallback paths)
# =============================================================================


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
