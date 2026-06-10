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
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.settings import ModelSettings, ToolChoice, ToolOrOutput
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


WARNS_CASES = [
    dict(
        id='list_partial_invalid',
        tool_choice=['a', 'typo'],
        params_kwargs={'function_tools': [make_tool('a'), make_tool('b')], 'allow_text_output': True},
        match=r"Some tools.*'typo'.*Available tools: \['a', 'b'\]",
        expected_mode='required',
        expected_tools={'a', 'typo'},
    ),
    dict(
        id='tool_or_output_partial_invalid',
        tool_choice=ToolOrOutput(function_tools=['a', 'typo']),
        params_kwargs={
            'function_tools': [make_tool('a'), make_tool('b')],
            'output_tools': [make_tool('final_result')],
            'allow_text_output': True,
        },
        match=r"Some tools.*'typo'.*Available function tools: \['a', 'b'\]",
        expected_mode='auto',
        expected_tools={'a', 'final_result', 'typo'},
    ),
]


@pytest.mark.parametrize('case', WARNS_CASES, ids=lambda c: c['id'])
def test_resolve_tool_choice_partial_invalid_warns(case: dict[str, Any]):
    """Partial-invalid tool names emit a warning (not an error) to support dynamic tool availability."""
    params = ModelRequestParameters(**case['params_kwargs'])
    with pytest.warns(UserWarning, match=case['match']):
        result = resolve_tool_choice({'tool_choice': case['tool_choice']}, params)
    assert isinstance(result, tuple)
    assert result[0] == case['expected_mode']
    assert set(result[1]) == case['expected_tools']


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
        result = anthropic_support_tool_forcing(settings, ModelRequestParameters(), resolved_tool_choice)
    else:  # bedrock
        pytest.importorskip('boto3')
        profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
        settings_bedrock: BedrockModelSettings = {
            'bedrock_additional_model_requests_fields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}}
        }
        result = bedrock_support_tool_forcing(
            'test-model', profile, settings_bedrock, ModelRequestParameters(), resolved_tool_choice
        )
    assert result is expected


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'settings,expected',
    [
        pytest.param(
            {'anthropic_thinking': {'type': 'disabled'}},
            True,
            id='disabled_thinking_allows_forcing',
        ),
        pytest.param(
            {'thinking': True},
            False,
            id='unified_thinking_blocks_forcing',
        ),
        pytest.param(
            {'thinking': 'high'},
            False,
            id='unified_thinking_effort_blocks_forcing',
        ),
        pytest.param(
            {'thinking': False},
            True,
            id='unified_thinking_false_allows_forcing',
        ),
        pytest.param(
            {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024}, 'thinking': False},
            False,
            id='provider_specific_takes_precedence',
        ),
    ],
)
def test_support_tool_forcing_thinking_detection(settings: Any, expected: bool):
    """Thinking detection checks anthropic_thinking, unified thinking field, and `params.thinking`."""
    result = anthropic_support_tool_forcing(settings, ModelRequestParameters(), 'required')
    assert result is expected


@pytest.mark.parametrize(
    'provider_name',
    [
        pytest.param(
            'anthropic', marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
        ),
        pytest.param('bedrock', marks=pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')),
    ],
)
def test_support_tool_forcing_reads_params_thinking(provider_name: str):
    """Regression: `Model.prepare_request` strips unified `thinking` from `model_settings` into
    `model_request_parameters.thinking` before tool-choice helpers run, so the helpers must
    inspect `params.thinking` — not just `model_settings`.
    """
    params = ModelRequestParameters(thinking=True)
    if provider_name == 'anthropic':
        # Empty settings simulates post-strip state
        result = anthropic_support_tool_forcing({}, params, 'required')
    else:
        profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
        result = bedrock_support_tool_forcing('test-model', profile, {}, params, 'required')
    assert result is False


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
def test_bedrock_single_tool_fallback_filters_when_unsupported():
    """When a Bedrock model can't force a single tool (here: thinking enabled blocks `toolChoice.tool`),
    the single-output-tool path must trim `tool_defs` to the forced name and emit `toolChoice={'auto': {}}`.
    The cache-preserving full-array shape only applies when `_support_tool_forcing` returns True.

    The thinking-enabled fallback is reached because `tool_choice='none'` + one output tool
    + no direct output resolves to `('required', {single_name})`, and `_support_tool_forcing`
    returns False (without raising) because explicit `tool_choice` is `'none'`, not `'required'`/list.
    """
    mock_client = MagicMock()
    provider = BedrockProvider(bedrock_client=mock_client)
    profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
    model = BedrockConverseModel('us.amazon.nova-lite-v1:0', provider=provider, profile=profile)
    params = ModelRequestParameters(
        function_tools=[make_tool('helper_tool')],
        output_tools=[make_tool('final_result')],
        allow_text_output=False,
        thinking=True,
    )

    tool_config = model._map_tool_config(params, BedrockModelSettings(tool_choice='none'))  # pyright: ignore[reportPrivateUsage]

    assert tool_config is not None
    assert tool_config.get('toolChoice') == {'auto': {}}
    assert [tool['toolSpec']['name'] for tool in tool_config['tools'] if 'toolSpec' in tool] == ['final_result']


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
@pytest.mark.parametrize(
    'tool_choice_value,function_tool_names,output_tool_names,expected_forced_name,expected_tool_names',
    [
        pytest.param(
            ToolOrOutput(function_tools=[]),
            ['helper_tool'],
            ['final_result'],
            'final_result',
            ['helper_tool', 'final_result'],
            id='tool_or_output_empty_with_single_output_tool',
        ),
        pytest.param(
            ToolOrOutput(function_tools=['tool_a']),
            ['tool_a', 'tool_b'],
            [],
            'tool_a',
            ['tool_a', 'tool_b'],
            id='tool_or_output_single_function_tool_subset',
        ),
    ],
)
def test_bedrock_tool_or_output_single_resolved_preserves_cache(
    tool_choice_value: Any,
    function_tool_names: list[str],
    output_tool_names: list[str],
    expected_forced_name: str,
    expected_tool_names: list[str],
):
    """`ToolOrOutput` paths that resolve to `('required', {single_name})` on a supporting model
    must preserve the full tools array and force via `toolChoice.tool` (no client-side filter).

    Covers the two `resolve_tool_choice` branches that aren't pinned by the explicit-list
    (`list_single`) or `tool_choice='none'` (`none_with_output`) matrix tests.
    """
    mock_client = MagicMock()
    provider = BedrockProvider(bedrock_client=mock_client)
    profile = BedrockModelProfile(bedrock_supports_tool_choice=True)
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider, profile=profile)
    params = ModelRequestParameters(
        function_tools=[make_tool(n) for n in function_tool_names],
        output_tools=[make_tool(n) for n in output_tool_names],
        allow_text_output=False,
    )

    tool_config = model._map_tool_config(params, BedrockModelSettings(tool_choice=tool_choice_value))  # pyright: ignore[reportPrivateUsage]

    assert tool_config is not None
    assert tool_config.get('toolChoice') == {'tool': {'name': expected_forced_name}}
    assert [tool['toolSpec']['name'] for tool in tool_config['tools'] if 'toolSpec' in tool] == expected_tool_names


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


NATIVE_TOOL_CONFIG_CASES = [
    dict(
        id='native-only-pre-gemini-3-omits-config',
        model='gemini-2.5-pro',
        request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        expected_tool_config=None,
    ),
    dict(
        id='native-only-gemini-3-keeps-only-server-side-flag',
        model='gemini-3-flash-preview',
        request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        expected_tool_config={'include_server_side_tool_invocations': True},
    ),
    dict(
        id='function-tool-keeps-config',
        model='gemini-2.5-pro',
        request_parameters=ModelRequestParameters(function_tools=[make_tool('get_weather')]),
        expected_tool_config={'function_calling_config': {'mode': 'AUTO'}},
    ),
]


@pytest.mark.skipif(not google_available(), reason='google not installed')
@pytest.mark.parametrize('case', NATIVE_TOOL_CONFIG_CASES, ids=lambda c: c['id'])
def test_google_native_tool_only_omits_function_calling_config(case: dict[str, Any]):
    """A `function_calling_config` only governs function tools, so it must be omitted when there are
    only native tools: since #3611 Gemini 400s on one with no `function_declarations`.

    Asserted on the request shape directly rather than via VCR: a cassette replay can't catch a
    malformed request, since it replays a recorded response without re-validating against the API.
    """
    m = GoogleModel(case['model'], provider=GoogleProvider(client=MagicMock()))

    _, tool_config, _ = m._get_tool_config(case['request_parameters'], {})  # pyright: ignore[reportPrivateUsage]

    assert tool_config == case['expected_tool_config']


@pytest.mark.skipif(not xai_available(), reason='xai not installed')
async def test_xai_fallback_single_tool_without_required_support(allow_model_requests: None):
    """Single tool with unsupported required falls back to auto and filters tool_defs to preserve user intent."""
    mock_client = MagicMock()
    provider = XaiProvider(xai_client=mock_client)
    profile = GrokModelProfile(grok_supports_tool_choice_required=False)
    m = XaiModel('grok-3-fast', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=True)

    tool_defs, tool_choice = m._get_tool_choice({'tool_choice': ['tool_a']}, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == 'auto'
    assert set(tool_defs.keys()) == {'tool_a'}


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


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_fallback_single_tool_with_thinking_filters_tool_defs(allow_model_requests: None):
    """`ToolOrOutput` single function tool with thinking enabled falls back to auto and filters tool_defs.

    Explicit `tool_choice=['tool_a']` with thinking would raise UserError before reaching this branch;
    `ToolOrOutput` is the path where the resolved `('required', {single_tool})` actually reaches the fallback.
    """
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    settings: AnthropicModelSettings = {
        'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024},
        'tool_choice': ToolOrOutput(function_tools=['tool_a']),
    }
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=False)

    tools, tool_choice = m._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == {'type': 'auto'}
    tool_names = {t['name'] for t in tools if isinstance(t, dict) and 'name' in t}
    assert tool_names == {'tool_a'}


# Models that reject a forced `tool_choice` outright, even without thinking (unlike other Anthropic models).
NO_FORCING_ANTHROPIC_MODELS = ['claude-fable-5', 'claude-mythos-5', 'claude-mythos-preview']


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('model_name', NO_FORCING_ANTHROPIC_MODELS)
async def test_anthropic_no_forcing_model_falls_back_to_auto(allow_model_requests: None, model_name: str):
    """Models that reject forcing outright fall back to auto for a resolved `('required', {single_tool})`,
    filtering tool_defs to the requested set."""
    m = AnthropicModel(model_name, provider=AnthropicProvider(api_key='test-key'))
    settings: AnthropicModelSettings = {'tool_choice': ToolOrOutput(function_tools=['tool_a'])}
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=False)

    tools, tool_choice = m._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]
    assert tool_choice == {'type': 'auto'}
    tool_names = {t['name'] for t in tools if isinstance(t, dict) and 'name' in t}
    assert tool_names == {'tool_a'}


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('model_name', NO_FORCING_ANTHROPIC_MODELS)
@pytest.mark.parametrize('tool_choice', ['required', ['tool_a']])
async def test_anthropic_no_forcing_model_explicit_forcing_raises(
    allow_model_requests: None, model_name: str, tool_choice: ToolChoice
):
    """An explicit forcing `tool_choice` (`'required'` or a list of tools) raises on models that reject
    forcing outright, since we can't silently downgrade a user's explicit request."""
    m = AnthropicModel(model_name, provider=AnthropicProvider(api_key='test-key'))
    params = ModelRequestParameters(function_tools=[make_tool('tool_a')], allow_text_output=True)
    settings: AnthropicModelSettings = {'tool_choice': tool_choice}
    with pytest.raises(UserError, match='Anthropic does not support .* for this model'):
        m._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_openai_chat_fallback_single_tool_filters_tool_defs(allow_model_requests: None):
    """`ToolOrOutput` single function tool on a no-forcing model falls back to auto and filters tool_defs."""
    mock_client = MagicMock()
    provider = OpenAIProvider(openai_client=mock_client)
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    m = OpenAIChatModel('gpt-4o-mini', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=False)

    tools, tool_choice = m._get_tool_choice(  # pyright: ignore[reportPrivateUsage]
        {'tool_choice': ToolOrOutput(function_tools=['tool_a'])}, params
    )
    assert tool_choice == 'auto'
    assert {t['function']['name'] for t in tools} == {'tool_a'}


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_openai_responses_fallback_single_tool_uses_allowed_tools(allow_model_requests: None):
    """`ToolOrOutput` single function tool on a no-forcing Responses model uses `allowed_tools` to preserve cache."""
    from pydantic_ai.models.openai import OpenAIResponsesModel

    mock_client = MagicMock()
    provider = OpenAIProvider(openai_client=mock_client)
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)
    m = OpenAIResponsesModel('gpt-4o-mini', provider=provider, profile=profile)
    params = ModelRequestParameters(function_tools=[make_tool('tool_a'), make_tool('tool_b')], allow_text_output=False)

    tools, tool_choice = m._get_responses_tool_choice(  # pyright: ignore[reportPrivateUsage]
        {'tool_choice': ToolOrOutput(function_tools=['tool_a'])}, params
    )
    assert isinstance(tool_choice, dict)
    assert tool_choice['type'] == 'allowed_tools'
    assert tool_choice['mode'] == 'auto'
    assert tool_choice['tools'] == [{'type': 'function', 'name': 'tool_a'}]
    assert {t['name'] for t in tools} == {'tool_a', 'tool_b'}


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
