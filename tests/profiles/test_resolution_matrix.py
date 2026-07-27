"""Regression matrix for resolved `ModelProfile` flags across providers, gateways, and routing.

Captures the resolved `Provider.model_profile(model_name)` output for every interesting
(provider, model_name) combination — every gateway (OpenRouter, Azure, Vertex), every
inference provider (Cerebras, Groq, Together, etc.), and every cross-provider routing case
(Bedrock-Anthropic, OpenRouter-Google, etc.).

The snapshots are the comparison artifact: any change to a provider profile, an upstream
profile, or `merge_profile()` semantics that affects a resolved flag must show up as a
diff here. The PR author then has to confront the diff and decide whether each delta is
intentional. Without this matrix, a one-line tweak to an upstream profile can silently
change behavior on a downstream routing case (e.g. an OpenAI profile change quietly
flipping a flag for the Azure-OpenAI route), and nobody on the PR is thinking about that.

Provider classes that do `merge_profile(fallback, upstream, override)` (3-layer) are the
densest here, because they have the most layers where an ordering change can silently
invert a flag.

The `_normalize` helper strips canonical default values (everything in `DEFAULT_PROFILE`
plus the subclass defaults), so each snapshot shows only the deltas the provider/gateway
is opinionating about. Snapshots stay scannable — a real diff stands out.

To regenerate after an intentional change: `pytest tests/profiles/test_resolution_lockin.py --inline-snapshot=fix`.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.native_tools import (
    SUPPORTED_NATIVE_TOOLS,
    AdvisorTool,
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    MemoryTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as anthropic_imports:
    from pydantic_ai.providers.anthropic import AnthropicJsonSchemaTransformer, AnthropicProvider

with try_import() as bedrock_imports:
    from pydantic_ai.providers.bedrock import BedrockJsonSchemaTransformer, BedrockProvider

with try_import() as cohere_imports:
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as google_imports:
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_imports:
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as huggingface_imports:
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as mistral_imports:
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as xai_imports:
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as openrouter_google_imports:
    # OpenRouter installs its own Google transformer; importable so inline_snapshot can name it.
    from pydantic_ai.providers.openrouter import (
        _OpenRouterGoogleJsonSchemaTransformer,  # pyright: ignore[reportPrivateUsage]
    )

# Canonical defaults — matches `DEFAULT_PROFILE` on both dataclass v1 and TypedDict v2.
# Defined locally so the test is stable across the migration: anything matching these
# values gets stripped from the snapshot, leaving only the non-default fields.
_CANONICAL_DEFAULTS: dict[str, Any] = {
    # Top-level `ModelProfile` defaults
    'supports_tools': True,
    'supports_tool_return_schema': False,
    'supports_json_schema_output': False,
    'supports_json_object_output': False,
    'supports_image_output': False,
    'default_structured_output_mode': 'tool',
    'prompted_output_template': dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    ),
    'native_output_requires_schema_in_instructions': False,
    'json_schema_transformer': None,
    'supports_thinking': False,
    'thinking_always_enabled': False,
    'thinking_tags': ('<think>', '</think>'),
    'ignore_streamed_leading_whitespace': False,
    'supported_native_tools': SUPPORTED_NATIVE_TOOLS,
    # OpenAIModelProfile subclass defaults
    'openai_chat_thinking_field': None,
    'openai_chat_send_back_thinking_parts': 'auto',
    'openai_supports_strict_tool_definition': True,
    'openai_unsupported_model_settings': (),
    'openai_supports_tool_choice_required': True,
    'openai_system_prompt_role': None,
    'openai_chat_supports_multiple_system_messages': True,
    'openai_chat_supports_web_search': False,
    'openai_chat_audio_input_encoding': 'base64',
    'openai_chat_supports_file_urls': False,
    'openai_supports_encrypted_reasoning_content': False,
    'openai_supports_reasoning': False,
    'openai_reasoning_enabled_by_default': False,
    'openai_supports_reasoning_effort_none': False,
    'openai_responses_supports_reasoning_mode': False,
    'openai_responses_requires_function_call_status_none': False,
    'openai_supports_phase': False,
    'openai_supports_prompt_cache_breakpoints': False,
    'openai_chat_supports_document_input': True,
    # AnthropicModelProfile subclass defaults
    'anthropic_supports_fast_speed': False,
    'anthropic_supports_adaptive_thinking': False,
    'anthropic_supports_effort': False,
    'anthropic_supports_xhigh_effort': False,
    'anthropic_supports_dynamic_filtering': False,
    'anthropic_disallows_budget_thinking': False,
    'anthropic_disallows_sampling_settings': False,
    'anthropic_default_code_execution_tool_version': '20250825',
    'anthropic_supported_code_execution_tool_versions': ('20250825',),
    'anthropic_supports_task_budgets': False,
    # GoogleModelProfile subclass defaults
    'google_supports_tool_combination': False,
    'google_supports_server_side_tool_invocations': False,
    'google_supported_mime_types_in_tool_returns': (),
    'google_supports_thinking_level': False,
    # GrokModelProfile subclass defaults
    'grok_supports_builtin_tools': False,
    'grok_supports_tool_choice_required': True,
    'grok_reasoning_efforts': frozenset(),
    # GroqModelProfile subclass defaults
    'groq_always_has_web_search_builtin_tool': False,
    # BedrockModelProfile subclass defaults
    'bedrock_supports_tool_choice': False,
    'bedrock_tool_result_format': 'text',
    'bedrock_send_back_thinking_parts': False,
    'bedrock_supports_prompt_caching': False,
    'bedrock_supports_tool_caching': False,
    'bedrock_supported_media_kinds_in_tool_returns': frozenset({'image'}),
    'bedrock_thinking_variant': None,
}


def _normalize(profile: Any) -> dict[str, Any] | None:
    """Reduce a `ModelProfile` `TypedDict` to a dict of non-default fields.

    Strips keys whose value matches `_CANONICAL_DEFAULTS`, so each snapshot
    shows only what the provider/gateway actually contributes.
    """
    if profile is None:
        return None
    return {k: v for k, v in profile.items() if k not in _CANONICAL_DEFAULTS or v != _CANONICAL_DEFAULTS[k]}


# =============================================================================
# Direct labs — single-layer profile functions (no gateway merging)
# =============================================================================


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
def test_anthropic_claude_sonnet_4_6():
    profile = AnthropicProvider.model_profile('claude-sonnet-4-6')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'json_schema_transformer': AnthropicJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'supported_native_tools': frozenset(
                {AdvisorTool, CodeExecutionTool, MCPServerTool, MemoryTool, ToolSearchTool, WebFetchTool, WebSearchTool}
            ),
            'anthropic_supports_adaptive_thinking': True,
            'anthropic_supports_dynamic_filtering': True,
            'anthropic_supports_effort': True,
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supports_forced_tool_choice': True,
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
        }
    )


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
def test_anthropic_claude_opus_4_7():
    profile = AnthropicProvider.model_profile('claude-opus-4-7')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'json_schema_transformer': AnthropicJsonSchemaTransformer,
            'supports_thinking': True,
            'anthropic_supports_fast_speed': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'supported_native_tools': frozenset(
                {AdvisorTool, CodeExecutionTool, MCPServerTool, MemoryTool, ToolSearchTool, WebFetchTool, WebSearchTool}
            ),
            'anthropic_supports_adaptive_thinking': True,
            'anthropic_supports_dynamic_filtering': True,
            'anthropic_supports_effort': True,
            'anthropic_supports_xhigh_effort': True,
            'anthropic_disallows_budget_thinking': True,
            'anthropic_disallows_sampling_settings': True,
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
            'anthropic_supports_forced_tool_choice': True,
            'anthropic_supports_task_budgets': True,
        }
    )


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
def test_anthropic_claude_haiku_4_5():
    profile = AnthropicProvider.model_profile('claude-haiku-4-5')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'json_schema_transformer': AnthropicJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_supports_forced_tool_choice': True,
            'supported_native_tools': frozenset(
                {AdvisorTool, CodeExecutionTool, MCPServerTool, MemoryTool, ToolSearchTool, WebFetchTool, WebSearchTool}
            ),
        }
    )


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
def test_anthropic_claude_3_5_sonnet_legacy():
    """Older model — no structured output, no tool search."""
    profile = AnthropicProvider.model_profile('claude-3-5-sonnet-20240620')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': AnthropicJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_supports_forced_tool_choice': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, MCPServerTool, MemoryTool, WebFetchTool, WebSearchTool}
            ),
        }
    )


def test_openai_gpt_5_4():
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile('gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, ToolSearchTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
        }
    )


def test_openai_gpt_5_6():
    """Not a VCR test: this pins the resolved GPT-5.6 profile against drift.

    GPT-5.6 reasons on by default at 'medium' (`openai_reasoning_enabled_by_default`) yet can be
    turned off via `effort='none'` (`openai_supports_reasoning_effort_none`), so it is NOT
    `thinking_always_enabled` (that flag is derived to False). `phase` is on (GPT-5.6 responses
    label messages with it) and native `tool_search` is on (verified live). Reasoning behavior
    verified against the Responses API.
    """
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile('gpt-5.6-sol')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool, ToolSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_reasoning_enabled_by_default': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_responses_supports_reasoning_mode': True,
            'openai_supports_phase': True,
            'openai_supports_prompt_cache_breakpoints': True,
        }
    )


@pytest.mark.parametrize('model_name', ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'])
def test_openai_gpt_5_6_reasoning_mode(model_name: str):
    """Not a VCR test: this validates local provider-profile capability resolution."""
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_responses_supports_reasoning_mode') is True


@pytest.mark.parametrize(
    'model_name',
    ['openai/gpt-5.6-sol', 'openai/gpt-5.6-terra', 'openai/gpt-5.6-luna'],
)
def test_openrouter_openai_gpt_5_6_reasoning_mode(model_name: str):
    """Not a VCR test: this validates local provider-profile capability resolution."""
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_responses_supports_reasoning_mode') is True


@pytest.mark.parametrize('model_name', ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'])
def test_azure_gpt_5_6_reasoning_mode(model_name: str):
    """Not a VCR test: this validates local provider-profile capability resolution."""
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_responses_supports_reasoning_mode') is True


def test_openai_gpt_4o():
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile('gpt-4o')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool}
            ),
        }
    )


def test_openai_o3_mini():
    from pydantic_ai.providers.openai import OpenAIProvider

    profile = OpenAIProvider.model_profile('o3-mini')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_reasoning_enabled_by_default': True,
            'openai_supports_reasoning': True,
        }
    )


@pytest.mark.skipif(not google_imports(), reason='google not installed')
def test_google_gemini_3_pro():
    profile = GoogleProvider.model_profile('gemini-3.0-pro')
    assert _normalize(profile) == snapshot(
        {
            'supports_tool_return_schema': True,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': GoogleJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'google_supports_tool_combination': True,
            'google_supports_server_side_tool_invocations': True,
            'google_supported_mime_types_in_tool_returns': (
                'image/png',
                'image/jpeg',
                'image/webp',
                'application/pdf',
                'text/plain',
            ),
            'google_supports_thinking_level': True,
        }
    )


@pytest.mark.skipif(not google_imports(), reason='google not installed')
def test_google_gemini_2_5_flash():
    profile = GoogleProvider.model_profile('gemini-2.5-flash')
    assert _normalize(profile) == snapshot(
        {
            'supports_tool_return_schema': True,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': GoogleJsonSchemaTransformer,
            'supports_thinking': True,
        }
    )


@pytest.mark.skipif(not xai_imports(), reason='xai not installed')
def test_xai_grok_4():
    profile = XaiProvider.model_profile('grok-4')
    assert _normalize(profile) == snapshot(
        {'supports_json_schema_output': True, 'supports_json_object_output': True, 'grok_supports_builtin_tools': True}
    )


@pytest.mark.skipif(not xai_imports(), reason='xai not installed')
def test_xai_grok_3_mini():
    profile = XaiProvider.model_profile('grok-3-mini')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'grok_reasoning_efforts': frozenset({'low', 'high'}),
            'supported_native_tools': frozenset(),
        }
    )


@pytest.mark.skipif(not mistral_imports(), reason='mistral not installed')
def test_mistral_mistral_large():
    profile = MistralProvider.model_profile('mistral-large-latest')
    assert _normalize(profile) == snapshot({'supports_inline_system_prompts': True})


@pytest.mark.skipif(not mistral_imports(), reason='mistral not installed')
def test_mistral_small_latest():
    """Small 4 / Medium 3.5 advertise adjustable (opt-in) reasoning, unlike always-on magistral."""
    profile = MistralProvider.model_profile('mistral-small-latest')
    assert _normalize(profile) == snapshot({'supports_thinking': True, 'supports_inline_system_prompts': True})


@pytest.mark.skipif(not cohere_imports(), reason='cohere not installed')
def test_cohere_command_r_plus():
    profile = CohereProvider.model_profile('command-r-plus')
    assert _normalize(profile) == snapshot({'supports_inline_system_prompts': True})


def test_deepseek_provider_deepseek_chat():
    """DeepSeek's own provider (OpenAI-compat) — three-layer merge."""
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    profile = DeepSeekProvider.model_profile('deepseek-chat')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_thinking_field': 'reasoning_content',
            'openai_chat_send_back_thinking_parts': 'field',
        }
    )


def test_deepseek_provider_deepseek_reasoner():
    """`deepseek-reasoner` overrides `openai_supports_tool_choice_required=False`."""
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    profile = DeepSeekProvider.model_profile('deepseek-reasoner')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'ignore_streamed_leading_whitespace': True,
            'openai_chat_thinking_field': 'reasoning_content',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_supports_tool_choice_required': False,
        }
    )


# =============================================================================
# Bedrock — cross-provider routing (Anthropic, Mistral, Amazon, Cohere, Meta,
# DeepSeek, OpenAI, Qwen). Highest-risk migration surface.
# =============================================================================


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_anthropic_claude_sonnet_4_5():
    """Anthropic via Bedrock: upstream Anthropic profile + Bedrock overrides (notably `supports_json_schema_output=False`)."""
    profile = BedrockProvider.model_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
            'supported_native_tools': frozenset(),
            'bedrock_tool_result_colocatable_content': frozenset({'image', 'text'}),
            'bedrock_supports_leading_assistant_message': True,
            'bedrock_supports_tool_choice': True,
            'bedrock_supports_adaptive_thinking': False,
            'bedrock_supports_effort': False,
            'bedrock_top_k_variant': 'anthropic',
            'bedrock_send_back_thinking_parts': True,
            'supports_json_schema_output': True,
            'bedrock_supports_prompt_caching': True,
            'bedrock_supports_tool_caching': True,
            'bedrock_supported_media_kinds_in_tool_returns': frozenset({'document', 'image'}),
            'anthropic_supports_forced_tool_choice': True,
            'bedrock_thinking_variant': 'anthropic',
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'bedrock_supports_strict_tool_definition': True,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_anthropic_with_geo_prefix():
    profile = BedrockProvider.model_profile('us.anthropic.claude-haiku-4-5-20251001-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'supported_native_tools': frozenset(),
            'bedrock_supports_tool_choice': True,
            'bedrock_send_back_thinking_parts': True,
            'bedrock_tool_result_colocatable_content': frozenset({'image', 'text'}),
            'bedrock_supports_leading_assistant_message': True,
            'bedrock_supports_prompt_caching': True,
            'bedrock_supports_adaptive_thinking': False,
            'bedrock_supports_effort': False,
            'bedrock_top_k_variant': 'anthropic',
            'bedrock_supports_tool_caching': True,
            'supports_json_schema_output': True,
            'bedrock_supported_media_kinds_in_tool_returns': frozenset({'document', 'image'}),
            'anthropic_supports_forced_tool_choice': True,
            'bedrock_thinking_variant': 'anthropic',
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'bedrock_supports_strict_tool_definition': True,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_anthropic_legacy_claude_3():
    """Older Claude via Bedrock — should still resolve, no JSON schema."""
    profile = BedrockProvider.model_profile('anthropic.claude-3-5-sonnet-20240620-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'supported_native_tools': frozenset(),
            'bedrock_supports_tool_choice': True,
            'bedrock_send_back_thinking_parts': True,
            'bedrock_tool_result_colocatable_content': frozenset({'image', 'text'}),
            'bedrock_supports_leading_assistant_message': True,
            'bedrock_supports_prompt_caching': True,
            'bedrock_supports_adaptive_thinking': False,
            'bedrock_supports_effort': False,
            'bedrock_top_k_variant': 'anthropic',
            'bedrock_supports_tool_caching': True,
            'bedrock_supported_media_kinds_in_tool_returns': frozenset({'document', 'image'}),
            'anthropic_supports_forced_tool_choice': True,
            'bedrock_thinking_variant': 'anthropic',
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'bedrock_supports_strict_tool_definition': False,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_mistral_large():
    profile = BedrockProvider.model_profile('mistral.mistral-large-2407-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supported_native_tools': frozenset(),
            'bedrock_tool_result_format': 'json',
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'bedrock_supports_strict_tool_definition': False,
            'bedrock_tool_result_colocatable_content': frozenset(),
            'bedrock_supported_media_kinds_in_tool_returns': frozenset({'document'}),
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_amazon_nova_pro():
    profile = BedrockProvider.model_profile('us.amazon.nova-pro-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'supported_native_tools': frozenset(),
            'bedrock_supports_tool_choice': True,
            'bedrock_supports_prompt_caching': True,
            'bedrock_top_k_variant': 'nova',
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_amazon_nova_2_lite():
    """Nova 2 adds `CodeExecutionTool` to `supported_native_tools` — verifies the strip-then-restore pattern."""
    profile = BedrockProvider.model_profile('us.amazon.nova-2-lite-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'supported_native_tools': frozenset({CodeExecutionTool}),
            'bedrock_supports_tool_choice': True,
            'bedrock_supports_prompt_caching': True,
            'bedrock_top_k_variant': 'nova',
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_amazon_titan():
    """Titan models — basic Amazon profile, no Nova-specific overrides."""
    profile = BedrockProvider.model_profile('amazon.titan-text-express-v1')
    assert _normalize(profile) == snapshot(
        {'json_schema_transformer': InlineDefsJsonSchemaTransformer, 'supported_native_tools': frozenset()}
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_cohere_command():
    profile = BedrockProvider.model_profile('cohere.command-r-plus-v1:0')
    assert _normalize(profile) == snapshot({'supported_native_tools': frozenset()})


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_meta_llama3():
    profile = BedrockProvider.model_profile('meta.llama3-70b-instruct-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'supported_native_tools': frozenset(),
            'bedrock_tool_result_colocatable_content': frozenset(),
            'bedrock_supported_media_kinds_in_tool_returns': frozenset({'image', 'document'}),
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_deepseek_r1():
    """`bedrock_send_back_thinking_parts=True` applied for `r1` models via separate function."""
    profile = BedrockProvider.model_profile('deepseek.deepseek-r1-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'ignore_streamed_leading_whitespace': True,
            'supported_native_tools': frozenset(),
            'bedrock_send_back_thinking_parts': True,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_openai():
    """Bedrock-hosted OpenAI — only has the `openai` thinking variant, no upstream profile."""
    profile = BedrockProvider.model_profile('openai.gpt-oss-120b-1:0')
    assert _normalize(profile) == snapshot(
        {'supports_thinking': True, 'bedrock_thinking_variant': 'openai', 'thinking_always_enabled': True}
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_qwen_qwq():
    profile = BedrockProvider.model_profile('qwen.qwq-32b-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'supported_native_tools': frozenset(),
            'bedrock_supports_leading_assistant_message': True,
            'supports_thinking': True,
            'bedrock_thinking_variant': 'qwen',
            'thinking_always_enabled': True,
            'bedrock_supports_strict_tool_definition': False,
            'bedrock_supported_media_kinds_in_tool_returns': frozenset(),
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_zai_glm():
    """Z.AI GLM via Bedrock: upstream `zai_model_profile` (thinking) + Bedrock tool/output overrides."""
    profile = BedrockProvider.model_profile('zai.glm-5')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'zai_supports_reasoning_effort': False,
            'supported_native_tools': frozenset(),
            'bedrock_supports_tool_choice': True,
            'bedrock_supports_leading_assistant_message': True,
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'supports_json_schema_output': True,
            'bedrock_supports_strict_tool_definition': True,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_moonshotai_kimi():
    """Moonshot AI Kimi via Bedrock (both `moonshot.` and `moonshotai.` prefixes share one profile)."""
    profile = BedrockProvider.model_profile('moonshotai.kimi-k2.5')
    assert _normalize(profile) == snapshot(
        {
            'ignore_streamed_leading_whitespace': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(),
            'bedrock_supports_tool_choice': True,
            'bedrock_supports_leading_assistant_message': True,
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'supports_json_schema_output': True,
            'bedrock_supports_strict_tool_definition': True,
            'bedrock_supported_media_kinds_in_tool_returns': frozenset(),
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_writer_palmyra():
    """Writer Palmyra via Bedrock — no upstream profile; isolates `toolResult` in its own turn."""
    profile = BedrockProvider.model_profile('writer.palmyra-x4-v1:0')
    assert _normalize(profile) == snapshot(
        {
            'supported_native_tools': frozenset(),
            'json_schema_transformer': BedrockJsonSchemaTransformer,
            'bedrock_tool_result_colocatable_content': frozenset(),
            'bedrock_supports_tool_result_status': False,
        }
    )


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
def test_bedrock_unknown_provider_returns_none():
    """Bedrock model IDs from an unknown provider prefix → `None`."""
    assert BedrockProvider.model_profile('unknown.foo-v1:0') is None


# =============================================================================
# OpenRouter — cross-provider routing via the OpenAI-compat surface.
# Three-layer merge: OpenAI transformer fallback / lab upstream / OpenRouter overrides.
# =============================================================================


def test_openrouter_anthropic_claude_sonnet_4_6():
    """Anthropic via OpenRouter — relays Anthropic's profile through OpenAI chat."""
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('anthropic/claude-sonnet-4-6')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_supports_adaptive_thinking': True,
            'anthropic_supports_effort': True,
            'anthropic_supports_dynamic_filtering': True,
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
            'anthropic_supports_forced_tool_choice': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'openrouter_supports_cache_control': True,
            'openrouter_supports_cache_ttl': True,
            'openrouter_supports_tool_cache': True,
            'openrouter_supports_dynamic_instruction_cache': True,
            'openrouter_max_cache_points': 4,
        }
    )


def test_openrouter_openai_gpt_5_4():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('openai/gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
            'openai_chat_supports_max_completion_tokens': False,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_google_gemini_3_pro():
    """Google via OpenRouter — uses `_OpenRouterGoogleJsonSchemaTransformer` (lab wins over OpenAI fallback)."""
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('google/gemini-3.0-pro')
    assert _normalize(profile) == snapshot(
        {
            'supports_tool_return_schema': True,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': _OpenRouterGoogleJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'google_supports_tool_combination': True,
            'google_supports_server_side_tool_invocations': True,
            'google_supported_mime_types_in_tool_returns': (
                'image/png',
                'image/jpeg',
                'image/webp',
                'application/pdf',
                'text/plain',
            ),
            'google_supports_thinking_level': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'openrouter_supports_cache_control': True,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_mistral_large():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('mistralai/mistral-large-latest')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'supports_thinking': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_xai_grok_4():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('x-ai/grok-4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'grok_supports_builtin_tools': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_qwen():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('qwen/qwen3-235b-a22b')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'supports_thinking': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_deepseek():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('deepseek/deepseek-chat')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_meta_llama():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('meta-llama/llama-3.3-70b-instruct')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'supports_thinking': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_moonshotai():
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('moonshotai/kimi-k2-0905')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'supports_thinking': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


def test_openrouter_unknown_provider_falls_back_to_overlay_only():
    """Unknown lab → no upstream profile, but OpenRouter overrides still apply."""
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    profile = OpenRouterProvider.model_profile('unknown-lab/unknown-model')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_thinking_field': 'reasoning',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_chat_supports_web_search': True,
            'openai_chat_supports_file_urls': True,
            'openai_chat_supports_max_completion_tokens': False,
            'supports_thinking': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_cache_ttl': False,
            'openrouter_supports_tool_cache': False,
            'openrouter_supports_dynamic_instruction_cache': False,
            'openrouter_max_cache_points': None,
        }
    )


# =============================================================================
# Azure — three-layer merge, prefix-based lab dispatch
# =============================================================================


def test_azure_openai_gpt_5():
    """Azure OpenAI — bare model name."""
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile('gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, ToolSearchTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
            'openai_chat_supports_document_input': False,
        }
    )


def test_azure_mistral_prefix():
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile('mistral-large-latest')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_supports_document_input': False,
        }
    )


def test_azure_mistral_small_latest():
    """Azure reuses the shared Mistral profile, so `thinking` is ignored: adjustable reasoning is native-provider-only."""
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile('mistral-small-latest')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_supports_document_input': False,
        }
    )


def test_azure_cohere_prefix():
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile('cohere-command-r-plus')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_supports_document_input': False,
        }
    )


def test_azure_grok_prefix():
    from pydantic_ai.providers.azure import AzureProvider

    profile = AzureProvider.model_profile('grok-4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'grok_supports_builtin_tools': True,
            'openai_chat_supports_document_input': False,
        }
    )


# =============================================================================
# Groq (inference provider — runs models themselves) — three-layer merge
# =============================================================================


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
def test_groq_moonshotai_kimi():
    """Groq's MoonshotAI route — `supports_json_object_output` and `supports_json_schema_output`
    are pre-set as fallbacks (upstream Moonshot profile wins if it sets them)."""
    profile = GroqProvider.model_profile('moonshotai/kimi-k2-0905')
    assert _normalize(profile) == snapshot(
        {
            'groq_supports_reasoning_disable': False,
            'groq_supports_graded_reasoning_effort': False,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'ignore_streamed_leading_whitespace': True,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
def test_groq_meta_llama4_maverick():
    """Special-cased Llama 4 Maverick gets `supports_json_object_output=True` overlay."""
    profile = GroqProvider.model_profile('llama-4-maverick-17b-128e-instruct')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'groq_supports_reasoning_disable': False,
            'groq_supports_graded_reasoning_effort': False,
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
def test_groq_meta_llama3_no_overlay():
    """Older Llama models don't get the structured-output overlay."""
    profile = GroqProvider.model_profile('llama-3.3-70b-versatile')
    assert _normalize(profile) == snapshot(
        {
            'groq_supports_reasoning_disable': False,
            'groq_supports_graded_reasoning_effort': False,
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
def test_groq_deepseek():
    profile = GroqProvider.model_profile('deepseek-r1-distill-llama-70b')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'groq_supports_reasoning_disable': False,
            'groq_supports_graded_reasoning_effort': False,
            'ignore_streamed_leading_whitespace': True,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
def test_groq_gpt_oss():
    """Groq's gpt-oss reasons always-on with graded `reasoning_effort` (`low`/`medium`/`high`);
    the Groq profile's reasoning flags override the generic OpenAI family profile."""
    profile = GroqProvider.model_profile('openai/gpt-oss-120b')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'groq_supports_reasoning_disable': False,
            'groq_supports_graded_reasoning_effort': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool}
            ),
            'supports_inline_system_prompts': True,
            'supports_json_object_output': True,
            'supports_json_schema_output': True,
        }
    )


# =============================================================================
# xAI (native, single-layer post-M4)
# =============================================================================


@pytest.mark.skipif(not xai_imports(), reason='xai not installed')
def test_xai_provider_grok_4():
    profile = XaiProvider.model_profile('grok-4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'grok_supports_builtin_tools': True,
        }
    )


@pytest.mark.skipif(not xai_imports(), reason='xai not installed')
def test_xai_provider_grok_3_mini():
    profile = XaiProvider.model_profile('grok-3-mini')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'grok_reasoning_efforts': frozenset({'low', 'high'}),
            'supported_native_tools': frozenset(),
        }
    )


# =============================================================================
# Ollama — three-layer merge with strict-mode override
# =============================================================================


def test_ollama_gpt_oss():
    from pydantic_ai.providers.ollama import OllamaProvider

    profile = OllamaProvider.model_profile('gpt-oss:20b')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'ignore_streamed_leading_whitespace': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool}
            ),
            'openai_chat_thinking_field': 'reasoning',
            'openai_supports_strict_tool_definition': False,
            'openai_supports_tool_choice_required': False,
        }
    )


def test_ollama_unknown_falls_back_to_overlay_only():
    from pydantic_ai.providers.ollama import OllamaProvider

    profile = OllamaProvider.model_profile('some-unknown-model')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'openai_chat_thinking_field': 'reasoning',
            'openai_supports_strict_tool_definition': False,
        }
    )


# =============================================================================
# Cerebras — three-layer merge with reasoning detection
# =============================================================================


def test_cerebras_qwen_reasoning():
    from pydantic_ai.providers.cerebras import CerebrasProvider

    profile = CerebrasProvider.model_profile('qwen-3-235b-a22b-thinking-2507')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'openai_unsupported_model_settings': (
                'frequency_penalty',
                'logit_bias',
                'presence_penalty',
                'parallel_tool_calls',
                'service_tier',
                'openai_service_tier',
            ),
        }
    )


def test_cerebras_llama_non_reasoning():
    from pydantic_ai.providers.cerebras import CerebrasProvider

    profile = CerebrasProvider.model_profile('llama-3.3-70b')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'openai_unsupported_model_settings': (
                'frequency_penalty',
                'logit_bias',
                'presence_penalty',
                'parallel_tool_calls',
                'service_tier',
                'openai_service_tier',
            ),
        }
    )


# =============================================================================
# OpenAI-compat passthrough gateways — two-layer merge: gateway transformer fallback + upstream wins
# =============================================================================


def test_litellm_openai_gpt():
    from pydantic_ai.providers.litellm import LiteLLMProvider

    profile = LiteLLMProvider.model_profile('gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, ToolSearchTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
        }
    )


def test_litellm_magistral():
    """Magistral's always-on flags survive the LiteLLM route. The sparse family profile skips the
    OpenAI baseline (structured output), a pre-existing gap shared with deepseek and cohere."""
    from pydantic_ai.providers.litellm import LiteLLMProvider

    profile = LiteLLMProvider.model_profile('mistral/magistral-medium-latest')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
        }
    )


def test_litellm_mistral_small_latest():
    """LiteLLM must not advertise thinking for adjustable Mistral ids (it rejects `reasoning_effort`
    for them); the route falls back to the plain OpenAI profile."""
    from pydantic_ai.providers.litellm import LiteLLMProvider

    profile = LiteLLMProvider.model_profile('mistral/mistral-small-latest')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_inline_system_prompts': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, WebSearchTool}
            ),
        }
    )


def test_fireworks_llama():
    from pydantic_ai.providers.fireworks import FireworksProvider

    profile = FireworksProvider.model_profile('accounts/fireworks/models/llama-v3p3-70b-instruct')
    assert _normalize(profile) == snapshot({'json_schema_transformer': InlineDefsJsonSchemaTransformer})


def test_together_qwen():
    from pydantic_ai.providers.together import TogetherProvider

    profile = TogetherProvider.model_profile('Qwen/Qwen2.5-72B-Instruct-Turbo')
    assert _normalize(profile) == snapshot(
        {'json_schema_transformer': InlineDefsJsonSchemaTransformer, 'ignore_streamed_leading_whitespace': True}
    )


def test_sambanova_llama():
    from pydantic_ai.providers.sambanova import SambaNovaProvider

    profile = SambaNovaProvider.model_profile('Meta-Llama-3.3-70B-Instruct')
    assert _normalize(profile) == snapshot({'json_schema_transformer': InlineDefsJsonSchemaTransformer})


def test_ovhcloud_llama():
    from pydantic_ai.providers.ovhcloud import OVHcloudProvider

    profile = OVHcloudProvider.model_profile('llama-3.3-70b-instruct')
    assert _normalize(profile) == snapshot({'json_schema_transformer': InlineDefsJsonSchemaTransformer})


def test_alibaba_qwen():
    from pydantic_ai.providers.alibaba import AlibabaProvider

    profile = AlibabaProvider.model_profile('qwen3-235b-a22b-thinking-2507')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'openai_chat_supports_document_input': False,
            'ignore_streamed_leading_whitespace': True,
        }
    )


def test_alibaba_qwen_audio():
    """Alibaba audio models get `openai_chat_audio_input_encoding='uri'`."""
    from pydantic_ai.providers.alibaba import AlibabaProvider

    profile = AlibabaProvider.model_profile('qwen3-audio-0809-online')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'openai_chat_supports_document_input': False,
            'ignore_streamed_leading_whitespace': True,
        }
    )


# =============================================================================
# Default-only providers (no per-model variation in the profile function)
# =============================================================================


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
def test_anthropic_unknown_model_returns_some_profile():
    """Anthropic profile function returns *something* for unknown names — locks the default."""
    profile = AnthropicProvider.model_profile('some-future-claude')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': AnthropicJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_supports_forced_tool_choice': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, MCPServerTool, MemoryTool, WebFetchTool, WebSearchTool}
            ),
        }
    )


def test_moonshotai_kimi():
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

    profile = MoonshotAIProvider.model_profile('kimi-k2-0905')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'openai_chat_thinking_field': 'reasoning_content',
            'openai_chat_send_back_thinking_parts': 'field',
            'openai_supports_tool_choice_required': False,
        }
    )


# =============================================================================
# GitHub Models — multi-lab routing via OpenAIChatModel, 2-layer merge
# =============================================================================


def test_github_openai_bare_name():
    """Bare model names (no `/` prefix) route to `openai_model_profile`."""
    from pydantic_ai.providers.github import GitHubProvider

    profile = GitHubProvider.model_profile('gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, ToolSearchTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
        }
    )


def test_github_xai_grok():
    from pydantic_ai.providers.github import GitHubProvider

    profile = GitHubProvider.model_profile('xai/grok-4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'grok_supports_builtin_tools': True,
        }
    )


def test_github_meta_llama():
    from pydantic_ai.providers.github import GitHubProvider

    profile = GitHubProvider.model_profile('meta/llama-3.3-70b-instruct')
    assert _normalize(profile) == snapshot({'json_schema_transformer': InlineDefsJsonSchemaTransformer})


def test_github_deepseek():
    from pydantic_ai.providers.github import GitHubProvider

    profile = GitHubProvider.model_profile('deepseek/deepseek-r1')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'ignore_streamed_leading_whitespace': True,
        }
    )


# =============================================================================
# Vercel — multi-lab routing via OpenAIChatModel
# =============================================================================


def test_vercel_unknown_bare_name():
    """Bare names get OpenAI transformer overlay only."""
    from pydantic_ai.providers.vercel import VercelProvider

    profile = VercelProvider.model_profile('gpt-5.4')
    assert _normalize(profile) == snapshot({'json_schema_transformer': OpenAIJsonSchemaTransformer})


def test_vercel_anthropic_claude_sonnet():
    from pydantic_ai.providers.vercel import VercelProvider

    profile = VercelProvider.model_profile('anthropic/claude-sonnet-4-6')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'anthropic_supports_adaptive_thinking': True,
            'anthropic_supports_effort': True,
            'anthropic_supports_dynamic_filtering': True,
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
            'anthropic_supports_forced_tool_choice': True,
            'supported_native_tools': frozenset(
                {AdvisorTool, CodeExecutionTool, MCPServerTool, MemoryTool, ToolSearchTool, WebFetchTool, WebSearchTool}
            ),
        }
    )


def test_vercel_openai_gpt():
    from pydantic_ai.providers.vercel import VercelProvider

    profile = VercelProvider.model_profile('openai/gpt-5.4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'supports_image_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_inline_system_prompts': True,
            'supports_thinking': True,
            'supported_native_tools': frozenset(
                {CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool, ToolSearchTool, WebSearchTool}
            ),
            'openai_supports_encrypted_reasoning_content': True,
            'openai_supports_reasoning': True,
            'openai_supports_reasoning_effort_none': True,
            'openai_supports_phase': True,
        }
    )


def test_vercel_vertex_gemini():
    """Vercel routes `vertex/...` through `google_model_profile`."""
    from pydantic_ai.providers.vercel import VercelProvider

    profile = VercelProvider.model_profile('vertex/gemini-3.0-pro')
    assert _normalize(profile) == snapshot(
        {
            'supports_tool_return_schema': True,
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': GoogleJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'google_supports_tool_combination': True,
            'google_supports_server_side_tool_invocations': True,
            'google_supported_mime_types_in_tool_returns': (
                'image/png',
                'image/jpeg',
                'image/webp',
                'application/pdf',
                'text/plain',
            ),
            'google_supports_thinking_level': True,
        }
    )


def test_vercel_xai_grok():
    from pydantic_ai.providers.vercel import VercelProvider

    profile = VercelProvider.model_profile('xai/grok-4')
    assert _normalize(profile) == snapshot(
        {
            'supports_json_schema_output': True,
            'supports_json_object_output': True,
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'grok_supports_builtin_tools': True,
        }
    )


# =============================================================================
# Heroku — routes model names through family profiles (issue #6022)
# =============================================================================


def test_heroku_returns_openai_transformer():
    """Heroku routes the model name through its family profile (here: Anthropic),
    merged onto the OpenAI-compatible base so thinking isn't dropped (#6022)."""
    from pydantic_ai.providers.heroku import HerokuProvider

    profile = HerokuProvider.model_profile('claude-sonnet-4-6')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'thinking_tags': ('<thinking>', '</thinking>'),
            'supports_json_schema_output': True,
            'supports_thinking': True,
            'anthropic_supports_adaptive_thinking': True,
            'anthropic_supports_effort': True,
            'anthropic_supports_dynamic_filtering': True,
            'anthropic_default_code_execution_tool_version': '20260120',
            'anthropic_supported_code_execution_tool_versions': ('20250825', '20260120'),
            'anthropic_supports_forced_tool_choice': True,
            'supported_native_tools': frozenset(
                {AdvisorTool, CodeExecutionTool, MCPServerTool, MemoryTool, ToolSearchTool, WebFetchTool, WebSearchTool}
            ),
        }
    )


# =============================================================================
# Nebius — multi-lab routing
# =============================================================================


def test_nebius_bare_name():
    from pydantic_ai.providers.nebius import NebiusProvider

    profile = NebiusProvider.model_profile('some-model')
    assert _normalize(profile) == snapshot({'json_schema_transformer': OpenAIJsonSchemaTransformer})


def test_nebius_meta_llama():
    from pydantic_ai.providers.nebius import NebiusProvider

    profile = NebiusProvider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    assert _normalize(profile) == snapshot({'json_schema_transformer': InlineDefsJsonSchemaTransformer})


def test_nebius_deepseek():
    from pydantic_ai.providers.nebius import NebiusProvider

    profile = NebiusProvider.model_profile('deepseek-ai/DeepSeek-R1')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': OpenAIJsonSchemaTransformer,
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'ignore_streamed_leading_whitespace': True,
        }
    )


def test_nebius_qwen():
    from pydantic_ai.providers.nebius import NebiusProvider

    profile = NebiusProvider.model_profile('Qwen/Qwen3-235B-A22B')
    assert _normalize(profile) == snapshot(
        {'json_schema_transformer': InlineDefsJsonSchemaTransformer, 'ignore_streamed_leading_whitespace': True}
    )


def test_nebius_moonshotai():
    from pydantic_ai.providers.nebius import NebiusProvider

    profile = NebiusProvider.model_profile('moonshotai/Kimi-K2-Instruct-0905')
    assert _normalize(profile) == snapshot(
        {'json_schema_transformer': OpenAIJsonSchemaTransformer, 'ignore_streamed_leading_whitespace': True}
    )


# =============================================================================
# Hugging Face — multi-lab routing, no OpenAI overlay
# =============================================================================


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_bare_name_returns_none():
    """HF requires `provider/model` form; bare name → `None`."""

    assert _normalize(HuggingFaceProvider.model_profile('some-model')) is None


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_meta_llama():
    profile = HuggingFaceProvider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    assert _normalize(profile) == snapshot(
        {'json_schema_transformer': InlineDefsJsonSchemaTransformer, 'supports_inline_system_prompts': True}
    )


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_deepseek():
    profile = HuggingFaceProvider.model_profile('deepseek-ai/DeepSeek-R1')
    assert _normalize(profile) == snapshot(
        {
            'supports_thinking': True,
            'thinking_always_enabled': True,
            'ignore_streamed_leading_whitespace': True,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_qwen():
    profile = HuggingFaceProvider.model_profile('Qwen/Qwen3-235B-A22B')
    assert _normalize(profile) == snapshot(
        {
            'json_schema_transformer': InlineDefsJsonSchemaTransformer,
            'ignore_streamed_leading_whitespace': True,
            'supports_inline_system_prompts': True,
        }
    )


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_moonshotai():
    profile = HuggingFaceProvider.model_profile('moonshotai/Kimi-K2-Instruct-0905')
    assert _normalize(profile) == snapshot(
        {'ignore_streamed_leading_whitespace': True, 'supports_inline_system_prompts': True}
    )


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
def test_huggingface_unknown_provider_returns_none():
    """Unknown provider prefix → `None` (no fallback overlay like other gateways)."""

    assert _normalize(HuggingFaceProvider.model_profile('unknown/some-model')) is None
