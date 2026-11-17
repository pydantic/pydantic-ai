"""Tests for the ChatGPT plan usage (Sign in with ChatGPT) model profile.

Pins the dialect flags OpenAI documents in its preview limitations for ChatGPT plan usage:
https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.native_tools import WebSearchTool
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]

DIALECT = {
    'openai_unsupported_model_settings': ('max_tokens', 'service_tier', 'temperature', 'top_p'),
    'openai_system_prompt_role': 'developer',
    'openai_responses_requires_streaming': True,
    'openai_responses_requires_store_false': True,
    'openai_supports_input_token_counting': False,
    'openai_supports_batch_api': False,
    'openai_supports_prompt_cache_breakpoints': False,
    'openai_responses_supports_prompt_cache_diagnostics': False,
    'openai_responses_supports_max_reasoning_effort': False,
    'supports_cache': False,
    'supported_cache_retentions': (),
    'default_cache_retention': None,
    'supported_native_tools': frozenset({WebSearchTool}),
    'tool_deferral_mode': None,
}


def test_chatgpt_provider_layers_the_dialect_over_the_first_party_profile():
    """Everything but the documented dialect comes from the OpenAI provider's own profile."""
    openai = dict(OpenAIProvider.model_profile('gpt-6.1-sol') or {})
    chatgpt = dict(OpenAIChatGPTProvider.model_profile('gpt-6.1-sol') or {})

    assert {key: chatgpt[key] for key in DIALECT} == DIALECT
    assert {key: value for key, value in chatgpt.items() if key not in DIALECT} == {
        key: value for key, value in openai.items() if key not in DIALECT
    }
    # Tools added mid-conversation travel as `additional_tools` items, which the route accepts.
    assert chatgpt['tool_addition_mode'] == 'with_definitions'
