"""Tests for the OpenAI Codex subscription-auth model profile.

Pins the Codex wire-dialect flags directly: the backend serves streaming responses only,
requires `store=false`, rejects sampling/tuning request fields, and exposes no server-side
input-token counting (all verified live on PR #6433).
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.openai import openai_model_profile
    from pydantic_ai.profiles.openai_codex import openai_codex_model_profile
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def test_codex_wire_dialect_flags():
    profile = openai_codex_model_profile('gpt-5.6-luna')
    assert profile.get('openai_responses_requires_streaming') is True
    assert profile.get('openai_responses_requires_store_false') is True
    assert profile.get('openai_supports_input_token_counting') is False
    assert profile.get('openai_unsupported_model_settings') == (
        'max_tokens',
        'temperature',
        'top_p',
    )


def test_codex_profile_extends_the_standard_openai_profile():
    """The codex profile is `openai_model_profile` plus the dialect overrides, nothing else."""
    base = dict(openai_model_profile('gpt-5.6-luna'))
    codex = dict(openai_codex_model_profile('gpt-5.6-luna'))
    overridden = {
        'openai_unsupported_model_settings',
        'openai_responses_requires_streaming',
        'openai_responses_requires_store_false',
        'openai_supports_input_token_counting',
    }
    assert {k: v for k, v in codex.items() if k not in overridden} == {
        k: v for k, v in base.items() if k not in overridden
    }


def test_codex_provider_layers_wire_dialect_over_first_party_profile():
    """The Codex provider preserves first-party flags beneath its narrower wire dialect."""
    openai = dict(OpenAIProvider.model_profile('gpt-5.6-luna') or {})
    codex = dict(OpenAICodexProvider.model_profile('gpt-5.6-luna') or {})
    dialect = {
        'openai_unsupported_model_settings',
        'openai_responses_requires_streaming',
        'openai_responses_requires_store_false',
        'openai_supports_input_token_counting',
    }

    assert {key: value for key, value in codex.items() if key not in dialect} == {
        key: value for key, value in openai.items() if key not in dialect
    }
    assert codex['tool_addition_mode'] == 'with_definitions'
    assert codex['tool_deferral_mode'] == 'with_tool_search'
    assert {key: codex[key] for key in dialect} == {
        key: openai_codex_model_profile('gpt-5.6-luna')[key] for key in dialect
    }
