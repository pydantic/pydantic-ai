"""Tests for Groq model profiles.

Verifies the `groq_always_has_web_search_builtin_tool` flag is set for all Groq compound
model IDs — the legacy `compound-beta` names and the current `groq/compound` names — and
not for non-compound models.
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.groq import groq_model_profile
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='groq not installed'),
]


@pytest.mark.parametrize(
    'model_name',
    [
        # legacy names (still resolve server-side)
        'compound-beta',
        'compound-beta-mini',
        # current names
        'groq/compound',
        'groq/compound-mini',
        # pinned versions
        'groq/compound-2025-08-16',
        'groq/compound-mini-2025-07-23',
    ],
)
def test_groq_compound_models_have_web_search_builtin_tool(model_name: str):
    """`groq_always_has_web_search_builtin_tool` is True for every compound model ID."""
    profile = groq_model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool') is True


@pytest.mark.parametrize(
    'model_name',
    [
        'llama-3.3-70b-versatile',
        'groq/llama-3.3-70b',
        'openai/gpt-oss-120b',
        'qwen/qwen3-32b',
        'deepseek-r1-distill-llama-70b',
    ],
)
def test_groq_non_compound_models_do_not_have_web_search_builtin_tool(model_name: str):
    """`groq_always_has_web_search_builtin_tool` is False for non-compound models."""
    profile = groq_model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool') is False


@pytest.mark.parametrize(
    'model_name',
    [
        'compound-beta',
        'compound-beta-mini',
        'groq/compound',
        'groq/compound-mini',
        'groq/compound-2025-08-16',
        'groq/compound-mini-2025-07-23',
    ],
)
def test_groq_provider_resolves_compound_web_search_flag(model_name: str):
    """`GroqProvider.model_profile` routes compound IDs through the Groq compound profile."""
    profile = GroqProvider.model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool') is True


@pytest.mark.parametrize(
    'model_name',
    [
        'llama-3.3-70b-versatile',
        'groq/llama-3.3-70b',
    ],
)
def test_groq_provider_resolves_non_compound_web_search_flag(model_name: str):
    """`GroqProvider.model_profile` does not set the web search flag for non-compound models."""
    profile = GroqProvider.model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool') is False
