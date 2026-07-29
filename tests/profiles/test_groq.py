"""Tests for Groq model profile detection.

Groq renamed its compound systems from `compound-beta`/`compound-beta-mini` to
`groq/compound`/`groq/compound-mini`. Both spellings must set
`groq_always_has_web_search_builtin_tool`, or `GroqModel` rejects `WebSearchTool` outright with a
`UserError` on the models that always have web search available.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.profiles.groq import groq_model_profile


@pytest.mark.parametrize(
    'model_name',
    [
        # Current IDs (https://console.groq.com/docs/compound). The responses in
        # `tests/models/cassettes/test_groq/test_groq_model_web_search_tool.yaml` come back with
        # `"model": "groq/compound"`.
        'groq/compound',
        'groq/compound-mini',
        # Pinned versions, which Groq documents alongside the default.
        'groq/compound-2025-08-16',
        'groq/compound-mini-2025-07-23',
        # The original IDs still resolve.
        'compound-beta',
        'compound-beta-mini',
    ],
)
def test_compound_models_always_have_web_search(model_name: str):
    """Compound systems run web search implicitly, under either the old or the new ID.

    `GroqModel._get_native_tools` raises `UserError('`WebSearchTool` is not supported by Groq')`
    unless this flag is set, so a missed match makes `WebSearchTool` unusable on the only Groq
    models that support it.
    """
    profile = groq_model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool') is True, model_name


@pytest.mark.parametrize(
    'model_name',
    [
        'llama-3.3-70b-versatile',
        'openai/gpt-oss-120b',
        'qwen/qwen3-32b',
        'moonshotai/kimi-k2-instruct',
        # `groq/` alone is not a compound system.
        'groq/llama-3.3-70b',
    ],
)
def test_non_compound_models_do_not_always_have_web_search(model_name: str):
    """The flag must not leak onto models where `WebSearchTool` really is unsupported."""
    profile = groq_model_profile(model_name)
    assert profile.get('groq_always_has_web_search_builtin_tool', False) is False, model_name


@pytest.mark.parametrize(
    'model_name',
    ['groq/compound', 'groq/compound-mini', 'compound-beta'],
)
def test_compound_models_are_not_reasoning_models(model_name: str):
    """Renaming shouldn't drag the compound IDs into any of the reasoning branches."""
    profile = groq_model_profile(model_name)
    assert profile.get('supports_thinking', False) is False, model_name
    assert profile.get('thinking_always_enabled', False) is False, model_name
    assert profile.get('groq_supports_reasoning_disable', False) is False, model_name
    assert profile.get('groq_supports_graded_reasoning_effort', False) is False, model_name
