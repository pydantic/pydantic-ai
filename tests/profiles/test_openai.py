"""Tests for OpenAI model profiles.

Tests verify model profile detection for different OpenAI models, particularly:
- `openai_supports_reasoning`: Whether the model supports reasoning (o-series, GPT-5, GPT-5.1+)
- `openai_supports_reasoning_effort_none`: GPT-5.1+ models support sampling params when reasoning_effort='none'
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


class TestSamplingParamsSupport:
    """Tests for sampling parameter support based on model reasoning capabilities."""

    def test_o_series_supports_reasoning(self):
        """o-series models (o1, o3, etc.) always have reasoning enabled."""
        for model in ['o1', 'o1-mini', 'o3', 'o3-mini', 'o4-mini']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_reasoning is True
            assert profile.openai_supports_reasoning_effort_none is False

    def test_gpt_5_supports_reasoning(self):
        """GPT-5 (not 5.1+) always has reasoning enabled."""
        for model in ['gpt-5', 'gpt-5-pro', 'gpt-5-turbo']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_reasoning is True
            assert profile.openai_supports_reasoning_effort_none is False

    def test_gpt_5_1_supports_reasoning_with_effort_none(self):
        """GPT-5.1 models support reasoning with reasoning_effort='none' (the default)."""
        for model in ['gpt-5.1', 'gpt-5.1-turbo', 'gpt-5.1-mini', 'gpt-5.1-codex-max']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_reasoning is True
            assert profile.openai_supports_reasoning_effort_none is True

    def test_gpt_5_2_supports_reasoning_with_effort_none(self):
        """GPT-5.2 models support reasoning with reasoning_effort='none' (the default)."""
        for model in ['gpt-5.2', 'gpt-5.2-turbo', 'gpt-5.2-mini']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_reasoning is True
            assert profile.openai_supports_reasoning_effort_none is True

    def test_gpt_5_chat_no_reasoning(self):
        """GPT-5-chat models don't have reasoning and support all sampling params."""
        profile = openai_model_profile('gpt-5-chat')
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.openai_supports_reasoning is False
        assert profile.openai_supports_reasoning_effort_none is False

    def test_gpt_4o_no_reasoning(self):
        """GPT-4o models don't have reasoning and support all sampling params."""
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-08-06']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_reasoning is False
            assert profile.openai_supports_reasoning_effort_none is False


class TestEncryptedReasoningContent:
    """Tests for encrypted reasoning content support."""

    def test_reasoning_models_support_encrypted_content(self):
        """Models with reasoning support encrypted reasoning content."""
        for model in ['o1', 'o3', 'gpt-5', 'gpt-5.1', 'gpt-5.2']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is True

    def test_non_reasoning_models_no_encrypted_content(self):
        """Models without reasoning don't support encrypted reasoning content."""
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-5-chat']:
            profile = openai_model_profile(model)
            assert isinstance(profile, OpenAIModelProfile)
            assert profile.openai_supports_encrypted_reasoning_content is False
