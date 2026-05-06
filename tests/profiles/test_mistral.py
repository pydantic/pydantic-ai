"""Tests for Mistral model profiles.

Tests verify model profile detection for different Mistral models, particularly:
- `supports_thinking`: Whether the model supports thinking/reasoning
- `thinking_always_enabled`: Whether thinking is always on (e.g. Magistral)
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.mistral import mistral_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
]


class TestAdjustableReasoningModels:
    """Models with adjustable reasoning (supports_thinking=True, thinking_always_enabled=False)."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'mistral-small-latest',
            'mistral-small-2501',
            'mistral-medium-latest',
            'mistral-medium-3-5',
        ],
    )
    def test_adjustable_reasoning_models(self, model_name: str):
        profile = mistral_model_profile(model_name)
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is False


class TestAlwaysThinkingModels:
    """Magistral models have thinking always enabled."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'magistral-latest',
            'magistral-medium-latest',
            'magistral-small-latest',
        ],
    )
    def test_magistral_models(self, model_name: str):
        profile = mistral_model_profile(model_name)
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True


class TestNonThinkingModels:
    """Models that don't support adjustable reasoning return None profile."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'mistral-large-latest',
            'codestral-latest',
            'mistral-moderation-latest',
        ],
    )
    def test_non_thinking_models(self, model_name: str):
        profile = mistral_model_profile(model_name)
        assert profile is None
