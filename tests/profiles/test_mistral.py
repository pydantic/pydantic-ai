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
            'mistral-small-2603',
            'mistral-medium-latest',
            'mistral-medium-3-5',
            'mistral-medium-3.5',
            'mistral-medium-2604',
        ],
    )
    def test_adjustable_reasoning_models(self, model_name: str):
        profile = mistral_model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking') is True
        assert profile.get('thinking_always_enabled') is False


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
        assert profile.get('supports_thinking') is True
        assert profile.get('thinking_always_enabled') is True


class TestNonThinkingModels:
    """Models that don't support adjustable reasoning return None profile.

    Includes older `mistral-small-*` / `mistral-medium-*` snapshots that report `reasoning: false`
    on the Mistral `/v1/models` API, so they must not be treated as reasoning-capable.
    """

    @pytest.mark.parametrize(
        'model_name',
        [
            'mistral-large-latest',
            'codestral-latest',
            'mistral-moderation-latest',
            'devstral-medium-latest',
            'voxtral-small-latest',
            'mistral-small-2506',
            'mistral-medium-2505',
            'mistral-medium-2508',
            # exact-match must not over-match adjacent names
            'mistral-small-2603-preview',
            'mistral-medium-3-50',
        ],
    )
    def test_non_thinking_models(self, model_name: str):
        profile = mistral_model_profile(model_name)
        assert profile is None
