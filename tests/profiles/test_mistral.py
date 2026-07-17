"""Tests for Mistral model profile detection.

- `supports_thinking`: whether the model supports thinking/reasoning
- `thinking_always_enabled`: whether thinking is always on (`magistral`) or opt-in via
  `reasoning_effort` (the adjustable-reasoning models)

Adjustable reasoning lives on `MistralProvider`, not the shared `mistral_model_profile`, so the
15 OpenAI-compatible providers that reuse the shared profile keep ignoring `thinking` instead of
sending Mistral effort values their routes reject.
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.mistral import mistral_model_profile
    from pydantic_ai.providers.mistral import MistralProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
]


@pytest.mark.parametrize(
    'model_name',
    [
        'mistral-small-latest',
        'mistral-small-2603',
        'mistral-medium-latest',
        'mistral-medium-3-5',
        'mistral-medium-2604',
    ],
)
def test_adjustable_reasoning_models(model_name: str):
    """Models with adjustable reasoning: thinking supported on the native provider, opt-in via
    `reasoning_effort`."""
    profile = MistralProvider.model_profile(model_name)
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled') is False


def test_adjustable_reasoning_not_on_shared_profile():
    """The shared `mistral_model_profile` must NOT advertise thinking for adjustable models.

    It is consumed by the 15 OpenAI-compatible proxy providers (LiteLLM, Azure, etc.) whose
    routes reject Mistral's `reasoning_effort`, so thinking stays a native-provider concern.
    """
    assert mistral_model_profile('mistral-small-latest') is None
    assert MistralProvider.model_profile('mistral-small-latest').get('supports_thinking') is True


@pytest.mark.parametrize(
    'model_name',
    [
        'magistral-latest',
        'magistral-medium-latest',
        'magistral-small-latest',
    ],
)
def test_magistral_models(model_name: str):
    """Magistral models have thinking always enabled."""
    profile = mistral_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled') is True


@pytest.mark.parametrize(
    'model_name',
    [
        'mistral-large-latest',
        'codestral-latest',
        'mistral-moderation-latest',
        'devstral-medium-latest',
        'voxtral-small-latest',
        # older snapshots that predate adjustable reasoning
        'mistral-small-2506',
        'mistral-medium-2505',
        'mistral-medium-2508',
        # exact-match must not over-match adjacent names
        'mistral-small-2603-preview',
        'mistral-medium-3-50',
    ],
)
def test_non_thinking_models(model_name: str):
    """Models without adjustable reasoning get no profile, so `thinking` is stripped upstream."""
    profile = mistral_model_profile(model_name)
    assert profile is None
