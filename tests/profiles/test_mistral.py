"""Tests for Mistral model profile detection.

- `supports_thinking`: whether the model supports thinking/reasoning
- `thinking_always_enabled`: whether thinking is always on (`magistral`) or opt-in via
  `reasoning_effort` (the adjustable-reasoning models)
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.mistral import mistral_model_profile

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
    """Models with adjustable reasoning: thinking supported, opt-in via `reasoning_effort`."""
    profile = mistral_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled') is False


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
