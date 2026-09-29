"""Tests for Mistral model profile detection.

Adjustable reasoning lives on `MistralProvider`, not the shared `mistral_model_profile`: the
OpenAI-compatible providers that reuse the shared profile must keep ignoring `thinking` instead
of sending effort values Mistral rejects.
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
        'mistral-medium',
        'mistral-medium-3',
        'mistral-medium-3-5',
        'mistral-medium-3.5',
        'mistral-medium-2604',
    ],
)
def test_adjustable_reasoning_models(model_name: str):
    """Adjustable-reasoning ids: the native provider advertises thinking, opt-in via `reasoning_effort`."""
    profile = MistralProvider.model_profile(model_name)
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled') is False


def test_adjustable_reasoning_not_on_shared_profile():
    """The shared profile must NOT advertise thinking for adjustable ids (see module docstring)."""
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


_NON_REASONING_MODELS = [
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
]


@pytest.mark.parametrize('model_name', _NON_REASONING_MODELS)
def test_non_thinking_models(model_name: str):
    """Models without adjustable reasoning get no profile, so `thinking` is stripped upstream."""
    profile = mistral_model_profile(model_name)
    assert profile is None


@pytest.mark.parametrize('model_name', _NON_REASONING_MODELS)
def test_non_thinking_models_native_provider(model_name: str):
    """The native provider must not advertise thinking for these ids either; catches allowlist widening."""
    profile = MistralProvider.model_profile(model_name)
    assert not profile.get('supports_thinking')
