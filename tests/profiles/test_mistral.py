"""Tests for Mistral model profile detection.

Adjustable reasoning lives on `MistralProvider`, not the shared `mistral_model_profile`: the
OpenAI-compatible providers that reuse the shared profile must keep ignoring `thinking` instead
of sending effort values Mistral rejects.
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
    from pydantic_ai.profiles.mistral import mistral_model_profile
    from pydantic_ai.providers.mistral import MistralProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
]


@pytest.mark.parametrize(
    'model_name',
    [
        'mistral-large-4',
        'mistral-large-4-0',
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
    assert not mistral_model_profile('mistral-small-latest').get('supports_thinking')
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
    'mistral-large-4-preview',
]


@pytest.mark.parametrize('model_name', _NON_REASONING_MODELS)
def test_non_thinking_models(model_name: str):
    """Models without adjustable reasoning don't advertise thinking, so `thinking` is stripped upstream."""
    profile = mistral_model_profile(model_name)
    assert not profile.get('supports_thinking')


@pytest.mark.parametrize('model_name', _NON_REASONING_MODELS)
def test_non_thinking_models_native_provider(model_name: str):
    """The native provider must not advertise thinking for these ids either; catches allowlist widening."""
    profile = MistralProvider.model_profile(model_name)
    assert not profile.get('supports_thinking')


@pytest.mark.parametrize(
    'model_name', ['mistral-small-latest', 'magistral-medium-latest', 'devstral-latest', 'voxtral-small-latest']
)
def test_tool_call_id_length(model_name: str):
    """Every Mistral model gets the 9-alphanumeric tool call ID constraint, on the shared profile and the native provider."""
    assert mistral_model_profile(model_name).get('tool_call_id_length') == 9
    assert MistralProvider.model_profile(model_name).get('tool_call_id_length') == 9


def test_zai_glm_5_3_inlines_defs_for_tool_schemas():
    """GLM-5.3 nested tool schemas must not use `$defs` references."""
    assert mistral_model_profile('zai-glm-5-3').get('json_schema_transformer') == InlineDefsJsonSchemaTransformer
    assert (
        MistralProvider.model_profile('zai-glm-5-3').get('json_schema_transformer') == InlineDefsJsonSchemaTransformer
    )
