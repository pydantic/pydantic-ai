from __future__ import annotations as _annotations

import pytest

from pydantic_ai.profiles.moonshotai import moonshotai_model_profile


@pytest.mark.parametrize(
    'model_name',
    ['kimi-k2.5', 'kimi-k2-thinking', 'Kimi-K2-Thinking'],
)
def test_kimi_reasoning_models_support_thinking(model_name: str):
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True


@pytest.mark.parametrize(
    'model_name',
    ['kimi-k2', 'kimi-k2-0905'],
)
def test_non_reasoning_moonshot_models_do_not_support_thinking(model_name: str):
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is False
