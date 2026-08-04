"""Tests for MoonshotAI/Kimi model profile detection.

The shared `moonshotai_model_profile` gates `supports_thinking` on the model id, and the
minor-version separator is spelled inconsistently across providers (`kimi-k2.5` on
Moonshot/Bedrock, `kimi-k2-5` on Heroku) with a `kimi-k2-thinking` alias. A regression that
only matches one spelling silently drops the `thinking` setting, so these assertions pin
the regex against both separators and the alias while keeping the bare `kimi-k2`/instruct
models excluded.
"""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.moonshotai import moonshotai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='moonshotai not installed'),
]


_REASONING_MODELS = [
    'kimi-k2.5',
    'kimi-k2-5',
    'kimi-k2.6',
    'kimi-k2.7-code',
    'kimi-k2.7-code-highspeed',
    'kimi-k2-thinking',
    'Kimi-K2-Thinking',
    'kimi-k3',
    'kimi-thinking',
    'kimi-thinking-preview',
]


_NON_REASONING_MODELS = [
    'kimi-k2',
    'kimi-k2-0905',
    'kimi-k2-0711-preview',
    'kimi-k2-instruct',
    'kimi-latest',
    'moonshot-v1-8k',
    'moonshot-v1-128k',
    'moonshot-v1-auto',
]


@pytest.mark.parametrize('model_name', _REASONING_MODELS)
def test_reasoning_models_advertise_thinking(model_name: str):
    """Reasoning ids resolve `supports_thinking=True` regardless of dot/hyphen separator or `k2-thinking` alias."""
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True


@pytest.mark.parametrize('model_name', _NON_REASONING_MODELS)
def test_non_reasoning_models_do_not_advertise_thinking(model_name: str):
    """Bare `kimi-k2`/instruct/v1 models genuinely lack reasoning, so `supports_thinking` stays False."""
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is False


def test_ignore_streamed_leading_whitespace_always_present():
    """The Moonshot profile always sets `ignore_streamed_leading_whitespace=True` (shared across all Kimi spellings)."""
    profile = moonshotai_model_profile('kimi-k2-thinking')
    assert profile is not None
    assert profile.get('ignore_streamed_leading_whitespace') is True
