"""Tests for MoonshotAI (Kimi) model profile detection.

Gateways disagree on how to punctuate a Kimi minor version: MoonshotAI, OpenRouter and Bedrock
serve `kimi-k2.5`, while Heroku serves `kimi-k2-5`. `kimi-k2-thinking` is a distinct reasoning
model rather than a minor version. Every one of these must resolve `supports_thinking=True`, or
`ModelSettings['thinking']` is silently dropped for them.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.profiles.moonshotai import moonshotai_model_profile


@pytest.mark.parametrize(
    'model_name',
    [
        # MoonshotAI's own IDs (`moonshotai:…` in `_known_model_names.py`).
        'kimi-k2.5',
        'kimi-k2.6',
        'kimi-k2.7-code',
        'kimi-k2.7-code-highspeed',
        'kimi-k3',
        'kimi-thinking-preview',
        # Heroku punctuates the minor version with a hyphen (`heroku:kimi-k2-5`).
        'kimi-k2-5',
        # `kimi-k2-thinking` (`heroku:kimi-k2-thinking`, `bedrock:moonshot.kimi-k2-thinking`, and
        # OpenRouter's `moonshotai/kimi-k2-thinking`, which reports `reasoning` in
        # `supported_parameters`).
        'kimi-k2-thinking',
        # Case is normalised, since gateways such as Nebius serve mixed-case IDs.
        'Kimi-K2-Thinking',
        'Kimi-K2.5',
    ],
)
def test_kimi_reasoning_models_support_thinking(model_name: str):
    """Every real Kimi reasoning ID must advertise `supports_thinking`.

    `models/__init__.py` strips `thinking` from `model_settings` unconditionally but only forwards
    it when the profile advertises support, so a missed match discards the setting with no error.
    """
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True, model_name


@pytest.mark.parametrize(
    'model_name',
    [
        # Plain K2 and its dated snapshots are not reasoning models — OpenRouter reports no
        # `reasoning` in `supported_parameters` for `moonshotai/kimi-k2` or `kimi-k2-0905`. The
        # `k2-thinking` alternative must not widen to these.
        'kimi-k2',
        'kimi-k2-0905',
        'kimi-k2-0711-preview',
        'kimi-k2-instruct',
        # `kimi-latest` currently points at a non-thinking model.
        'kimi-latest',
        # The moonshot-v1 family predates reasoning entirely.
        'moonshot-v1-8k',
        'moonshot-v1-128k',
    ],
)
def test_non_reasoning_moonshot_models_do_not_support_thinking(model_name: str):
    """`supports_thinking` must not leak onto models that reject `reasoning_effort`."""
    profile = moonshotai_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is False, model_name


def test_all_moonshot_models_ignore_streamed_leading_whitespace():
    """The shared quirk applies regardless of the reasoning branch."""
    for model_name in ('kimi-k2-thinking', 'kimi-k2', 'moonshot-v1-8k'):
        profile = moonshotai_model_profile(model_name)
        assert profile is not None
        assert profile.get('ignore_streamed_leading_whitespace') is True, model_name
