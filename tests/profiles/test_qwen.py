"""Tests for Qwen model profile detection.

Gateways disagree on whether the generation number in a Qwen model ID is hyphenated: Cerebras
serves `qwen-3-coder-480b`, while Bedrock, Heroku and OpenRouter all use `qwen3-coder-*`. Both
spellings must resolve the same profile, or the Qwen3-Coder capability flags are silently lost.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile


@pytest.mark.parametrize(
    'model_name',
    [
        # Cerebras (`tests/models/cassettes/.../test_openai_model_cerebras_provider_qwen_3_coder.yaml`).
        'qwen-3-coder-480b',
        # Heroku (`heroku:qwen3-coder-480b` in `_known_model_names.py`).
        'qwen3-coder-480b',
        # Bedrock (`bedrock:qwen.qwen3-coder-30b-a3b-v1:0`, `bedrock:qwen.qwen3-coder-next`), after
        # `split_bedrock_model_id` strips the `qwen.` vendor prefix.
        'qwen3-coder-30b-a3b-v1:0',
        'qwen3-coder-next',
        # OpenRouter, after the `qwen/` prefix is stripped.
        'qwen3-coder',
        'qwen3-coder-flash',
        'qwen3-coder-plus',
        'qwen3-coder-30b-a3b-instruct',
    ],
)
def test_qwen_3_coder_profile_accepts_both_generation_spellings(model_name: str):
    """Every real Qwen3-Coder ID must get the coder profile, hyphenated or not.

    Qwen3-Coder supports neither forced tool use nor strict tool definitions, so a missed match
    means those two parameters are sent to a model that rejects them.
    """
    profile = qwen_model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_supports_tool_choice_required') is False, model_name
    assert profile.get('openai_supports_strict_tool_definition') is False, model_name
    assert profile.get('json_schema_transformer') == InlineDefsJsonSchemaTransformer, model_name


@pytest.mark.parametrize(
    'model_name',
    [
        # Qwen2.5-Coder is a different generation and keeps the generic profile; the `^qwen-?3-coder`
        # anchor is what stops `qwen-2.5-coder-32b-instruct` from matching on the `-coder` suffix.
        'qwen-2.5-coder-32b-instruct',
        'qwen2.5-coder-32b',
        # Non-coder Qwen3 models.
        'qwen3-235b-a22b',
        'qwen-3-32b',
    ],
)
def test_non_qwen_3_coder_models_keep_the_generic_profile(model_name: str):
    """The coder flags must not leak onto models that do support these features."""
    profile = qwen_model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_supports_tool_choice_required') is None, model_name
    assert profile.get('openai_supports_strict_tool_definition') is None, model_name
