"""Tests for Qwen model profile detection.

`qwen_model_profile` must accept both spellings of the Qwen3-Coder generation
number: Cerebras serves `qwen-3-coder-*` (hyphenated), while Bedrock, Heroku and
OpenRouter serve `qwen3-coder-*` (un-hyphenated). Only the Qwen3-Coder branch
sets `openai_supports_tool_choice_required=False` and
`openai_supports_strict_tool_definition=False`; falling through to the generic
profile would wrongly advertise both capabilities.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.profiles.qwen import qwen_model_profile


@pytest.mark.parametrize(
    'model_name',
    [
        # Cerebras spelling (hyphenated):
        'qwen-3-coder-480b',
        # Bedrock, Heroku and OpenRouter spelling (un-hyphenated):
        'qwen3-coder-480b',
        'qwen3-coder-30b-a3b-v1:0',
        'qwen3-coder-next',
        'qwen3-coder',
        'qwen3-coder-flash',
        'qwen3-coder-plus',
        'qwen3-coder-30b-a3b-instruct',
    ],
)
def test_qwen3_coder_matches_both_spellings(model_name: str):
    """Both `qwen-3-coder-*` and `qwen3-coder-*` resolve to the Qwen3-Coder profile."""
    profile = qwen_model_profile(model_name)
    assert profile is not None
    assert profile.get('openai_supports_tool_choice_required') is False
    assert profile.get('openai_supports_strict_tool_definition') is False


@pytest.mark.parametrize(
    'model_name',
    [
        # Qwen2.5-Coder is a different generation and must keep the generic profile.
        'qwen-2.5-coder-32b-instruct',
        'qwen-2.5-coder-32b',
        # Unrelated coder/non-coder models.
        'deepseek-coder',
        'qwen3-235b-a22b',
    ],
)
def test_non_qwen3_coder_does_not_match_coder_branch(model_name: str):
    """The `^` anchor prevents `qwen-2.5-coder-*` and unrelated models from matching."""
    profile = qwen_model_profile(model_name)
    assert profile is not None
    # The coder branch is the only one that sets these to False; the generic
    # profile leaves them at their True defaults.
    assert profile.get('openai_supports_tool_choice_required') is not False
    assert profile.get('openai_supports_strict_tool_definition') is not False
