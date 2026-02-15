from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIModelProfile, openai_model_profile


def harmony_model_profile(model_name: str) -> ModelProfile:
    """The model profile for the OpenAI Harmony Response format.

    See <https://cookbook.openai.com/articles/openai-harmony> for more details.
    """
    base_profile = openai_model_profile(model_name)

    # GPT-OSS models (like gpt-oss-120b) support reasoning
    model_lower = model_name.lower()
    is_reasoning = 'gpt-oss' in model_lower

    # Create harmony-specific overrides and merge with base profile
    harmony_overrides = OpenAIModelProfile(
        openai_supports_tool_choice_required=False,
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
        thinking_always_enabled=is_reasoning,  # Only always-on for reasoning models
    )

    return base_profile.update(harmony_overrides)
