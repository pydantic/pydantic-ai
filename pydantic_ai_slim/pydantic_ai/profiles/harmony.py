from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIModelProfile, openai_model_profile


def harmony_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for the OpenAI Harmony Response format.

    See <https://cookbook.openai.com/articles/openai-harmony> for more details.
    """
    profile = openai_model_profile(model_name)
    return OpenAIModelProfile(
        openai_supports_tool_choice_required=False, ignore_streamed_leading_whitespace=True
    ).update(profile)
