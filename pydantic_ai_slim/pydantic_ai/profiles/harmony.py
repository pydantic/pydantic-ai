from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .openai import OpenAIModelProfile, openai_model_profile


def harmony_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for the OpenAI Harmony Response format.

    See <https://cookbook.openai.com/articles/openai-harmony> for more details.
    """
    return merge_profile(
        openai_model_profile(model_name),
        OpenAIModelProfile(
            openai_supports_tool_choice_required=False,
            ignore_streamed_leading_whitespace=True,
            # Harmony models reason unconditionally — reasoning is always-on with a `medium`
            # default and no off switch (see the guide linked in the docstring). Setting these
            # here ensures bare gpt-oss names that fall through `_REASONING_SUPPORT_BY_PREFIX`
            # (which only enumerates OpenAI Responses API models) still resolve thinking support,
            # so `ModelSettings(thinking=...)` is not silently dropped by `Model.prepare_request`.
            supports_thinking=True,
            thinking_always_enabled=True,
        ),
    )
