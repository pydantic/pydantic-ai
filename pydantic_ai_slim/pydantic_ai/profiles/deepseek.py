from __future__ import annotations as _annotations

from .openai import OpenAIModelProfile


def deepseek_model_profile(model_name: str) -> OpenAIModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    return OpenAIModelProfile(
        ignore_streamed_leading_whitespace='r1' in model_name,
        openai_chat_custom_reasoning_field='reasoning_content',
        # For compatibility with existing behavior. May want to change later.
        openai_chat_include_reasoning_in_request='combined',
    )
