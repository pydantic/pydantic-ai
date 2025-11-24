from __future__ import annotations as _annotations

from .openai import OpenAIModelProfile


def moonshotai_model_profile(model_name: str) -> OpenAIModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    return OpenAIModelProfile(
        ignore_streamed_leading_whitespace=True,
        openai_chat_reasoning_field='reasoning_content',
    )
