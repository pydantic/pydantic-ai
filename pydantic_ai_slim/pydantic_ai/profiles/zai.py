from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models.

    Configures thinking/reasoning support via the `reasoning_content` field
    for models that support it (glm-5, glm-4.7, glm-4.6).
    """
    model_lower = model_name.lower()
    if 'glm-5' in model_lower or 'glm-4.7' in model_lower or 'glm-4.6' in model_lower:
        return OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        )
    return None
