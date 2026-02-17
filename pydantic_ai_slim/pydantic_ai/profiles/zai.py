from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import OpenAIModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models.

    Configures thinking/reasoning support via the `reasoning_content` field
    for models that support it (glm-5, glm-4.7, glm-4.6, glm-4.5).

    Note: Vision models (glm-4.6v, glm-4.5v) are excluded as they are not
    documented as supporting thinking mode.
    """
    model_lower = model_name.lower()
    thinking_prefixes = ('glm-5', 'glm-4.7', 'glm-4.6', 'glm-4.5')
    vision_prefixes = ('glm-4.6v', 'glm-4.5v')
    if model_lower.startswith(vision_prefixes):
        return None
    if model_lower.startswith(thinking_prefixes):
        return OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        )
    return None
