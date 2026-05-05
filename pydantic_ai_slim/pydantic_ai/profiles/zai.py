from __future__ import annotations as _annotations

from . import ModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models.

    Marks thinking-capable models (`glm-5`, `glm-4.7`, `glm-4.6`, `glm-4.5`)
    via `supports_thinking=True`. Vision models (`glm-4.6v`, `glm-4.5v`) are
    excluded as they are not documented as supporting thinking mode.

    Provider-specific request/response shape (e.g. the `reasoning_content` field
    used by Z.AI's API) is configured in `ZaiProvider.model_profile()` so that
    other providers reusing this profile (e.g. Cerebras serving `zai-*` models)
    don't inherit Z.AI-only behavior.
    """
    model_lower = model_name.lower()
    thinking_prefixes = ('glm-5', 'glm-4.7', 'glm-4.6', 'glm-4.5')
    vision_prefixes = ('glm-4.6v', 'glm-4.5v')
    if model_lower.startswith(vision_prefixes):
        return None
    if model_lower.startswith(thinking_prefixes):
        return ModelProfile(supports_thinking=True)
    return None
