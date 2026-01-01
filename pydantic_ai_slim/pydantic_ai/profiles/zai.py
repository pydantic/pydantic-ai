from __future__ import annotations as _annotations

from . import ModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI models.

    ZAI GLM models support reasoning capabilities.
    """
    model_lower = model_name.lower()

    # ZAI GLM models support reasoning
    is_reasoning = 'glm' in model_lower

    return ModelProfile(
        supports_thinking=is_reasoning,
    )
