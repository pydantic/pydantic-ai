from __future__ import annotations as _annotations

from . import ModelProfile


def cohere_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Cohere model."""
    is_reasoning = 'reasoning' in model_name
    if is_reasoning:
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
