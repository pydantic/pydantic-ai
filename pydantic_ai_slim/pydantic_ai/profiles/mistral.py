from __future__ import annotations as _annotations

from . import ModelProfile


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    is_magistral = model_name.startswith('magistral')
    if is_magistral:
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
