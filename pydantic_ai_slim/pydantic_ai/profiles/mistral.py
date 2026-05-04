from __future__ import annotations as _annotations

from . import ModelProfile

_ADJUSTABLE_REASONING_MODELS_PREFIX = {'mistral-medium', 'mistral-small'}


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    is_magistral = model_name.startswith('magistral')
    if is_magistral:
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    if any(model_name.startswith(prefix) for prefix in _ADJUSTABLE_REASONING_MODELS_PREFIX):
        return ModelProfile(supports_thinking=True, thinking_always_enabled=False)
    return None
