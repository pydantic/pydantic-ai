from __future__ import annotations as _annotations

from . import ModelProfile

# Cohere reasoning models
_COHERE_REASONING_MODELS = ('command-a-reasoning',)


def cohere_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Cohere model."""
    model_lower = model_name.lower()

    is_reasoning = any(name in model_lower for name in _COHERE_REASONING_MODELS)

    if is_reasoning:
        return ModelProfile(
            supports_thinking=True,
        )

    return None
