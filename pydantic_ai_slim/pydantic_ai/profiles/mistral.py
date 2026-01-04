from __future__ import annotations as _annotations

from . import ModelProfile

# Mistral reasoning models (Magistral series)
_MISTRAL_REASONING_MODELS = (
    'magistral',  # Matches magistral-small, magistral-medium, etc.
)


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    model_lower = model_name.lower()

    # Magistral models are Mistral's reasoning models with chain-of-thought capabilities
    is_reasoning = any(name in model_lower for name in _MISTRAL_REASONING_MODELS)

    if is_reasoning:
        return ModelProfile(
            supports_thinking=True,
            # Magistral supports reasoning levels (low/medium/high) but thinking can be controlled
        )

    return None
