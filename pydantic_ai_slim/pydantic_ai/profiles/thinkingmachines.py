from __future__ import annotations as _annotations

from . import ModelProfile

_HYBRID_THINKING_MODELS = frozenset({'inkling', 'inkling-small'})


def thinkingmachines_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for a Thinking Machines Inkling model, matched by its published id.

    Inkling and Inkling Small think by default and can be turned off. Other ids return `None`.
    """
    if model_name.lower() in _HYBRID_THINKING_MODELS:
        return ModelProfile(supports_thinking=True, thinking_enabled_by_default=True)
    return None
