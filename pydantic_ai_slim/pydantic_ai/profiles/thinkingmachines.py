from __future__ import annotations as _annotations

from . import ModelProfile


def thinkingmachines_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for Thinking Machines Inkling models."""
    return ModelProfile(supports_thinking=True)
