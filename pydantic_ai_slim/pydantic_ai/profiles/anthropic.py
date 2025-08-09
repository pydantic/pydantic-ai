from __future__ import annotations as _annotations

from . import ModelProfile


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    return ModelProfile(thinking_tags=('<thinking>', '</thinking>'))
