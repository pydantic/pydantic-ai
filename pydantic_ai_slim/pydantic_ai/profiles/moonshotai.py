from __future__ import annotations as _annotations

from . import ModelProfile


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    return ModelProfile(ignore_streamed_leading_whitespace=True)
