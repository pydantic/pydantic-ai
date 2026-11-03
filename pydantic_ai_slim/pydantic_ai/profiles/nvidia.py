from __future__ import annotations as _annotations

from . import ModelProfile


def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for NVIDIA Nemotron models."""
    return ModelProfile(supports_thinking=True)
