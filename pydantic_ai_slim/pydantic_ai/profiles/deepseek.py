from __future__ import annotations as _annotations

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    return ModelProfile(ignore_streamed_leading_whitespace='r1' in model_name)
