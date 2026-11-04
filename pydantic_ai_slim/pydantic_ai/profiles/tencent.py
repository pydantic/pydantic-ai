from __future__ import annotations as _annotations

from . import ModelProfile


def tencent_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for Tencent Hunyuan models."""
    return ModelProfile(supports_thinking=True)
