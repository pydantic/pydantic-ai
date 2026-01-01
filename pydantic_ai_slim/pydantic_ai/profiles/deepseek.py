from __future__ import annotations as _annotations

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    is_r1 = 'r1' in model_name
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_r1,
        # DeepSeek R1 models have thinking always enabled (cannot be disabled)
        supports_thinking=is_r1,
        thinking_always_enabled=is_r1,  # R1 reasoning is always enabled
    )
