from __future__ import annotations as _annotations

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    is_r1 = 'r1' in model_name
    # V4 supports thinking but can toggle it via reasoning_effort, unlike R1 which is always on.
    is_v4 = model_name.startswith('deepseek-v4-')
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_r1,
        supports_thinking=is_r1 or is_v4,
        thinking_always_enabled=is_r1,
    )
