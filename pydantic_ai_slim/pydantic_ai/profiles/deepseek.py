from __future__ import annotations as _annotations

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    is_r1 = model_name.startswith('deepseek-r1') or model_name == 'deepseek-reasoner'
    # V4 models (deepseek-v4-flash, deepseek-v4-pro, …) support thinking via reasoning_effort
    # but do not always enable it — thinking_always_enabled stays False.
    is_v4 = model_name.startswith('deepseek-v4-')
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_r1,
        supports_thinking=is_r1 or is_v4,
        thinking_always_enabled=is_r1,
    )
