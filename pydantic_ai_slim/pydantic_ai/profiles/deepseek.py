from __future__ import annotations as _annotations

from . import ModelProfile


def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    is_r1 = 'r1' in model_name.lower()
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_r1,
        # DeepSeek R1 models support thinking (enabled by default, can be disabled via API)
        supports_thinking=is_r1,
        # Note: R1 thinking CAN be disabled via thinking: {"type": "disabled"} in the API
    )
