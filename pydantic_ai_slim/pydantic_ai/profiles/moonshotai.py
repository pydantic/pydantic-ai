from __future__ import annotations as _annotations

from . import ModelProfile


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    # Kimi reasoning models (kimi-k2.5/k2.6/k2.7-code, …) accept reasoning_effort and emit
    # reasoning_content; the moonshot-v1/instruct models don't. Moonshot rejects
    # reasoning_effort='none', so thinking_always_enabled makes thinking=False omit it.
    is_reasoning = model_name.lower().startswith(('kimi-k2.5', 'kimi-k2.6', 'kimi-k2.7', 'kimi-thinking'))
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
        thinking_always_enabled=is_reasoning,
    )
