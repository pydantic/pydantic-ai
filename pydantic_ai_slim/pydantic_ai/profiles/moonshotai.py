from __future__ import annotations as _annotations

from . import ModelProfile


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    # Kimi reasoning models (`kimi-thinking-preview`, `kimi-k2-thinking`) accept request-side
    # thinking parameters and emit `reasoning_content`; the instruct/base models
    # (`moonshot-v1-*`, `kimi-k2-*-instruct`, `kimi-k2-0711-preview`, `kimi-latest`, ...) do not.
    # Matching the `-thinking` marker keeps future `*-thinking` Kimi models covered without a code
    # change and mirrors the per-model `supports_thinking` gating used by the Z.AI and OpenRouter
    # profiles. Without this flag the unified `thinking` setting is silently dropped on the wire.
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking='thinking' in model_name.lower(),
    )
